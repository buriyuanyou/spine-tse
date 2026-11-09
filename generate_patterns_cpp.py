import json
import os
import time
import threading
import numpy as np
import tiktoken
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

# ================= Configuration Area =================
INPUT_FILE = "path/to/cpp-simplified.jsonl"
OUTPUT_FILE = ""

OPENAI_API_KEY = ""
OPENAI_BASE_URL = "" 
CHAT_MODEL_NAME = "gpt-5.2" 
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

NUM_CLUSTERS = 882  
SAMPLES_PER_CLUSTER = 5 

MAX_WORKERS = 10

FILE_LOCK = threading.Lock()
USAGE_LOCK = threading.Lock()

TOTAL_EMBEDDING_TOKENS = 0
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_CHAT_TOKENS = 0

CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

class TreeSitterFeatureExtractor:
    def __init__(self):
        self.interesting_types = {
            'function_definition', 'for_statement', 'while_statement', 'if_statement',
            'call_expression', 'binary_expression', 'unary_expression',
            'assignment_expression', 'field_expression', 'identifier', 'return_statement',
            'subscript_expression', 'pointer_expression', 'compound_statement'
        }

    def parse(self, code_str: str) -> str:
        try:
            tree = parser.parse(bytes(code_str, "utf8"))
            cursor = tree.walk()
            return self._walk_tree(cursor, code_str)
        except Exception:
            return "(error_parsing)"

    def _walk_tree(self, cursor, code_str, depth=0):
        node = cursor.node
        node_type = node.type
        result = ""

        if node_type in self.interesting_types or "operator" in node_type:
            indent = "  " * depth
            content = ""
            
            if node_type in ['identifier', 'field_expression', 'type_identifier']:
                name = code_str[node.start_byte:node.end_byte]
                content = f": {name}"
            elif node_type == 'call_expression':
                func_node = node.child_by_field_name('function')
                if func_node:
                    func_name = code_str[func_node.start_byte:func_node.end_byte]
                    content = f" [call: {func_name}]"
                else:
                    func_name = code_str[node.child(0).start_byte:node.child(0).end_byte]
                    content = f" [call: {func_name}]"

            result += f"{indent}({node_type}{content})\n"
            
            if cursor.goto_first_child():
                result += self._walk_tree(cursor, code_str, depth + 1)
                while cursor.goto_next_sibling():
                    result += self._walk_tree(cursor, code_str, depth + 1)
                cursor.goto_parent()
        else:
            if cursor.goto_first_child():
                result += self._walk_tree(cursor, code_str, depth)
                while cursor.goto_next_sibling():
                    result += self._walk_tree(cursor, code_str, depth)
                cursor.goto_parent()

        return result

extractor = TreeSitterFeatureExtractor()

def get_embeddings_batched(client: OpenAI, texts: List[str], batch_size: int = 100) -> np.ndarray:
    global TOTAL_EMBEDDING_TOKENS
    print(f"Starting embedding generation for {len(texts)} items (Smart Chunking)...")
    
    try:
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    MAX_CTX_TOKENS = 8190
    final_embeddings = []

    def fetch_and_track_embedding(input_list):
        global TOTAL_EMBEDDING_TOKENS
        resp = client.embeddings.create(input=input_list, model=EMBEDDING_MODEL_NAME)
        with USAGE_LOCK:
            TOTAL_EMBEDDING_TOKENS += resp.usage.total_tokens
        return [d.embedding for d in resp.data]

    def process_single_text_logic(text):
        clean_text = text.replace("\n", " ")
        tokens = encoding.encode(clean_text)
        if len(tokens) <= MAX_CTX_TOKENS:
            return [clean_text], False
        chunks = []
        for i in range(0, len(tokens), MAX_CTX_TOKENS):
            chunk_tokens = tokens[i : i + MAX_CTX_TOKENS]
            chunk_str = encoding.decode(chunk_tokens)
            chunks.append(chunk_str)
        return chunks, True

    current_batch_texts = []
    for text in tqdm(texts, desc="Processing Vectors"):
        chunks, was_split = process_single_text_logic(text)
        if was_split:
            if current_batch_texts:
                try:
                    vecs = fetch_and_track_embedding(current_batch_texts)
                    final_embeddings.extend(vecs)
                    current_batch_texts = []
                except Exception:
                    final_embeddings.extend([[0.0]*3072]*len(current_batch_texts))
                    current_batch_texts = []
            try:
                chunk_vecs = fetch_and_track_embedding(chunks)
                averaged_vec = np.mean(chunk_vecs, axis=0).tolist()
                final_embeddings.append(averaged_vec)
            except Exception:
                final_embeddings.append([0.0]*3072)
        else:
            current_batch_texts.append(chunks[0])
            if len(current_batch_texts) >= batch_size:
                try:
                    vecs = fetch_and_track_embedding(current_batch_texts)
                    final_embeddings.extend(vecs)
                except Exception:
                    final_embeddings.extend([[0.0]*3072]*len(current_batch_texts))
                current_batch_texts = []

    if current_batch_texts:
        try:
            vecs = fetch_and_track_embedding(current_batch_texts)
            final_embeddings.extend(vecs)
        except Exception:
            final_embeddings.extend([[0.0]*3072]*len(current_batch_texts))

    return np.array(final_embeddings)

def cluster_diffs_embedding(data: List[Dict], client: OpenAI):
    print("Preparing text for embedding...")
    corpus = []
    for item in data:
        diff_lines = item.get('diff', [])
        diff_text = "\n".join(diff_lines)
        corpus.append(diff_text)
    X = get_embeddings_batched(client, corpus)
    print(f"Clustering {len(X)} items into {NUM_CLUSTERS} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42, batch_size=256)
    kmeans.fit(X)
    clusters = {i: [] for i in range(NUM_CLUSTERS)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(data[idx])
    return clusters

def generate_memory_unit_prompt(cluster_samples: List[Dict]) -> str:
    examples_text = ""
    
    for i, sample in enumerate(cluster_samples):
        code_v0 = sample.get('code_v0_no_empty_lines', '')
        ts_ast = extractor.parse(code_v0)
        lift_val = sample.get('runtime_lift', 1.0)
        lift_str = f"{lift_val:.2f}x"
        
        examples_text += f"""
=== Sample {i+1} (ID: {sample.get('problem_id')}) ===
[Performance]: Speedup Factor: {lift_str} (Higher is better)
[Source Code (Slow)]:
{code_v0}

[Concrete AST (Reference)]:
{ts_ast}

[Diff]:
{json.dumps(sample.get('diff', []))}
"""

    return f"""
You are an Advanced Code Optimization Architect. 
I am providing {len(cluster_samples)} examples of C++ code that share the SAME inefficiency pattern.
Note that the examples provided have high performance gains (Speedup Factor). 
Pay special attention to the changes that led to these massive speedups.

Input Data:
{examples_text}

YOUR TASK:
Synthesize a **Memory Unit** that abstracts the common pattern from these examples.
Strictly follow the structure below.

### 1. MEMORY UNIT DEFINITIONS (S, P, T, A):

* **S (Structural Pattern)**:
   - **GOAL**: A **Generalized Tree-sitter S-expression** for the SLOW code.
   - **RULE**: REMOVE specific variable names. Replace with placeholders like `(identifier) @var`.
   - **RULE**: Keep topology (structural relationships).
   - **RULE**: Keep specific methods ONLY if critical (e.g., `.push_back`).
   - Format: Single S-expression string.

* **P (Profile Signature - Dynamic)**:
   - Describe runtime characteristics (e.g., "O(N^2) complexity").
   - Identify the hotspot (e.g., "Repeated vector searching").

* **T (Trace - Provenance)**:
   - Summary of evolution path.

* **A (Algorithmic Action)**:
   - High-level strategy + Pseudo-code skeleton.
   - Explain WHY this specific change yields such high speedup (refer to the examples).

### 2. CATEGORY DEFINITIONS:

* **C1: Loops & Algorithms**: Nested loops, recursion, redundant calculations, complexity issues.
* **C2: Data Structures**: Vector vs Set/Map, inappropriate usage of STL containers.
* **C3: Memory Usage**: Excessive object creation, copying overhead (pass by value vs ref).
* **C4: String Operations**: Concatenation loops, regex, parsing.
* **C5: I/O Operations**: Input/Output bottlenecks (cin/cout vs scanf/printf).

### OUTPUT FORMAT (JSON ONLY):

{{
    "memory_unit": {{
        "S": "(for_statement body: ...)", 
        "P": "String...",
        "A": "String...",
        "T": "String..."
    }},
    "metadata": {{
        "category": "C1/C2/C3/C4/C5",
        "keywords": ["key1", "key2"],
        "confidence": "High/Medium/Low"
    }}
}}
"""

def call_llm_with_retry(client, prompt, cluster_id) -> Tuple[Dict, Dict]:
    retry_count = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Output strictly valid JSON. 'S' must be a S-expression string."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, 
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if not content: raise ValueError("Empty response")
            
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            return json.loads(content), usage_info
        except Exception as e:
            retry_count += 1
            if retry_count > 3: return None, {} # type: ignore
            time.sleep(min(2 ** retry_count, 30))

def process_single_cluster(cluster_id: int, members: List[Dict], client: OpenAI):
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_CHAT_TOKENS
    if not members: return

    sorted_members = sorted(members, key=lambda x: x.get('runtime_lift', 0), reverse=True)
    current_samples = sorted_members[:SAMPLES_PER_CLUSTER]
    
    llm_output, usage_info = call_llm_with_retry(client, generate_memory_unit_prompt(current_samples), cluster_id)
    if not llm_output: return 

    if usage_info:
        with USAGE_LOCK:
            TOTAL_PROMPT_TOKENS += usage_info["prompt_tokens"]
            TOTAL_COMPLETION_TOKENS += usage_info["completion_tokens"]
            TOTAL_CHAT_TOKENS += usage_info["total_tokens"]

    avg_lift = sum(m.get('runtime_lift', 0) for m in members) / len(members)
    max_lift = current_samples[0].get('runtime_lift', 0)
    
    final_entry = {
        "cluster_id": cluster_id,
        "sample_count": len(members),
        "avg_runtime_lift": avg_lift,
        "max_runtime_lift": max_lift,
        "memory_unit": {
            "S": llm_output["memory_unit"]["S"], 
            "P": llm_output["memory_unit"]["P"],
            "T": {
                "summary": llm_output["memory_unit"].get("T_summary", ""),
                "provenance_ids": [s.get('problem_id') for s in current_samples]
            },
            "A": llm_output["memory_unit"]["A"]
        },
        "additional_attributes": {**llm_output.get("metadata", {}), "token_usage": usage_info}
    }

    with FILE_LOCK:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write(json.dumps(final_entry, ensure_ascii=False) + "\n")

def process_clusters():
    print(f"Reading data: {INPUT_FILE}")
    all_data = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    v0 = item.get('cpu_time_v0', 0)
                    v1 = item.get('cpu_time_v1', 0)
                    item['runtime_lift'] = (v0 / v1) if (v1 and v1 != 0) else 1.0
                    all_data.append(item)
    except Exception as e:
        print(f"Error loading file: {e}"); return

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    clusters = cluster_diffs_embedding(all_data, client)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    cluster_ids = list(clusters.keys())
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_cluster, cid, clusters[cid], client) for cid in cluster_ids]
        for _ in tqdm(as_completed(futures), total=len(cluster_ids), desc="Processing Clusters"):
            pass

    print("\n" + "="*40)
    print("Token Usage Summary:")
    print(f"Total Embedding Tokens:   {TOTAL_EMBEDDING_TOKENS}")
    print(f"Total Chat Prompt Tokens: {TOTAL_PROMPT_TOKENS}")
    print(f"Total Chat Completion Tokens: {TOTAL_COMPLETION_TOKENS}")
    print(f"Total Chat Total Tokens:  {TOTAL_CHAT_TOKENS}")
    print("="*40)
    print("All processing completed!")

if __name__ == "__main__":
    process_clusters()