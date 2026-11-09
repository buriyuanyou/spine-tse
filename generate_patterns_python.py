import json
import os
import time
import threading
import numpy as np
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# ================= Configuration Area =================
INPUT_FILE = "path/to/python-simplified.jsonl"
OUTPUT_FILE = ""

OPENAI_API_KEY = ""
OPENAI_BASE_URL = "" 
CHAT_MODEL_NAME = "gpt-5.2"  
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

NUM_CLUSTERS = 369  
SAMPLES_PER_CLUSTER = 5 

MAX_WORKERS = 10

FILE_LOCK = threading.Lock()
USAGE_LOCK = threading.Lock()

TOTAL_EMBEDDING_TOKENS = 0
TOTAL_CHAT_PROMPT_TOKENS = 0
TOTAL_CHAT_COMPLETION_TOKENS = 0
TOTAL_CHAT_TOTAL_TOKENS = 0

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class TreeSitterFeatureExtractor:
    def __init__(self):
        self.interesting_types = {
            'function_definition', 'for_statement', 'while_statement', 'if_statement',
            'call_expression', 'binary_operator', 'list_comprehension', 
            'assignment', 'attribute', 'identifier', 'return_statement',
            'subscript', 'comparison_operator', 'aug_assign'
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
            
            if node_type in ['identifier', 'attribute']:
                name = code_str[node.start_byte:node.end_byte]
                content = f": {name}"
            elif node_type == 'call_expression':
                func_node = node.child_by_field_name('function')
                if func_node:
                    func_name = code_str[func_node.start_byte:func_node.end_byte]
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
    print(f"Starting embedding generation for {len(texts)} items...")
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch = texts[i : i + batch_size]
        try:
            batch = [text.replace("\n", " ") for text in batch]
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL_NAME
            )
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                with USAGE_LOCK:
                    TOTAL_EMBEDDING_TOKENS += response.usage.total_tokens
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            embeddings.extend([[0.0] * 3072] * len(batch)) 

    return np.array(embeddings)

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
        lift = sample.get('runtime_lift', 1.0)
        
        examples_text += f"""
=== Sample {i+1} (ID: {sample.get('problem_id')}) ===
[Performance]: Speedup Factor: {lift:.2f}x (Higher is better)
[Source Code (Slow)]:
{code_v0}

[Concrete AST (Reference)]:
{ts_ast}

[Diff]:
{json.dumps(sample.get('diff', []))}
"""

    return f"""
You are an Advanced Code Optimization Architect. 
I am providing {len(cluster_samples)} examples of Python code that share the SAME inefficiency pattern.
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
   - **RULE**: Keep specific methods ONLY if critical (e.g., `.append`).
   - Format: Single S-expression string.

* **P (Profile Signature - Dynamic)**:
   - Describe runtime characteristics (e.g., "O(N^2) complexity").
   - Identify the hotspot (e.g., "Repeated list searching").

* **T (Trace - Provenance)**:
   - Summary of evolution path.

* **A (Algorithmic Action)**:
   - High-level strategy + Pseudo-code skeleton.
   - Explain WHY this specific change yields such high speedup (refer to the examples).

### 2. CATEGORY DEFINITIONS:

* **C1: Loops & Algorithms**: Nested loops, recursion, redundant calculations, complexity issues.
* **C2: Data Structures**: List vs Set/Dict, inappropriate usage.
* **C3: Memory Usage**: Excessive object creation, copying overhead.
* **C4: String Operations**: Concatenation loops, regex, parsing.
* **C5: I/O Operations**: Input/Output bottlenecks.

### OUTPUT FORMAT (JSON ONLY):

{{
    "memory_unit": {{
        "S": "(for_statement body: ...)",
        "P": "String...",
        "T": "String..."
        "A": "String...",
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
            
            # Extract usage info
            usage_data = {}
            if hasattr(response, 'usage') and response.usage:
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return json.loads(content), usage_data
        except Exception as e:
            retry_count += 1
            if retry_count > 3:
                print(f"\n[Cluster {cluster_id}] Failed after retries.")
                return None, {} # type: ignore
            wait_time = min(2 ** retry_count, 30)
            print(f"\n[Cluster {cluster_id}] Error (Attempt {retry_count}): {e}")
            time.sleep(wait_time)

def process_single_cluster(cluster_id: int, members: List[Dict], client: OpenAI):
    global TOTAL_CHAT_PROMPT_TOKENS, TOTAL_CHAT_COMPLETION_TOKENS, TOTAL_CHAT_TOTAL_TOKENS
    if not members:
        return

    sorted_members = sorted(
        members, 
        key=lambda x: x.get('runtime_lift', 0), 
        reverse=True
    )

    current_samples = sorted_members[:SAMPLES_PER_CLUSTER]
    provenance_ids = [s.get('problem_id') for s in current_samples]

    llm_output, usage_info = call_llm_with_retry(client, generate_memory_unit_prompt(current_samples), cluster_id)

    if not llm_output:
        return 

    if usage_info:
        with USAGE_LOCK:
            TOTAL_CHAT_PROMPT_TOKENS += usage_info.get("prompt_tokens", 0)
            TOTAL_CHAT_COMPLETION_TOKENS += usage_info.get("completion_tokens", 0)
            TOTAL_CHAT_TOTAL_TOKENS += usage_info.get("total_tokens", 0)

    additional_attrs = llm_output.get("metadata", {})
    additional_attrs["token_usage"] = usage_info

    final_entry = {
        "cluster_id": cluster_id,
        "sample_count": len(members),
        "avg_runtime_lift": sum(m.get('runtime_lift', 0) for m in members) / len(members),
        "max_runtime_lift": current_samples[0].get('runtime_lift', 0),
        "memory_unit": {
            "S": llm_output["memory_unit"]["S"],
            "P": llm_output["memory_unit"]["P"], 
            "T": {
                "summary": llm_output["memory_unit"].get("T_summary", ""),
                "provenance_ids": provenance_ids
            },
            "A": llm_output["memory_unit"]["A"]
        },
        "additional_attributes": additional_attrs
    }

    with FILE_LOCK:
        try:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(final_entry, ensure_ascii=False) + "\n")
                f_out.flush()
        except Exception as e:
            print(f"Error writing cluster {cluster_id} to file: {e}")

    return cluster_id

def process_clusters():
    print(f"Reading data: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print("File not found")
        return

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    clusters = cluster_diffs_embedding(all_data, client)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Writing results to: {OUTPUT_FILE}")
    print(f"Starting concurrent processing, Max Workers: {MAX_WORKERS}")

    cluster_ids = list(clusters.keys())
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_cluster = {
            executor.submit(process_single_cluster, cid, clusters[cid], client): cid 
            for cid in cluster_ids
        }
        
        for future in tqdm(as_completed(future_to_cluster), total=len(cluster_ids), desc="Concurrent Processing"):
            cluster_id = future_to_cluster[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Cluster {cluster_id} generated an exception: {exc}")

    print("\n" + "="*30)
    print("All processing completed!")
    print(f"Embedding Token Usage: {TOTAL_EMBEDDING_TOKENS}")
    print(f"Chat Prompt Tokens: {TOTAL_CHAT_PROMPT_TOKENS}")
    print(f"Chat Completion Tokens: {TOTAL_CHAT_COMPLETION_TOKENS}")
    print(f"Chat Total Tokens: {TOTAL_CHAT_TOTAL_TOKENS}")
    print("="*30)

if __name__ == "__main__":
    if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
        print("Please check your OPENAI_API_KEY configuration.")
    else:
        process_clusters()