import os
import re
import sys
import time
import logging
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
from time import sleep
from copy import deepcopy
import multiprocessing
import warnings

warnings.filterwarnings("ignore")

import google.generativeai as genai
from openai import OpenAI

# OpenAI Key and Base URL
OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""

openai_api_keys = [OPENAI_API_KEY] * 10

api_keys = openai_api_keys 

logger = logging.getLogger(__name__)

class PatternRetriever:
    def __init__(self, pattern_path):
        self.patterns = []
        self.pattern_vectors = None
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        self.embedding_model = "text-embedding-3-large"
        self.load_patterns(pattern_path)

    def load_patterns(self, path):
        if not os.path.exists(path):
            print(f"Warning: Pattern path {path} does not exist. Retrieval will be disabled.")
            return

        print(f"Loading patterns from {path}...")
        texts_to_embed = []
        
        with jsonlines.open(path) as reader:
            for obj in reader:
                self.patterns.append(obj)
                mu = obj.get('memory_unit', {})
                add_attr = obj.get('additional_attributes', {})
                
                # Combine Profile, Structure, and Keywords as a semantic representation.
                text_representation = f"{mu.get('P', '')} {mu.get('S', '')} {' '.join(add_attr.get('keywords', []))}"
                texts_to_embed.append(text_representation)
        
        if texts_to_embed:
            print(f"Embedding {len(texts_to_embed)} patterns using {self.embedding_model}...")
            try:
                all_embeddings = []
                batch_size = 50
                for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding Batches"):
                    batch_texts = texts_to_embed[i : i + batch_size]
                    batch_texts = [t.replace("\n", " ") for t in batch_texts]
                    
                    response = self.client.embeddings.create(
                        input=batch_texts,
                        model=self.embedding_model
                    )
                    batch_embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(batch_embeddings)
                
                self.pattern_vectors = np.array(all_embeddings)
                print(f"Loaded and embedded {len(self.patterns)} patterns.")
                
            except Exception as e:
                print(f"Error during pattern embedding: {e}")
                self.pattern_vectors = None

    def search(self, query_code, top_k=1):
        if self.pattern_vectors is None or len(self.patterns) == 0:
            return []
        
        try:
            query_code_clean = query_code.replace("\n", " ")
            
            response = self.client.embeddings.create(
                input=[query_code_clean],
                model=self.embedding_model
            )
            query_vec = np.array(response.data[0].embedding)
            
            
            norm_patterns = np.linalg.norm(self.pattern_vectors, axis=1)
            norm_query = np.linalg.norm(query_vec)
            
            if norm_query == 0:
                return []
                
            dot_products = np.dot(self.pattern_vectors, query_vec)
            
            cosine_similarities = dot_products / (norm_patterns * norm_query)
            
            top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append(self.patterns[idx])
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []

def clean(text):
    text = re.sub('[\u4e00-\u9fa5]+', '', text)
    text = re.sub('[\u3040-\u309F]+', '', text)
    text = re.sub('[\u30A0-\u30FF]+', '', text)
    text = re.sub('[\uAC00-\uD7A3]+', '', text)
    return text

format_pattern_rag = '''Task Description & Instructions:
You will be provided with a code snippet to be optimized.
To help you, we have retrieved a similar **Code Optimization Pattern** from our knowledge base.
This pattern describes a common performance issue and how to fix it using a Memory Unit structure:
- **Structure (S)**: The abstract syntax structure of the inefficient code.
- **Profile (P)**: The runtime characteristics/hotspots.
- **Action (A)**: The algorithmic strategy and pseudo-code to optimize it.
- **Trace (T)**: Record how this pattern evolved from "slow code" to "fast code".

You must strictly follow the **Four-Step Reasoning Framework** below to ensure the optimization is structurally valid and empirically grounded.

### Phase 1: Diagnostic Alignment
**Instruction:** Evaluate whether the input code resides in the retrieved Source State ($S$) and satisfies the Guard Condition ($P$).
**Action:**
1. Explicitly map the abstract structural nodes in **Structure (S)** to the concrete variables and control flows in the input code.
2. Verify if the specific performance bottlenecks described in **Profile (P)** (e.g., complexity class, I/O overhead) are present in the input code.
*Constraint:* If the structure or profile does not align, identify the discrepancy before proceeding.

### Phase 2: Strategy Instantiation
**Instruction:** Execute the Transition Function ($A$) to create a concrete implementation plan.
**Action:**
1. Act as an interpreter for the **Action (A)**. Bind the abstract placeholders in the pseudo-code/strategy to the specific variable names in the input code.
2. Translate the high-level algorithmic directive into a context-specific plan.

### Phase 3: Evolutionary Validation
**Instruction:** Validate your plan against the historical **Trace (T)**.
**Action:**
1. Review the evolutionary path in **Trace (T)** (how similar code evolved from slow to fast).
2. Confirm that your instantiated plan from Phase 2 aligns with this proven trajectory. Ensure the logic is consistent with historical success cases to mitigate invalid transformations.

### Phase 4: Code Synthesis
**Instruction:** Realize the transition to the target state.
**Action:**
1. Generate the final optimized code snippet.
2. Ensure the code is functionally equivalent but strictly follows the optimized logic derived in Phase 2 and validated in Phase 3.

Desired output format:
1. Analysis of original code: <answer>
2. Application of the Optimization Pattern: <answer>
3. Optimization opportunities: <answer>
4. Optimized Code: <optimization points> ```{}
<code>
```
'''

format = '''Task Desctiption & Instructions: 
You will be provided with a code snippet. Your task is to propose a more efficient optimization method to achieve a higher speedup rate.

Desired output format:
1. Analyze the original code: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```{}\n<code>\n```
'''

def prompt_construction(cfg, queries, retriever=None):
    results = []
    if cfg.iteration > 0:
        try:
            with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')) as f:
                for obj in f:
                    results.append(obj)
        except FileNotFoundError:
            pass 

    prompt = []
    ids = set()
    
    results_map = {res['idx']: res for res in results}

    print(f"Constructing prompts and performing retrieval for {len(queries)} queries...")
    for query in tqdm(queries, desc="Retrieval"):
        idx = query['idx']
        ids.add(idx)
        
        previous_performance = ''
        if idx in results_map:
            pperf = results_map[idx]
            if pperf.get('stop', 0) >= 1:
                continue
                
            if 'best_candidates' in pperf:
                for cand_idx, opt in enumerate(pperf['best_candidates'][:cfg.beam_number]):
                    if opt['acc'] == 1 and opt['input_time']/opt['time'] > 1:
                        previous_performance += 'A correct and optimized version {}:'.format(str(cand_idx+1))
                    elif opt['acc'] == 1 and opt['input_time']/opt['time'] <= 1:
                        previous_performance += 'A correct but unoptimized version {}:'.format(str(cand_idx+1))
                    elif opt['acc'] < 1:
                        previous_performance += 'An incorrect optimized version {}:'.format(str(cand_idx+1))
                    previous_performance += '\n{}\nAccuracy: {} Speedup Rate: {}\n'.format(opt['content'].strip(), str(opt['acc']), str(round(opt['input_time']/opt['time'], 2)))

        retrieved_context = ""
        if retriever:
            top_patterns = retriever.search(query['query'], top_k=1)
            if top_patterns:
                p = top_patterns[0]
                mu = p.get('memory_unit', {})

                trace_info = mu.get('T', {})
                if isinstance(trace_info, dict):
                    trace_str = trace_info.get('summary', 'N/A')
                else:
                    trace_str = str(trace_info)

                retrieved_context = (
                    f"**Retrieved Optimization Pattern**:\n"
                    f"Profile (P): {mu.get('P', 'N/A')}\n"
                    f"Structure (S): {mu.get('S', 'N/A')}\n"
                    f"Action (A): {mu.get('A', 'N/A')}\n"
                    f"Trace (T): {trace_str}\n"
                )
                query['retrieved_pattern_id'] = p.get('cluster_id')

        user_content = ""
        
        target_lang = "cpp" if cfg.lang == "cpp" else "python"
        
        if retrieved_context:
            user_content = format_pattern_rag.format(target_lang) + \
                           f"\nThe code you need to optimize:\n```{target_lang}\n{query['query']}\n```\n" + \
                           f"{retrieved_context}\n"
            if previous_performance:
                 user_content += f"Some existing versions with their performance:\n{previous_performance}\n"
            
            user_content += "Please follow the above instructions and output format specification step-by-step to generate a better program.\n"
        else:
            user_content = format.format(target_lang) + \
                           f"The code you need to optimize:\n```{target_lang}\n{query['query']}\n```\n" 
            if previous_performance:
                user_content += f"Some existing versions with their performance:\n{previous_performance}\n"
            user_content += "Please follow the above instructions and output format specification step-by-step to generate a better program.\n"

        query['prompt_chat'] = [
            {"role": "user", "content": user_content}
        ]
        
        prompt.append(query)
    
    return ids, prompt

# OpenAI Official Function
def openai_official(chunk_data):
    cfg = chunk_data['cfg']
    queries = chunk_data['queries']
    model = "gpt-5.2"
    
    client = OpenAI(
        api_key=api_keys[cfg.api_idx],
        base_url=OPENAI_BASE_URL
    )
    
    multi_fail_count = 0 
    exceed_time = 0
    key_suffix = api_keys[cfg.api_idx][-4:] if len(api_keys[cfg.api_idx]) > 4 else "Show"
    print(f"Worker {cfg.api_idx} started with key ending in ...{key_suffix}")

    for pos in tqdm(range(len(queries))):
        query = queries[pos]
        success = 0
        fail_count = 0
        
        query['token_usage'] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

        while success != 1:
            messages = [
                {"role": "system", "content": "You are a software developer and now you will help to improve code efficiency. Please follow the instructions and output format specification to generate a more efficient code. The improved code should be in code blocks (```{} ```).".format(cfg.lang)},
            ]
            if 'prompt_chat' in query:
                messages.extend(query['prompt_chat'])
            
            try:    
                batch_preds = []
                batch_detailed_preds = []
                
                for _ in range(cfg.generation_number):
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages, # type: ignore 
                        n=1,  
                        temperature=0.7
                    )
                    
                    choice = response.choices[0]
                    content = choice.message.content
                    if content is not None:
                        batch_preds.append(content.strip())
                        batch_detailed_preds.append(content)
                    else:
                        batch_preds.append("")
                        batch_detailed_preds.append("")
                    
                    if response.usage:
                        query['token_usage']['prompt_tokens'] += response.usage.prompt_tokens
                        query['token_usage']['completion_tokens'] += response.usage.completion_tokens
                        query['token_usage']['total_tokens'] += response.usage.total_tokens
                
                success = 1
                query['prediction'] = batch_preds
                query['detailed_prediction'] = batch_detailed_preds

                if 'prompt_chat' in query:
                    del query['prompt_chat']
                
                output_file = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(cfg.api_idx),str(len(api_keys)), 'test'))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                with jsonlines.open(output_file, "a") as f:
                    f.write_all([query])
                
                sleep(0.02)

            except Exception as e:
                info = str(e)
                fail_count += 1
                if 'insufficient_quota' in info or 'billing' in info:
                    print(f"Insufficient balance for key {cfg.api_idx}")
                    break 
                elif 'Rate limit' in info:
                    sleep(2 * fail_count)
                elif 'context_length_exceeded' in info:
                    model = "gpt-5.2"
                    exceed_time += 1
                    fail_count = 0 
                else:
                    print(f"Error occurred: {info}")
                    sleep(1)
            
            if fail_count > 10:
                print(f"Failed too many times at pos {pos}")
                if 'prediction' not in query:
                    query['prediction'] = []
                multi_fail_count += 1
                break
                
    print(f'Worker {cfg.api_idx} exceed_time: {exceed_time}')

def openai_llama(chunk_data):
    pass 

def openai_gemini(chunk_data):
    pass

def read_file(cfg):
    queries = []
    
    with jsonlines.open(cfg.baseline_data_path) as reader:
        for obj in reader:
            queries.append(obj)
    
    if cfg.limit > 0:
        print(f"Limit is set. Processing only the first {cfg.limit} items.")
        queries = queries[:cfg.limit]
    
    start_idx = int(len(queries) * (cfg.slice - 1) / cfg.total)
    end_idx = int(len(queries) * cfg.slice / cfg.total)
    queries = queries[start_idx:end_idx]

    unsort_queries = []
    for num in range(len(api_keys)):
        fname = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))
        if os.path.exists(fname):
            with jsonlines.open(fname) as f:
                for obj in f:
                    unsort_queries.append(obj)
    
    if len(unsort_queries) > 0:
        main_file_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))
        with jsonlines.open(main_file_path, 'a') as f: 
             f.write_all(unsort_queries)

    exist_queries = set()
    main_pred_file = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))
    
    if os.path.exists(main_pred_file):
        with jsonlines.open(main_pred_file) as f:
            for obj in f:
                exist_queries.add(obj['idx'])

    new_queries = []
    for query in queries:
        if query['idx'] not in exist_queries:
            new_queries.append(query)
            
    if len(new_queries) == 0:
        return exist_queries, [], len(exist_queries)
    
    queries = new_queries
    retriever = None
    if cfg.pattern_path and os.path.exists(cfg.pattern_path):
        retriever = PatternRetriever(cfg.pattern_path)
    
    ids, queries = prompt_construction(cfg, queries, retriever)

    print('{} codes will be optimized in iteration {}'.format(len(queries), cfg.iteration))
        
    return ids, queries, len(exist_queries)

def separate_lines(code):
    lines = code.split('\n')
    comments = []
    code_lines = []
    start = False
    for line in lines:
        if not start and line.strip().startswith('#'):
            comments.append(line.strip()[1:].strip())
        else:
            start = True
            code_lines.append(line)
    return ' '.join(comments), '\n'.join(code_lines)

def post_process(cfg, ids):
    data = []
    pred_file = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))
    
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        return

    with jsonlines.open(pred_file) as f:
        for obj in f:
            data.append(obj)
    
    count = 0
    processed_data = []
    
    for obj in data:
        if obj['idx'] not in ids:
            continue
        
        if cfg.lang == 'python':
            post_prediction = []
            detailed_prediction = []
            preds_source = obj.get('prediction', [])
            
            for res in preds_source:
                detailed_prediction.append(res)
                if '```python' in res:
                    code_block = res.split('```python')[-1].split('```')[0]
                    post_prediction.append(code_block)
                elif '```' in res:
                    parts = res.split('```')
                    if len(parts) >= 3:
                        post_prediction.append(parts[1])
                    else:
                         post_prediction.append(res)
                else:
                    r, c = separate_lines(res)
                    post_prediction.append(c)

        else: # CPP
            post_prediction = []
            detailed_prediction = []
            preds_source = obj.get('prediction', [])
            for res in preds_source:
                detailed_prediction.append(res)
                if '```cpp' in res:
                    post_prediction.append(res.split('```cpp')[-1].split('```')[0])
                elif '```' in res:
                    parts = res.split('```')
                    if len(parts) >= 3:
                        post_prediction.append(parts[1])
                    else:
                        post_prediction.append(res)
                else:
                    post_prediction.append(res)
        
        obj['prediction'] = post_prediction
        if 'token_usage' not in obj:
            obj['token_usage'] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            
        processed_data.append(obj)
        
    with jsonlines.open(os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'tmp_queries_{}test.jsonl'.format(cfg.model_name)), 'w') as f:
        f.write_all(processed_data)
        
    print(f"Post-processed {len(processed_data)} items.")
    
    for num in range(len(api_keys)):
        fname = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}_{}_{}.jsonl'.format(cfg.model_name,str(num),str(len(api_keys)),'test'))
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_file", action='store_true')
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--generation_path", default=None, type=str, required=True)
    parser.add_argument("--baseline_data_path", default='None', type=str)
    parser.add_argument("--training_data_path", default='None', type=str)
    parser.add_argument("--lang", default='None', type=str)
    parser.add_argument("--api_idx", default=0, type=int)
    parser.add_argument("--iteration", default=1, type=int)
    parser.add_argument("--restart_pos", default=0, type=int)
    parser.add_argument("--beam_number", default=5, type=int)
    parser.add_argument("--generation_number", default=5, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--slice", default=1, type=int)
    parser.add_argument("--total", default=1, type=int)
    
    parser.add_argument("--pattern_path", default=None, type=str, help="Path to the pattern library jsonl file")
    
    parser.add_argument("--limit", default=-1, type=int, help="Limit the number of queries to process (for debugging). -1 means all.")

    cfg = parser.parse_args()
    
    out_dir = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fail_count = 0
    
    if cfg.model_name == 'chatgpt':
        api_keys = openai_api_keys
        
    while True:
        ids, queries, previous_length = read_file(cfg)
        
        if len(queries) == 0:
            print("No new queries to process. Running post-process...")
            post_process(cfg, ids)
            sys.exit(0)

        if cfg.model_name == 'gemini' or cfg.model_name == 'codellama':
            print("Model not fully implemented in this refactor, strictly following OpenAI path.") 
            pass
        
        print(f'Number of queries in round {fail_count}: {len(queries)}')
        
        chunks_query = np.array_split(queries, len(api_keys))
        chunks_data = []
        for i in range(len(api_keys)):
            if len(chunks_query[i]) == 0:
                continue
            tmp_data = {}
            tmp_data['cfg'] = deepcopy(cfg)
            tmp_data['cfg'].api_idx = i
            tmp_data['queries'] = chunks_query[i].tolist() 
            chunks_data.append(tmp_data)
        
        if not chunks_data:
            break

        pool = multiprocessing.Pool(len(chunks_data))
        
        if cfg.model_name == 'codellama':
            results = pool.map(openai_llama, chunks_data)
        elif cfg.model_name == 'gemini':
             results = pool.map(openai_gemini, chunks_data)
        else:
            results = pool.map(openai_official, chunks_data)
            
        pool.close()
        pool.join()

        data = []
        pred_file = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), 'prediction_{}_{}.jsonl'.format(cfg.model_name,'test'))
        
        if os.path.exists(pred_file):
            with jsonlines.open(pred_file) as f:
                for i in f:
                    data.append(i)
        
        current_total = len(data)
        target_total = len(ids)
        
        if current_total >= (previous_length + len(queries)):
            print("All tasks completed.")
            post_process(cfg, list(ids))
            sys.exit(0)
        else:
            print(f"Progress: {current_total} / {previous_length + len(queries)}")
            fail_count += 1
            if fail_count > 5:
                print("Too many failures. Exiting.")
                sys.exit(-1)