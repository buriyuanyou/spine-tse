import os
import argparse
import logging
import random
import numpy as np
import ast
import difflib
import cProfile
import pstats
import io
import time
import copy
import json
import timeit
import asyncio
from collections import Counter
from scipy.stats import linregress
from typing import List, Dict, Any, Tuple, Optional, cast
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

API_KEY = ""
CUSTOM_BASE_URL = ""

client = AsyncOpenAI(
    base_url=CUSTOM_BASE_URL,
    api_key=API_KEY
)

def cfg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--lang", default='python', type=str)
    parser.add_argument("--output_path", default=None, type=str)
    parser.add_argument("--generation_path", default=None, type=str)
    parser.add_argument("--baseline_data_path", default='', type=str)
    parser.add_argument("--training_data_path", default='', type=str)
    parser.add_argument("--test_case_path", default='', type=str)
    
    parser.add_argument("--process_number", default=20, type=int)
    parser.add_argument("--api_idx", default=0, type=int)
    
    parser.add_argument("--model_name", default='chatgpt', type=str)
    
    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument("--generation_number", default=2, type=int)
    parser.add_argument("--beam_number", default=3, type=int)
    parser.add_argument("--from_file", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--slice", default=1, type=int)
    parser.add_argument("--total", default=1, type=int)
    parser.add_argument("--testing_number", default=0, type=int)
    parser.add_argument("--restart_pos", default=0, type=int)

    cfg = parser.parse_args()
    return cfg

async def call_llm_api(prompt: str, is_json: bool = False, max_retries: int = 3) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    create_args = {
        "model": "gpt-5.2", 
        "messages": messages,
        "timeout": 120
    }
    if is_json:
        create_args["response_format"] = {"type": "json_object"}

    delay = 1
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**create_args)
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"  [API Error] Attempt {attempt + 1}: {e}")
            if attempt + 1 == max_retries: return None
            await asyncio.sleep(delay)
            delay *= 2
    return None

class SimFixDiffer:
    def __init__(self):
        self.mappings = {}

    def compute_diff_score(self, code_a: str, code_b: str) -> float:
        try:
            tree_a = ast.parse(code_a)
            tree_b = ast.parse(code_b)
            self.mappings = {}
            self.match(tree_a, tree_b)
            nodes_a = sum(1 for _ in ast.walk(tree_a))
            nodes_b = sum(1 for _ in ast.walk(tree_b))
            matched_count = len(self.mappings)
            return 1.0 - (2 * matched_count) / (nodes_a + nodes_b + 1e-5)
        except:
            return 1.0 

    def match(self, a, b):
        if type(a) == type(b):
            self.mappings[a] = b
            self.match_children(a, b)
        else:
            for child_b in ast.iter_child_nodes(b):
                self.match(a, child_b)

    def match_children(self, a, b):
        children_a = list(ast.iter_child_nodes(a))
        children_b = list(ast.iter_child_nodes(b))
        matcher = difflib.SequenceMatcher(None, [type(n).__name__ for n in children_a], [type(n).__name__ for n in children_b])
        for block in matcher.get_matching_blocks():
            for i in range(block.size):
                self.match(children_a[block.a + i], children_b[block.b + i])

class ASTFeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.features = Counter()
        self.max_loop_depth = 0
        self.loop_depth = 0

    def visit_For(self, node):
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.features['loops'] += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.features['loops'] += 1
        self.generic_visit(node)
        self.loop_depth -= 1
    
    def visit_Call(self, node):
        self.features['calls'] += 1
        if isinstance(node.func, ast.Name) and node.func.id in ['sum', 'min', 'max', 'sorted']:
            self.features['expensive'] += 1
        self.generic_visit(node)

    def get_report(self) -> str:
        res = []
        if self.max_loop_depth > 1: res.append(f"Deeply Nested Loops (Depth {self.max_loop_depth})")
        if self.features['expensive'] > 0: res.append("Potential O(N) calls inside logic")
        if self.features['loops'] == 0: res.append("No explicit loops found")
        return ", ".join(res) if res else "Flat structure"

class FunctionalVerifier:
    @staticmethod
    async def generate_harness(code_content: str) -> Dict[str, Any] | None:
        """
        Generates a benchmark harness for the given code using LLM.
        """
        prompt = f"""
Create a Python performance benchmark harness.
Return JSON: {{"entry_point": "func", "data_generator_code": "def generate_data(n):...", "test_scales": [100, 1000]}}
Make sure `generate_data` returns a tuple of arguments.
Code:
```python
{code_content}
""" 
        res = await call_llm_api(prompt, is_json=True)
        if not res: return None
        try:
            return json.loads(res)
        except:
            return None

    @staticmethod
    def verify(original_code: str, new_code: str, harness: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            local_scope_orig = {}
            local_scope_new = {}
            exec(original_code, globals(), local_scope_orig)
            exec(harness['data_generator_code'], globals(), local_scope_orig)
            exec(new_code, globals(), local_scope_new)
            exec(harness['data_generator_code'], globals(), local_scope_new)

            ep = harness['entry_point']
            def get_func(scope):
                if '.' in ep:
                    c, m = ep.split('.')
                    if c in scope:
                        try: instance = scope[c]()
                        except: return None
                        return getattr(instance, m, None)
                return scope.get(ep)

            func_orig = get_func(local_scope_orig)
            func_new = get_func(local_scope_new)
            data_gen = local_scope_orig.get('generate_data')

            if not (callable(func_orig) and callable(func_new) and callable(data_gen)):
                return False, "Entry point not callable"

            raw_args = data_gen(50) # small scale for verify
            if not isinstance(raw_args, (list, tuple)): raw_args = (raw_args,)
            
            try: res_orig = func_orig(*copy.deepcopy(raw_args))
            except: return True, "" # Skip if original fails

            try: res_new = func_new(*copy.deepcopy(raw_args))
            except Exception as e: return False, f"Runtime Error: {e}"

            is_equal = False
            if 'numpy' in str(type(res_orig)) or 'ndarray' in str(type(res_orig)):
                if isinstance(res_orig, (int, float, complex)):
                    is_equal = np.allclose(cast(Any, res_orig), cast(Any, res_new))
                else:
                    is_equal = np.array_equal(cast(Any, res_orig), cast(Any, res_new))
            else:
                is_equal = (res_orig == res_new)

            return (True, "") if is_equal else (False, "Output Mismatch")
        except Exception as e:
            return False, f"System Error: {e}"

async def run_profile(code_string: str, harness: Dict[str, Any]) -> str:
    """ Runs cProfile on the code to find hotspots. """
    try:
        local_scope = {}
        exec(code_string, globals(), local_scope)
        exec(harness['data_generator_code'], globals(), local_scope)
        data_gen = local_scope.get('generate_data')
        ep = harness['entry_point']

        func = local_scope.get(ep)
        if '.' in ep:
            c, m = ep.split('.')
            if c in local_scope: func = getattr(local_scope[c](), m)
            
        if not func or not callable(data_gen): return ""

        args = data_gen(100)
        if not isinstance(args, (list, tuple)): args = (args,)
        
        profiler = cProfile.Profile()
        profiler.enable()
        func(*args)
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(5)
        return s.getvalue()
    except:
        return ""

async def run_benchmark(code: str, harness: Dict[str, Any]) -> float | None:
    """ Calculates time complexity slope k. """
    try:
        local_scope = {}
        exec(code, globals(), local_scope)
        exec(harness['data_generator_code'], globals(), local_scope)
        data_gen = local_scope.get('generate_data')
        ep = harness['entry_point']

        func = local_scope.get(ep)
        if '.' in ep:
            c, m = ep.split('.')
            if c in local_scope: func = getattr(local_scope[c](), m)
            
        if not func or not callable(data_gen): return None
        
        log_n: List[float] = []
        log_y: List[float] = []
        
        for n in harness.get('test_scales', [100, 1000, 2000]):
            args = data_gen(n)
            if not isinstance(args, (list, tuple)): args = (args,)
            
            try:
                func(*args) # Warmup
                t0 = timeit.default_timer()
                func(*args)
                dur = timeit.default_timer() - t0
                if dur > 0:
                    log_n.append(float(np.log(n)))
                    log_y.append(float(np.log(dur)))
            except:
                continue
        
        if len(log_n) < 2: return None
        
        slope, _, _, _, _ = linregress(log_n, log_y)
        return float(slope) # type: ignore
    except: 
        return None