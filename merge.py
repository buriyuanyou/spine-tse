import os
import jsonlines
import argparse
from tqdm import tqdm
from utils import SimFixDiffer

def selection(candidates, beam_number, differ):
    valid_cands = []
    invalid_cands = []
    
    for c in candidates:
        if c['acc'] == 1:
            valid_cands.append(c)
        else:
            invalid_cands.append(c)
            
    valid_cands.sort(key=lambda x: x['time'])
    
    selected = valid_cands[:beam_number]
    
    if len(selected) < beam_number:
        selected.extend(invalid_cands[:(beam_number - len(selected))])
        
    return selected

def main(cfg):
    report_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, str(cfg.iteration), f'test_execution_{cfg.model_name}test.report')
    
    eval_results_map = {} 
    
    token_stats = {
        'prompt_tokens': 0, 
        'completion_tokens': 0, 
        'total_tokens': 0
    }

    if os.path.exists(report_path):
        with jsonlines.open(report_path) as f:
            for obj in f:
                if 'token_usage' in obj:
                    token_stats['prompt_tokens'] += obj['token_usage'].get('prompt_tokens', 0)
                    token_stats['completion_tokens'] += obj['token_usage'].get('completion_tokens', 0)
                    token_stats['total_tokens'] += obj['token_usage'].get('total_tokens', 0)

                key = obj.get('idx')
                if key is None:
                    key = obj['code_v0_no_empty_lines'].strip()
                
                if key not in eval_results_map:
                    eval_results_map[key] = []
                eval_results_map[key].append(obj)
    else:
        print(f"Warning: Report file not found at {report_path}")

    print(f"Iteration {cfg.iteration} Token Usage: {token_stats}")
    token_log_dir = os.path.join(cfg.generation_path, cfg.lang, cfg.mode)
    if not os.path.exists(token_log_dir):
        os.makedirs(token_log_dir, exist_ok=True)
    token_log_path = os.path.join(token_log_dir, 'token_usage.log')
    
    with open(token_log_path, 'a') as log_f:
        log_f.write(f"Iteration {cfg.iteration}: {token_stats}\n")

    results_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')
    if not os.path.exists(results_path):
        results_path = os.path.join(cfg.generation_path, cfg.lang, 'initial_results_{}.jsonl'.format(cfg.model_name))
        
    history_data = []
    if os.path.exists(results_path):
        with jsonlines.open(results_path) as f:
            for obj in f: history_data.append(obj)
    else:
        print(f"Error: History file not found at {results_path}")
        return

    if cfg.limit > 0:
        print(f"Limit is set. Merging only the first {cfg.limit} items.")
        history_data = history_data[:cfg.limit]

    updated_history = []
    differ = SimFixDiffer()
    
    for obj in tqdm(history_data, desc="Merging"):
        current_candidates_raw = []
        
        if obj.get('idx') in eval_results_map:
            current_candidates_raw = eval_results_map[obj['idx']]
        elif obj.get('query','').strip() in eval_results_map:
             current_candidates_raw = eval_results_map[obj['query'].strip()]
        
        current_candidates = []
        for perf in current_candidates_raw:
            cand = {
                'code': perf['model_generated_potentially_faster_code_col'],
                'acc': perf['model_generated_potentially_faster_code_col_acc'],
                'time': perf['model_generated_potentially_faster_code_col_time_mean'],
                'error_msg': perf.get('error_msg', ''),
                'content': perf['model_generated_potentially_faster_code_col']
            }
            if 'retrieved_pattern_id' in perf:
                cand['retrieved_pattern_id'] = perf['retrieved_pattern_id']
                
            current_candidates.append(cand)
        
        if not current_candidates:
            updated_history.append(obj)
            continue

        if 'best_candidates' not in obj: obj['best_candidates'] = []
        obj['best_candidates'].extend(current_candidates)
        
        obj['best_candidates'] = selection(obj['best_candidates'], cfg.beam_number, differ)
        
        if obj['best_candidates']:
            best_cand = obj['best_candidates'][0]
            current_best_time = obj['best_result']['time']
            
            if best_cand['acc'] == 1 and best_cand['time'] < current_best_time:
                obj['best_result'] = {
                    'code': best_cand['code'],
                    'acc': 1,
                    'time': best_cand['time'],
                    'input_time': obj['best_result']['input_time'], 
                    'reference_time': obj['best_result']['reference_time']
                }
                obj['error_feedback'] = ""
            elif best_cand['acc'] == 0:
                obj['error_feedback'] = best_cand.get('error_msg', "")

        updated_history.append(obj)

    target_path = os.path.join(cfg.generation_path, cfg.lang, cfg.mode, 'results.jsonl')
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with jsonlines.open(target_path, 'w') as f:
        f.write_all(updated_history)
        
    print(f"Merged {len(updated_history)} queries. Saved to {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--generation_number", type=int, default=1)
    parser.add_argument("--lang", type=str, default="python")
    parser.add_argument("--model_name", type=str, default="chatgpt")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--training_data_path", type=str, default="")
    parser.add_argument("--generation_path", type=str, required=True)
    parser.add_argument("--beam_number", type=int, default=1)
    
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of merged items for debugging")
    
    cfg = parser.parse_args()
    main(cfg)