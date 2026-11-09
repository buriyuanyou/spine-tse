import argparse
import jsonlines
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='chatgpt', type=str)
    parser.add_argument("--lang", default='python', type=str)
    cfg = parser.parse_args()

    # Read Raw Data
    data = []
    # Assuming standard path as per SBLLM
    raw_path = f'./processed_data/{cfg.lang}/processed_test.jsonl'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    with jsonlines.open(raw_path) as f:
        for obj in f: data.append(obj)

    # Initialize Results Structure
    processed_data = []
    for obj in data:
        best_result = {
            'code': '', 
            'acc': 0, 
            'time': 99999, 
            'input_time': 30, # Default baseline 
            'reference_time': 30
        }
        processed_data.append({
            'idx': obj.get('idx', 0), 
            'query': obj['query'], 
            'reference': obj.get('reference',''), 
            'stop': 0, 
            'best_result': best_result, 
            'best_candidates': [], 
            'error_feedback': "" # Added for Reflexion
        })
    
    out_dir = f'../output/{cfg.lang}'
    os.makedirs(out_dir, exist_ok=True)
    
    with jsonlines.open(os.path.join(out_dir, f'initial_results_{cfg.model_name}.jsonl'), 'w') as f:
        f.write_all(processed_data)
        
    print(f"Initialized {len(processed_data)} entries.")

if __name__ == "__main__":
    main()