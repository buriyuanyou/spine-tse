import json
import random
import os

def process_pie_dataset():

    input_path = "path/to/python_splits/train.jsonl"
    output_dir = ""
    output_path = os.path.join(output_dir, "python-simplified.jsonl")

    # Define the list of fields to retain
    target_fields = [
        "problem_id",
        "cpu_time_v0",
        "cpu_time_v1",
        "memory_v0",
        "memory_v1",
        "code_v0_no_empty_lines",
        "code_v1_no_empty_lines",
        "diff",
        "measured_runtime_v0",
        "measured_runtime_v1",
        "runtime_lift"
    ]

    print(f"Reading file: {input_path} ...")
    
    all_data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File not found {input_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    total_lines = len(all_data)
    print(f"Read complete, found {total_lines} items in total.")

    sample_size = 100000
    if total_lines > sample_size:
        sampled_data = random.sample(all_data, sample_size)
        print(f"Successfully sampled {sample_size} items randomly.")
    else:
        sampled_data = all_data
        print(f"Total data is less than {sample_size}, using all available data.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Writing to new file: {output_path} ...")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in sampled_data:
            filtered_item = {key: item.get(key) for key in target_fields}
            
            f_out.write(json.dumps(filtered_item, ensure_ascii=False) + '\n')

    print("Processing completed!")

if __name__ == "__main__":
    process_pie_dataset()