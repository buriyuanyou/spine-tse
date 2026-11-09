lang=python
private_test_case_path=./processed_data/generated_test_cases
training_data_path=./processed_data/$lang/processed_train.jsonl
model_name=chatgpt

generation_number=5

baseline_data_path=./processed_data/$lang/processed_test.jsonl
output_path=./output/$lang

# Path to Pattern Library
pattern_path=./patterns/patterns-python.jsonl
# cpp
# pattern_path=./patterns/patterns-cpp.jsonl

debug_limit=-1
base_mode=python_oneshot
generation_path=./output
api_idx=0
restart_pos=0

if [ ! -d "./output/$lang/$base_mode/0" ]; then
  mkdir -p ./output/$lang/$base_mode/1
  python initial.py --model_name $model_name --lang $lang
  cp ./output/$lang/initial_results_$model_name.jsonl ./output/$lang/$base_mode/results.jsonl
fi

i=0
mode=$base_mode

echo "--- One-Shot Generation ---"

python generate.py  --from_file --mode $mode --generation_path $generation_path --model_name $model_name --baseline_data_path $baseline_data_path \
                      --api_idx $api_idx --restart_pos $restart_pos --generation_number $generation_number --iteration $i --beam_number $beam_number --lang $lang \
                      --pattern_path "$pattern_path" \
                      --limit $debug_limit

if [ $? -eq 0 ]; then
  python merge.py  --iteration $i --generation_number $generation_number  --lang $lang \
                 --model_name $model_name --mode $base_mode --training_data_path $training_data_path \
                 --generation_path $generation_path --beam_number $beam_number \
                 --limit $debug_limit
else
  echo "generate.py failed."
  exit 1
fi

eval_output_path=./output/$lang
final_mode="${base_mode}_final/$i"

python evaluate.py  --do_train --mode $base_mode/$i --lang $lang  --process_number 60 \
                    --output_path $eval_output_path --model_name $model_name \
                    --iteration $i \
                    --slice 1 --testing_number 0 --test_case_path $private_test_case_path

echo "Done."