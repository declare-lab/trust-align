#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python gen_answers.py --data_file asqa_error_instruction.json --output_file asqa_new_answers_cite.jsonl --dataset_name asqa 2>&1 | tee datasetB/asqa_ans.log
python gen_answers.py --data_file eli5_error_instruction.json --output_file eli5_new_answers_cite.jsonl --dataset_name eli5 2>&1 | tee datasetB/eli5_ans.log