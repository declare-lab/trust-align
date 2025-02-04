#!/bin/bash

TODAY=$(date +%Y-%m-%d)

BASE_DIR=""
OUTPUT_DIR=""
mkdir -p $OUTPUT_DIR

# checkpoint-56
DATAFILE="$BASE_DIR/asqa-checkpoint-56-gtr-shot2-ndoc5-42-temp1.0.json"
LOG_FILE="$OUTPUT_DIR/asqa-checkpoint-56-gtr-shot2-ndoc5-42-temp1.0.log"
CUDA_VISIBLE_DEVICES=2,3 python error_selection.py --data_file $DATAFILE --output_dir $OUTPUT_DIR --dataset_name asqa 2>&1 | tee $LOG_FILE 

DATAFILE="$BASE_DIR/eli5-checkpoint-56-gtr-shot2-ndoc5-42-temp1.0.json"
LOG_FILE="$OUTPUT_DIR/eli5-checkpoint-56-gtr-shot2-ndoc5-42-temp1.0.log"
CUDA_VISIBLE_DEVICES=2,3 python error_selection.py --data_file $DATAFILE --output_dir $OUTPUT_DIR --dataset_name eli5 2>&1 | tee $LOG_FILE 

DATAFILE="$BASE_DIR/qampari-checkpoint-56-gtr-shot2-ndoc5-42-temp1.0.json"
LOG_FILE="$OUTPUT_DIR/qampari-checkpoint-56-gtr-shot2-ndoc5-42-temp1.0.log"
CUDA_VISIBLE_DEVICES=2,3 python error_selection.py --data_file $DATAFILE --output_dir $OUTPUT_DIR --dataset_name qampari 2>&1 | tee $LOG_FILE 

# Wait for all background processes to finish
wait

echo "DONE"