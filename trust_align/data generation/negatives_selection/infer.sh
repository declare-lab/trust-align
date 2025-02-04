#!/bin/bash

export CUDA_VISIBLE_DEVICES="4"
OUTPUT_DIR=""
INFER_FILE=""
LABEL="sft"
TEMP=0.5

# llama2
python inference.py --config resources/configs/asqa_llama2_shot2_ndoc5_gtr_rejection.yaml --temperature ${TEMP} --infer_file ${INFER_FILE} --output_dir ${OUTPUT_DIR} | tee ${OUTPUT_DIR}/${LABEL}-run-${TEMP}.log
