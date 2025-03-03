#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7 python main.py configs/asqa_openai.yaml 2>&1 | tee logs/asqa_openai.log

# CUDA_VISIBLE_DEVICES=6,7 python main.py configs/eli5_default_baseline.yaml 2>&1 | tee logs/eli5_default_llama3_8b.log

# CUDA_VISIBLE_DEVICES=6,7 python main.py configs/qampari_default_baseline.yaml 2>&1 | tee logs/qampari_default_llama3_8b.log 


