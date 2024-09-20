#!/bin/bash
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

task="sft" 

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file training_configs/deepspeed_zero3.yaml \
${task}_only_out.py \
training_configs/${task}_full_only_out.yaml