#!/bin/bash
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=None
export CUDA_VISIBLE_DEVICES="0,1"

task="dpo" 

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25679 --config_file training_configs/deepspeed_zero3.yaml \
${task}.py \
training_configs/${task}_full_align.yaml