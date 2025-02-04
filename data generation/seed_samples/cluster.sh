#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=TRACE
export VLLM_TRACE_FUNCTION=1
SAVE_DIR="../data"

# asqa
python collect_pipeline.py --mode run \
  --save_load_dir ${SAVE_DIR}/asqa_data \
  --dataset_name asqa \
  --input_dataset ${SAVE_DIR}/asqa_data/train_test.json \
  --input_content question \
  --embed_batch_size 2048 \
  --n_samples 100000 \
  --n_selected 3000 \
  --build_hf_ds \
  --topic_mode single_topic \
  --dbscan_eps 0.3 \
  --dbscan_min_samples 20 \
  --summary_model_kwargs \
    model="Mixtral-8x7B-Instruct-v0.1-awq" \
    quantization="awq" \
    dtype=torch.float16 \
    max_seq_len_to_capture=128 \
    tensor_parallel_size=4 \
    gpu_memory_utilization=0.9 \
    max_num_seqs=256 \
    enforce_eager=True \
    disable_custom_all_reduce=True \
  --embed_devices cuda:5


# qampari
python collect_pipeline.py --mode run \
  --save_load_dir ${SAVE_DIR}/qampari_data \
  --dataset_name qampari \
  --input_dataset ${SAVE_DIR}/qampari_data/train_data.jsonl \
  --input_content question \
  --embed_batch_size 2048 \
  --n_samples 100000 \
  --n_selected 3000 \
  --build_hf_ds \
  --topic_mode single_topic \
  --dbscan_eps 0.2 \
  --dbscan_min_samples 50 \
  --summary_model_kwargs \
    model="Mixtral-8x7B-Instruct-v0.1-awq" \
    quantization="awq" \
    dtype=torch.float16 \
    max_seq_len_to_capture=128 \
    tensor_parallel_size=4 \
    gpu_memory_utilization=0.9 \
    max_num_seqs=256 \
    enforce_eager=True \
    disable_custom_all_reduce=True \
  --embed_devices cuda:4 cuda:5


# eli5
python collect_pipeline.py --mode skip \
  --save_load_dir ${SAVE_DIR}/eli5_data \
  --dataset_name eli5 \
  --input_dataset ${SAVE_DIR}/eli5_data/eli5 \
  --input_content question \
  --n_samples 100000 \
  --n_selected 4000 \
  --build_hf_ds \


# expertqa
python collect_pipeline.py --mode skip \
  --save_load_dir ${SAVE_DIR}/expertqa_data \
  --dataset_name expertqa \
  --input_dataset ${SAVE_DIR}/r2_compiled_anon.jsonl \
  --input_content question \
  --n_samples 100000 \