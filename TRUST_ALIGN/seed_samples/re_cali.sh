#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
NUM_GPUS=6
SAVE_DIR="../data"

# Sphere
export BM25_SPHERE_PATH=""

# Wiki
export DPR_WIKI_TSV=""
export GTR_EMB=""

accelerate launch --num_processes=${NUM_GPUS} retrieval_cali.py \
    --dataset_name asqa \
    --data_file ${SAVE_DIR}/asqa_data/asqa_run.json \
    --output_file ${SAVE_DIR}/asqa_data/asqa_doc.json \
    --retriever gtr \
    --oracle \
    --answer_found 

accelerate launch --num_processes=${NUM_GPUS} retrieval_cali.py \
    --dataset_name qampari \
    --data_file ${SAVE_DIR}/qampari_data/qampari_run.json \
    --output_file ${SAVE_DIR}/qampari_data/qampari_doc.json \
    --retriever gtr  \
    --oracle \
    --answer_found 

accelerate launch --num_processes=${NUM_GPUS}  retrieval_cali.py \
    --dataset_name eli5 \
    --data_file ${SAVE_DIR}/eli5_data/eli5_run.json \
    --output_file ${SAVE_DIR}/eli5_data/eli5_doc.json \
    --retriever bm25 \
    --oracle \
    --answer_found 

accelerate launch --num_processes=${NUM_GPUS}  retrieval_cali.py \
    --dataset_name expertqa \
    --data_file ${SAVE_DIR}/expertqa_data/expertqa_run.json \
    --output_file ${SAVE_DIR}/expertqa_data/expertqa_doc.json \
    --retriever gtr \
    --oracle \
    --answer_found 