#!/bin/bash

# usage: sh doc_recombination.sh asqa

# Script to run doc_recombination.py for different datasets

# Check if dataset name is passed as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <DATASET_NAME>"
  exit 1
fi

# Define variables for dataset names and paths
DATASET_NAME=$1
TODAY=$(date +%Y%m%d)

BASE_DIR="../data"

DATA_FILE="$BASE_DIR/${DATASET_NAME}_doc.json"
OUTPUT_FILE="$BASE_DIR/${DATASET_NAME}_doc_augment.json"
LOG_FILE="$BASE_DIR/${DATASET_NAME}_log.log"
METADATA_FILE="$BASE_DIR/${DATASET_NAME}_metadata.json"

# Clear the log file if it exists
> $LOG_FILE

# Run the script for specified dataset
echo "Processing $DATASET_NAME dataset..." | tee -a $LOG_FILE
python doc_recombination.py --dataset_name $DATASET_NAME --data_file $DATA_FILE --metadata_file $METADATA_FILE --output_file $OUTPUT_FILE 2>&1 | tee -a $LOG_FILE

echo "$DATASET_NAME dataset processed successfully." | tee -a $LOG_FILE