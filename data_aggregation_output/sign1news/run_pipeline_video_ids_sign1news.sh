#!/bin/bash
# Pipeline for youtube_asl_clips_video_ids_2

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate internvl

INPUT_DIR="/scratch/mh2803/source/sign1news_video_ids"
OUTPUT_DIR="/scratch/mh2803/source/sign1news_video_processed_ids"
METADATA="/home/stu2/s15/mh2803/workspace/doc_sign_search/data_aggregation_output/sign1news/sign1news_clips.csv"
LOG_FILE="$OUTPUT_DIR/pipeline.log"
WORKERS=${WORKERS:-16}

mkdir -p "$OUTPUT_DIR"

echo "Starting pipeline..."
echo "Input:   $INPUT_DIR"
echo "Output:  $OUTPUT_DIR"
echo "Workers: $WORKERS"
echo "Log:     $LOG_FILE"

python /home/stu2/s15/mh2803/workspace/doc_sign_search/data_aggregation_output/process_video_pipeline.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --metadata "$METADATA" \
    --log-file "$LOG_FILE" \
    --workers "$WORKERS" \
    "$@"
