#!/bin/bash
# Simple script to run youtube_download.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration: set input file and output directory
INPUT_FILE="${SCRIPT_DIR}/youtube_asl_clips_video_ids_3.txt"
OUTPUT_DIR="/scratch/mh2803/source/youtube_asl_clips_video_ids_3"

# Export environment variables for Python script
export YOUTUBE_DOWNLOAD_INPUT_FILE="$INPUT_FILE"
export YOUTUBE_DOWNLOAD_OUTPUT_DIR="$OUTPUT_DIR"

# Run the download script
cd "$SCRIPT_DIR"
python youtube_download.py

