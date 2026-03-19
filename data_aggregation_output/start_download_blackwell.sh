#!/bin/bash
# Simple script to run youtube_download.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration: set input file and output directory
INPUT_FILE="${SCRIPT_DIR}/sign1news_video_ids.txt"
OUTPUT_DIR="/scratch/mh2803/source/sign1news_video_ids"

# Export environment variables for Python script
export YOUTUBE_DOWNLOAD_INPUT_FILE="$INPUT_FILE"
export YOUTUBE_DOWNLOAD_OUTPUT_DIR="$OUTPUT_DIR"

# Run the download script
cd "$SCRIPT_DIR"
python youtube_download.py

