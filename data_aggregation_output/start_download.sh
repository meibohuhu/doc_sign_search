#!/bin/bash
# Start video download in conda internvl environment
# Usage: ./start_download.sh [input_file] [output_dir]
#   input_file: Path to file containing video IDs (default: youtube_video_ids_stage2_notstrict.txt in script directory)
#   output_dir: Directory to save downloaded videos (default: /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/downloads)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
# INPUT_FILE="${1:-$SCRIPT_DIR/youtube_video_ids_stage2.txt}"
INPUT_FILE="${1:-$SCRIPT_DIR/youtube_video_ids_diff.txt}"
OUTPUT_DIR="${2:-/shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_video_ids_stage2_diff/}"

# Convert input file to absolute path
if [ -f "$INPUT_FILE" ]; then
    INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"
elif [ ! -f "$INPUT_FILE" ] && [ "${INPUT_FILE:0:1}" != "/" ]; then
    # Relative path that doesn't exist yet - make it relative to script dir
    INPUT_FILE="$SCRIPT_DIR/$INPUT_FILE"
fi

# Convert output dir to absolute path
if [ -d "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
elif [ "${OUTPUT_DIR:0:1}" != "/" ]; then
    # Relative path - make it absolute
    OUTPUT_DIR="$(cd "$SCRIPT_DIR" && cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate internvl

# Set LD_LIBRARY_PATH for ffmpeg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Export environment variables for Python script
export YOUTUBE_DOWNLOAD_INPUT_FILE="$INPUT_FILE"
export YOUTUBE_DOWNLOAD_OUTPUT_DIR="$OUTPUT_DIR"

echo "Starting video download..."
echo "Environment: internvl"
echo "Python: $(which python)"
echo "yt-dlp: $(python -m yt_dlp --version 2>&1 | head -1)"
echo "ffmpeg: $(which ffmpeg)"
echo ""
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the download script
python youtube_download.py

