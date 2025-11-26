#!/bin/bash
# Background script to run video cropping with multiple workers
# Usage: 
#   bash run_crop_video_background.sh           # Use all CPU cores
#   bash run_crop_video_background.sh 32        # Use 32 workers
#   bash run_crop_video_background.sh 16        # Use 16 workers

# Set working directory to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Output directory
OUTPUT_DIR="/shared/rc/llm-gen-agent/mhu/videos/open_asl/cropped_videos_nonad"

# Number of workers (0 = use all CPU cores, or specify a number like 16, 32, etc.)
# Can be overridden by command line argument
NUM_WORKERS=${1:-0}  # Default to 0 (all cores), or use first argument if provided

# Log files
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/crop_video_${TIMESTAMP}.log"
ERR_FILE="$LOG_DIR/crop_video_${TIMESTAMP}.err"

echo "Starting video cropping in background..."
echo "Output directory: $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS (0 = all CPU cores)"
echo "Log file: $LOG_FILE"
echo "Error log: $ERR_FILE"
echo "PID will be saved to: $LOG_DIR/crop_video.pid"

# Run in background with nohup
nohup python crop_video_openasl.py \
    --output "$OUTPUT_DIR" \
    --workers "$NUM_WORKERS" \
    > "$LOG_FILE" 2> "$ERR_FILE" &

# Save PID
PID=$!
echo $PID > "$LOG_DIR/crop_video.pid"

echo ""
echo "Process started with PID: $PID"
echo "To check progress, run: tail -f $LOG_FILE"
echo "To check errors, run: tail -f $ERR_FILE"
echo "To stop the process, run: kill $PID"
echo "Or use: kill \$(cat $LOG_DIR/crop_video.pid)"

