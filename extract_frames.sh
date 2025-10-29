#!/bin/bash
# Script to extract frames from a video using ffmpeg

VIDEO_PATH="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_320x320/fzDHRCKr7wU_4-8-rgb_front.mp4"
NUM_FRAMES=5

# Get total number of frames
TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$VIDEO_PATH")
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO_PATH")

echo "Video info:"
echo "  Total frames: $TOTAL_FRAMES"
echo "  Duration: ${DURATION}s"
echo ""

# Create output directory
OUTPUT_DIR="$(dirname "$VIDEO_PATH")/extracted_frames"
mkdir -p "$OUTPUT_DIR"

# Get base name without extension
BASE_NAME=$(basename "$VIDEO_PATH" .mp4)

echo "Extracting $NUM_FRAMES frames..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Extract frames evenly spaced throughout the video
for i in $(seq 0 $((NUM_FRAMES - 1))); do
    # Calculate frame index (evenly spaced)
    if [ $NUM_FRAMES -eq 1 ]; then
        frame_idx=0
    else
        frame_idx=$(( i * (TOTAL_FRAMES - 1) / (NUM_FRAMES - 1) ))
    fi
    
    # Calculate timestamp
    timestamp=$(echo "$frame_idx * $DURATION / $TOTAL_FRAMES" | bc -l)
    
    echo "Extracting frame $((i+1))/$NUM_FRAMES (index: $frame_idx, time: ${timestamp}s)..."
    
    # Extract frame at specific timestamp
    ffmpeg -i "$VIDEO_PATH" -ss "$timestamp" -vframes 1 \
           "$OUTPUT_DIR/${BASE_NAME}_frame_$(printf %02d $((i+1)))_${frame_idx}.png" -y -loglevel error
done

echo ""
echo "Extraction complete! Saved to: $OUTPUT_DIR"
