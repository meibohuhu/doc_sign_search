#!/bin/bash
# Video Segmentation Runner Script
# ================================
# This script runs the video segmentation process with different options

echo "🎬 How2Sign Video Segmentation Script"
echo "====================================="

# Set the current directory
cd "$(dirname "$0")"

# Check if required files exist
if [ ! -f "how2sign_realigned_test.csv" ]; then
    echo "❌ Error: how2sign_realigned_test.csv not found!"
    exit 1
fi

if [ ! -d "raw_videos" ]; then
    echo "❌ Error: raw_videos directory not found!"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📄 CSV file: how2sign_realigned_test.csv"
echo "🎥 Video directory: raw_videos"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  test        - Process only 10 clips for testing"
    echo "  single      - Process clips from a single video (specify video name)"
    echo "  all         - Process all clips (default)"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test                    # Process 10 clips for testing"
    echo "  $0 single -fZc293MpJk      # Process all clips from -fZc293MpJk-1-rgb_front.mp4"
    echo "  $0 all                     # Process all clips"
}

# Parse arguments
case "${1:-all}" in
    "test")
        echo "🧪 Running TEST mode (10 clips only)..."
        python segment_videos.py --max-clips 10 --output-dir "test_segmented_clips"
        ;;
    "single")
        if [ -z "$2" ]; then
            echo "❌ Error: Please specify a video name for single mode"
            echo "Example: $0 single -fZc293MpJk"
            exit 1
        fi
        echo "🎯 Running SINGLE video mode for: $2"
        python segment_videos.py --video-filter "$2" --output-dir "single_video_clips"
        ;;
    "all")
        echo "🚀 Running FULL segmentation (all clips)..."
        echo "⚠️  This may take a long time and create many files!"
        read -p "Are you sure you want to continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python segment_videos.py --output-dir "segmented_clips"
        else
            echo "❌ Cancelled by user"
            exit 0
        fi
        ;;
    "help"|"-h"|"--help")
        show_usage
        exit 0
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "✅ Segmentation completed!"
echo "📊 Check the log file: video_segmentation.log"
echo "📁 Check the output directory for segmented clips"

