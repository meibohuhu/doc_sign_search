#!/bin/bash
# JSON Generator Runner Script
# ===========================
# This script runs the JSON generation process for segmented videos

echo "📄 How2Sign Segmented Videos JSON Generator"
echo "=========================================="

# Set the current directory
cd "$(dirname "$0")"

# Check if required files exist
if [ ! -f "how2sign_realigned_test.csv" ]; then
    echo "❌ Error: how2sign_realigned_test.csv not found!"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📄 CSV file: how2sign_realigned_test.csv"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION] [VIDEO_DIR]"
    echo ""
    echo "Options:"
    echo "  test        - Generate JSON for test_new_naming directory"
    echo "  custom      - Generate JSON for custom directory (specify directory)"
    echo "  all         - Generate JSON for all segmented clips (if available)"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test                           # Generate JSON for test_new_naming"
    echo "  $0 custom my_segmented_videos     # Generate JSON for my_segmented_videos"
    echo "  $0 all                            # Generate JSON for all available clips"
}

# Parse arguments
case "${1:-test}" in
    "test")
        VIDEO_DIR="test_new_naming"
        OUTPUT_FILE="segmented_videos_test.json"
        echo "🧪 Running TEST mode for: $VIDEO_DIR"
        ;;
    "custom")
        if [ -z "$2" ]; then
            echo "❌ Error: Please specify a video directory for custom mode"
            echo "Example: $0 custom my_segmented_videos"
            exit 1
        fi
        VIDEO_DIR="$2"
        OUTPUT_FILE="segmented_videos_${2}.json"
        echo "🎯 Running CUSTOM mode for: $VIDEO_DIR"
        ;;
    "all")
        # Look for common segmented video directories
        if [ -d "segmented_clips" ]; then
            VIDEO_DIR="segmented_clips"
            OUTPUT_FILE="segmented_videos_all.json"
            echo "🚀 Running ALL mode for: $VIDEO_DIR"
        elif [ -d "test_new_naming" ]; then
            VIDEO_DIR="test_new_naming"
            OUTPUT_FILE="segmented_videos_all.json"
            echo "🚀 Running ALL mode for: $VIDEO_DIR (fallback)"
        else
            echo "❌ Error: No segmented video directory found!"
            echo "Available directories:"
            ls -la | grep "^d" | awk '{print "  " $9}' | grep -v "^\.$\|^\.\.$"
            exit 1
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

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    echo "❌ Error: Video directory not found: $VIDEO_DIR"
    echo "Available directories:"
    ls -la | grep "^d" | awk '{print "  " $9}' | grep -v "^\.$\|^\.\.$"
    exit 1
fi

echo "🎥 Video directory: $VIDEO_DIR"
echo "📄 Output file: $OUTPUT_FILE"
echo ""

# Count video files
VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" | wc -l)
echo "📊 Found $VIDEO_COUNT video files to process"

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "❌ No video files found in $VIDEO_DIR"
    exit 1
fi

# Run the JSON generator
echo "🔄 Generating JSON file..."
python generate_segmented_json.py \
    --video-dir "$VIDEO_DIR" \
    --output-json "$OUTPUT_FILE" \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ JSON generation completed successfully!"
    echo "📄 Output file: $OUTPUT_FILE"
    echo "📊 Check the log file: generate_segmented_json.log"
    echo ""
    echo "🔍 Preview of generated JSON:"
    echo "=============================="
    head -20 "$OUTPUT_FILE"
    echo "..."
    echo "=============================="
else
    echo "❌ JSON generation failed!"
    exit 1
fi

