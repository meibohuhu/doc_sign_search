#!/bin/bash
# Copy DailyMoth videos to How2Sign folder for combined training

SOURCE_DIR="/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/videos"
DEST_DIR="/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224"

echo "📁 Copying DailyMoth videos to How2Sign folder..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""

# Count total files first
TOTAL=$(find "$SOURCE_DIR" -type f -name "*.mp4" | wc -l)
echo "📊 Found $TOTAL MP4 files to copy"
echo ""

# Copy files with parallel processing for speed (8 parallel jobs)
echo "🔄 Starting copy with 8 parallel jobs (this may take a while for 48k files)..."
echo "   Progress will be shown every 5000 files..."
echo ""

# Use xargs with parallel processing
find "$SOURCE_DIR" -type f -name "*.mp4" -print0 | \
    xargs -0 -P 8 -I {} cp {} "$DEST_DIR/"

echo ""

echo ""
echo "✅ Copy complete!"
echo "📊 Checking destination..."

DEST_COUNT=$(find "$DEST_DIR" -type f -name "*.mp4" | wc -l)
echo "   Total files in destination: $DEST_COUNT"

