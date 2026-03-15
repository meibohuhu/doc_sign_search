#!/bin/bash

# 裁剪 youtube_part1 的视频
echo "=========================================="
echo "开始处理 youtube_part1 目录..."
echo "=========================================="
python3 /home/mh2803/projects/sign_language_llm/clip_youtube_asl_videos.py \
    --csv /home/mh2803/projects/sign_language_llm/youtube_asl_clips_stage2_part1.csv \
    --input-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1 \
    --output-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1_output \
    --num-workers 4

echo ""
echo "=========================================="
echo "开始处理 youtube_part2 目录..."
echo "=========================================="
python3 /home/mh2803/projects/sign_language_llm/clip_youtube_asl_videos.py \
    --csv /home/mh2803/projects/sign_language_llm/youtube_asl_clips_stage2_part2.csv \
    --input-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part2 \
    --output-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part2_output \
    --num-workers 4

echo ""
echo "=========================================="
echo "所有视频裁剪完成！"
echo "=========================================="

