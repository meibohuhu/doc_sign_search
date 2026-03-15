# 处理单个视频文件
conda run -n internvl python data_aggregation_output/clip_sign1news_videos.py \
    --metadata youtube-asl_metadata.csv \
    --input data_aggregation_output/_FW1JoDAW8w.mp4 \
    --output-dir data_aggregation_output/clips_FW1JoDAW8w

du -sh /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1
find /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1 -type f -name "*.mp4" | wc -l
find /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_video_ids_stage2 -type f -name "*.mp4" | wc -l

find /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1_output/clips/ -type f -name "*.mp4" | wc -l
find /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part2_output/clips/ -type f -name "*.mp4" | wc -l

# 处理整个目录的视频
conda run -n internvl python data_aggregation_output/clip_sign1news_videos.py \
    --metadata youtube-asl_metadata.csv \
    --input-dir data_aggregation_output \
    --output-dir data_aggregation_output/clips


python data_aggregation_output/generate_sign1news_bbox.py --clips-dir data_aggregation_output/clips_FW1JoDAW8w --output data_aggregation_output/clips_FW1JoDAW8w_bbox.json

python data_aggregation_output/visualize_bbox.py \
    --bbox-file data_aggregation_output/clips_bbox.json \
    --clips-dir data_aggregation_output/clips_FW1JoDAW8w \
    --output-dir data_aggregation_output/clips_FW1JoDAW8w_cropped_224 \
    --target-size 224


clip_sign1news_videos.py -> generate_sign1news_bbox.py -> visualize_bbox.py

youtube_part1_output/
├── clips/              # 步骤1：根据 timestamps 裁剪的 clips
├── clips_bbox.json     # 步骤2：生成的 bbox 文件
└── clips_cropped_224/  # 步骤3：最终裁剪到 224x224 的视频

cd /home/mh2803/projects/sign_language_llm

conda run -n internvl python data_aggregation_output/process_video_pipeline.py \
    --input-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part2 \
    --output-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part2_output \
    --metadata youtube-asl_metadata.csv \
    --target-size 224

conda run -n internvl python data_aggregation_output/process_video_pipeline.py \
    --input-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1 \
    --output-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1_output \
    --metadata youtube-asl_metadata.csv \
    --target-size 224


    youtube_part1_output/
├── clips/              # 步骤1：根据 timestamps 裁剪的 clips
├── clips_bbox.json     # 步骤2：生成的 bbox 文件
└── clips_cropped_224/  # 步骤3：最终裁剪到 224x224 的视频