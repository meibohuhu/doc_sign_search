"""
根据 caption timestamps 裁剪视频
从 CSV 文件读取时间戳信息，使用 ffmpeg 裁剪视频
"""

import os
import json
import csv
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm


def format_timestamp(seconds):
    """将秒数转换为 HH:MM:SS.mmm 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{int((secs % 1) * 1000):03d}"


def format_filename_timestamp(seconds):
    """将秒数转换为文件名格式：HH_MM_SS_mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}_{minutes:02d}_{int(secs):02d}_{int((secs % 1) * 1000):03d}"


def clip_video(input_video, output_video, start_time, end_time, ffmpeg_path="ffmpeg"):
    """
    使用 ffmpeg 裁剪视频
    遵循 prep/crop_video.py 的模式：使用 -ss 和 -to 参数，直接重新编码以确保精确裁剪
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        ffmpeg_path: ffmpeg 可执行文件路径
    """
    # 将秒数转换为 HH:MM:SS.mmm 格式（与 crop_video.py 保持一致）
    start_str = format_timestamp(start_time)
    end_str = format_timestamp(end_time)
    
    # 构建 ffmpeg 命令，遵循 crop_video.py 的模式
    # 使用 -ss 和 -to 都在 -i 之前，这样可以更精确地定位
    cmd = [
        ffmpeg_path,
        '-ss', start_str,
        '-to', end_str,
        '-i', input_video,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '20',  # 与 crop_video.py 保持一致
        '-threads', '1',  # 与 crop_video.py 保持一致
        '-y',  # 覆盖输出文件
        output_video
    ]
    
    try:
        # 使用 subprocess.call 并隐藏输出，与 crop_video.py 保持一致
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300  # 5分钟超时
        )
        
        return result.returncode == 0 and os.path.exists(output_video)
    except subprocess.TimeoutExpired:
        print(f"  ⚠ 超时: {output_video}")
        return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False


def process_video(video_id, video_info, input_video_path, output_dir, ffmpeg_path):
    """
    处理单个视频，根据 caption timestamps 裁剪
    
    Args:
        video_id: 视频ID
        video_info: 视频信息字典
        input_video_path: 输入视频文件路径（可以是完整路径或目录）
        output_dir: 输出视频目录
        ffmpeg_path: ffmpeg 路径
    """
    # 如果 input_video_path 是文件，直接使用
    if os.path.isfile(input_video_path):
        input_video = input_video_path
    else:
        # 否则当作目录处理，查找输入视频文件
        input_dir = input_video_path
        input_video = os.path.join(input_dir, f"{video_id}.mp4")
        
        # 如果找不到，尝试其他可能的文件名格式
        if not os.path.exists(input_video):
            # 尝试带前缀的文件名（如 -_ojvPJp0c0.mp4）
            alt_input = os.path.join(input_dir, f"-{video_id}.mp4")
            if os.path.exists(alt_input):
                input_video = alt_input
            else:
                # 尝试查找任何包含 video_id 的文件
                input_dir_path = Path(input_dir)
                matching_files = list(input_dir_path.glob(f"*{video_id}*.mp4"))
                if matching_files:
                    input_video = str(matching_files[0])
                else:
                    # 找不到视频文件，跳过
                    print(f"  ⚠ 跳过（找不到视频文件）: {video_id}")
                    return 0
    
    if not os.path.exists(input_video):
        # 视频文件不存在，跳过
        print(f"  ⚠ 跳过（视频文件不存在）: {video_id}")
        return 0
    
    # 解析 caption_timestamps
    caption_timestamps_str = video_info.get("caption_timestamps", "[]")
    if not caption_timestamps_str:
        print(f"  ⚠ 没有 caption timestamps: {video_id}")
        return 0
    
    try:
        caption_timestamps = json.loads(caption_timestamps_str)
    except json.JSONDecodeError as e:
        print(f"  ✗ 解析 caption timestamps 失败: {video_id}, 错误: {e}")
        return 0
    
    if not caption_timestamps or len(caption_timestamps) == 0:
        print(f"  ⚠ 空的 caption timestamps: {video_id}")
        return 0
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个时间戳片段
    success_count = 0
    for idx, timestamp in enumerate(caption_timestamps):
        start_time = timestamp.get("start", 0)
        end_time = timestamp.get("end", 0)
        text = timestamp.get("text", "")
        
        if start_time >= end_time:
            continue
        
        # 生成输出文件名：优先用 sentence_id，否则退回时间戳格式
        sentence_id = timestamp.get("sentence_id", "")
        if sentence_id:
            output_filename = f"{sentence_id}.mp4"
        else:
            start_str = format_filename_timestamp(start_time)
            end_str = format_filename_timestamp(end_time)
            output_filename = f"{video_id}-{start_str}-{end_str}.mp4"
        output_video = os.path.join(output_dir, output_filename)
        
        # 如果文件已存在，跳过
        if os.path.exists(output_video):
            success_count += 1
            continue
        
        # 裁剪视频（使用重新编码模式以确保精确裁剪）
        if clip_video(input_video, output_video, start_time, end_time, ffmpeg_path):
            success_count += 1
        else:
            print(f"  ✗ 裁剪失败: {video_id} [{start_time:.3f}-{end_time:.3f}]")
    
    return success_count


def load_metadata_from_csv(csv_path):
    """
    从 CSV 文件加载 metadata。支持两种格式：
    1. 原始格式（逗号分隔）：video_id, caption_timestamps(JSON), ...
    2. Clips 格式（制表符分隔）：VIDEO_ID, START_REALIGNED, END_REALIGNED, SENTENCE, ...
       每行是一个 clip，按 video_id 聚合后生成 caption_timestamps。

    Returns:
        dict: {video_id: {所有字段的字典，包含 caption_timestamps}}
    """
    csv.field_size_limit(10 * 1024 * 1024)  # 10MB

    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    # 检测分隔符和格式
    delimiter = '\t' if '\t' in first_line else ','
    fieldnames = [h.strip() for h in first_line.split(delimiter)]

    if 'VIDEO_ID' in fieldnames:
        # Clips 格式：按 VIDEO_ID 聚合，构造 caption_timestamps
        metadata = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                video_id = row['VIDEO_ID'].strip()
                try:
                    start = float(row['START_REALIGNED'])
                    end = float(row['END_REALIGNED'])
                except (ValueError, KeyError):
                    continue
                text = row.get('SENTENCE', '').strip()
                if video_id not in metadata:
                    metadata[video_id] = {'video_id': video_id, '_timestamps': []}
                sentence_id = row.get('SENTENCE_ID', '').strip()
                metadata[video_id]['_timestamps'].append(
                    {'start': start, 'end': end, 'text': text, 'sentence_id': sentence_id}
                )
        # 将聚合的 timestamps 序列化为 JSON 字符串（process_video 期望的格式）
        for vid, info in metadata.items():
            info['caption_timestamps'] = json.dumps(info.pop('_timestamps'))
        return metadata
    else:
        # 原始格式
        metadata = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                video_id = row['video_id']
                metadata[video_id] = row
        return metadata


def main():
    parser = argparse.ArgumentParser(description="根据 caption timestamps 裁剪视频")
    parser.add_argument(
        "--metadata",
        type=str,
        default="youtube-asl_metadata.csv",
        help="metadata CSV 文件路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入视频文件路径或目录（如果指定文件，则只处理该文件）"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="输入视频目录（如果未指定 --input）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clips",
        help="输出视频片段目录"
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="ffmpeg 可执行文件路径"
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="只处理指定的视频ID（用于测试）"
    )
    
    args = parser.parse_args()
    
    # 读取 metadata
    print(f"正在读取 metadata: {args.metadata}")
    metadata = load_metadata_from_csv(args.metadata)
    
    print(f"找到 {len(metadata)} 个视频")
    
    # 确定输入路径
    if args.input:
        input_path = args.input
        print(f"输入: {input_path}")
    elif args.input_dir:
        input_path = args.input_dir
        print(f"输入目录: {input_path}")
    else:
        print("错误: 必须指定 --input 或 --input-dir")
        return
    
    print(f"输出目录: {args.output_dir}")
    print(f"FFmpeg: {args.ffmpeg}")
    print("-" * 50)
    
    # 检查 ffmpeg
    try:
        result = subprocess.run(
            [args.ffmpeg, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        if result.returncode != 0:
            print(f"错误: 无法运行 ffmpeg ({args.ffmpeg})")
            return
    except Exception as e:
        print(f"错误: 无法找到或运行 ffmpeg ({args.ffmpeg}): {e}")
        return
    
    # 如果指定了输入文件，从文件名提取 video_id
    if args.input and os.path.isfile(args.input):
        # 从文件名提取 video_id（例如：_Bb-sdKEk5w.mp4 -> _Bb-sdKEk5w）
        video_filename = os.path.basename(args.input)
        video_id = os.path.splitext(video_filename)[0]
        
        # 如果文件名以 - 开头，可能需要去掉
        if video_id.startswith('-'):
            # 尝试去掉开头的 -，如果找不到再尝试原文件名
            alt_video_id = video_id[1:]
            if alt_video_id in metadata:
                video_id = alt_video_id
            elif video_id not in metadata:
                # 如果都找不到，尝试去掉开头的下划线
                if video_id.startswith('_'):
                    alt_video_id2 = video_id[1:]
                    if alt_video_id2 in metadata:
                        video_id = alt_video_id2
        
        if video_id not in metadata:
            print(f"错误: 在 metadata 中找不到视频 ID: {video_id}")
            print(f"提示: 尝试查找包含 '{video_id}' 的视频...")
            # 尝试模糊匹配
            matching_ids = [vid for vid in metadata.keys() if video_id in vid or vid in video_id]
            if matching_ids:
                print(f"找到可能的匹配: {matching_ids[:5]}")
                video_id = matching_ids[0]
            else:
                return
        
        video_ids = [video_id]
    elif args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = list(metadata.keys())
    
    # 处理视频
    total_clips = 0
    processed_videos = 0
    
    for video_id in tqdm(video_ids, desc="处理视频"):
        if video_id not in metadata:
            print(f"  ⚠ 跳过: {video_id} (不在 metadata 中)")
            continue
        
        video_info = metadata[video_id]
        clip_count = process_video(
            video_id,
            video_info,
            input_path,
            args.output_dir,
            args.ffmpeg
        )
        
        if clip_count > 0:
            processed_videos += 1
            total_clips += clip_count
    
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"处理视频数: {processed_videos}/{len(video_ids)}")
    print(f"生成片段数: {total_clips}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

