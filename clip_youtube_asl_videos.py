"""
根据 youtube_asl_clips.csv 文件裁剪视频
从 CSV 文件读取时间戳信息，使用 ffmpeg 裁剪视频（支持多线程）

# 基本用法
python clip_youtube_asl_videos.py --input-dir /path/to/videos --output-dir clips

# 指定单个输入文件
python clip_youtube_asl_videos.py --input /home/mh2803/projects/sign_language_llm/data_aggregation_output/ZSO1STw4OBk.mp4 --output-dir data_aggregation_output/clips_ZSO1STw4OBk    

# 使用多线程加速（默认4线程）
python clip_youtube_asl_videos.py --input-dir /path/to/videos --output-dir /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/youtube_part1_output --num-workers 4

# 只处理特定视频ID（测试用）
python clip_youtube_asl_videos.py --input-dir /path/to/videos --video-id sA4dfSu_8L8 --output-dir clips

# 限制处理数量（测试用）
python clip_youtube_asl_videos.py --input-dir /path/to/videos --max-clips 100 --output-dir clips --num-workers 4
"""

import os
import csv
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 增加 CSV 字段大小限制
csv.field_size_limit(sys.maxsize)


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
    # 将秒数转换为 HH:MM:SS.mmm 格式
    start_str = format_timestamp(start_time)
    end_str = format_timestamp(end_time)
    
    # 构建 ffmpeg 命令
    cmd = [
        ffmpeg_path,
        '-ss', start_str,
        '-to', end_str,
        '-i', input_video,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '20',
        '-threads', '1',
        '-y',  # 覆盖输出文件
        output_video
    ]
    
    try:
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


def find_input_video(video_id, input_path):
    """
    查找输入视频文件
    
    Args:
        video_id: 视频ID
        input_path: 输入路径（文件或目录）
    
    Returns:
        视频文件路径，如果找不到返回 None
    """
    # 如果是文件，直接返回
    if os.path.isfile(input_path):
        return input_path
    
    # 如果是目录，查找视频文件
    input_dir = input_path
    possible_names = [
        f"{video_id}.mp4",
        f"-{video_id}.mp4",
        f"_{video_id}.mp4",
    ]
    
    # 首先尝试常见格式
    for name in possible_names:
        video_path = os.path.join(input_dir, name)
        if os.path.exists(video_path):
            return video_path
    
    # 如果找不到，尝试模糊匹配
    input_dir_path = Path(input_dir)
    matching_files = list(input_dir_path.glob(f"*{video_id}*.mp4"))
    if matching_files:
        return str(matching_files[0])
    
    return None


def load_clips_from_csv(csv_path):
    """
    从 CSV 文件加载视频片段信息
    
    Returns:
        list: 包含所有片段的列表，每个元素是一个字典
    """
    clips = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            clips.append(row)
    return clips


def process_clip(clip_info, input_path, output_dir, ffmpeg_path):
    """
    处理单个视频片段
    
    Args:
        clip_info: 片段信息字典（包含 VIDEO_ID, START_REALIGNED, END_REALIGNED, SENTENCE_NAME 等）
        input_path: 输入视频路径（文件或目录）
        output_dir: 输出视频目录
        ffmpeg_path: ffmpeg 路径
    
    Returns:
        tuple: (status, video_id) status 可以是 'success', 'skip', 'fail', 'missing_video', 'invalid'
    """
    video_id = clip_info['VIDEO_ID']
    start_time = float(clip_info['START_REALIGNED'])
    end_time = float(clip_info['END_REALIGNED'])
    sentence_name = clip_info['SENTENCE_NAME']
    
    # 检查时间戳有效性
    if start_time >= end_time:
        return ('invalid', video_id)
    
    # 查找输入视频文件
    input_video = find_input_video(video_id, input_path)
    if not input_video:
        return ('missing_video', video_id)
    
    # 生成输出文件名（使用 SENTENCE_NAME）
    output_filename = f"{sentence_name}.mp4"
    output_video = os.path.join(output_dir, output_filename)
    
    # 如果文件已存在，跳过
    if os.path.exists(output_video):
        return ('skip', video_id)
    
    # 创建输出目录（线程安全，多次调用无妨）
    os.makedirs(output_dir, exist_ok=True)
    
    # 裁剪视频
    if clip_video(input_video, output_video, start_time, end_time, ffmpeg_path):
        return ('success', video_id)
    else:
        return ('fail', video_id)


def main():
    parser = argparse.ArgumentParser(description="根据 youtube_asl_clips.csv 文件裁剪视频")
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/mh2803/projects/sign_language_llm/youtube_asl_clips_stage2.csv",
        help="CSV 文件路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入视频文件路径（单个文件）"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="输入视频目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="clips_youtube_asl",
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
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="最多处理的片段数量（用于测试）"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="并行处理的线程数（默认: 4）"
    )
    
    args = parser.parse_args()
    
    # 读取 CSV 文件
    print(f"正在读取 CSV 文件: {args.csv}")
    clips = load_clips_from_csv(args.csv)
    print(f"找到 {len(clips)} 个视频片段")
    
    # 如果指定了 video_id，只处理该视频的片段
    if args.video_id:
        clips = [c for c in clips if c['VIDEO_ID'] == args.video_id]
        print(f"过滤后: {len(clips)} 个片段（仅 {args.video_id}）")
    
    # 如果指定了 max_clips，限制处理数量
    if args.max_clips:
        clips = clips[:args.max_clips]
        print(f"限制处理数量: {len(clips)} 个片段")
    
    # 确定输入路径
    if args.input:
        input_path = args.input
        print(f"输入文件: {input_path}")
    elif args.input_dir:
        input_path = args.input_dir
        print(f"输入目录: {input_path}")
    else:
        print("错误: 必须指定 --input 或 --input-dir")
        return
    
    print(f"输出目录: {args.output_dir}")
    print(f"FFmpeg: {args.ffmpeg}")
    print(f"线程数: {args.num_workers}")
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
    
    # 创建输出目录（提前创建，避免多线程竞争）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理视频片段（多线程）
    success_count = 0
    skip_count = 0
    fail_count = 0
    missing_video_count = 0
    invalid_count = 0
    
    # 用于跟踪哪些视频文件找不到
    missing_videos = set()
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 提交所有任务
        future_to_clip = {
            executor.submit(process_clip, clip_info, input_path, args.output_dir, args.ffmpeg): clip_info
            for clip_info in clips
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(clips), desc="处理片段") as pbar:
            for future in as_completed(future_to_clip):
                clip_info = future_to_clip[future]
                try:
                    status, video_id = future.result()
                    
                    if status == 'success':
                        success_count += 1
                    elif status == 'skip':
                        skip_count += 1
                    elif status == 'fail':
                        fail_count += 1
                    elif status == 'missing_video':
                        missing_videos.add(video_id)
                        missing_video_count += 1
                    elif status == 'invalid':
                        invalid_count += 1
                except Exception as e:
                    print(f"\n  ✗ 处理片段时出错: {clip_info.get('SENTENCE_NAME', 'unknown')}, 错误: {e}")
                    fail_count += 1
                
                pbar.update(1)
    
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"成功: {success_count}")
    print(f"跳过（文件已存在）: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"跳过（找不到视频文件）: {missing_video_count}")
    print(f"无效时间戳: {invalid_count}")
    if missing_videos:
        print(f"找不到视频文件的 video_id 示例（前10个）: {list(missing_videos)[:10]}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

