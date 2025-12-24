"""
使用caption_timestamps来裁剪视频片段
需要安装: pip install ffmpeg-python
或者直接使用ffmpeg命令行工具
"""
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict


def extract_segments_from_metadata(
    metadata_file: str,
    video_dir: str = "videos",
    output_dir: str = "segments",
    video_id: str = None
):
    """
    从CSV元数据文件中读取时间戳，并裁剪视频片段
    
    Args:
        metadata_file: CSV元数据文件路径
        video_dir: 原始视频文件目录
        output_dir: 输出片段目录
        video_id: 如果指定，只处理该视频ID；否则处理所有视频
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            current_video_id = row['video_id']
            
            # 如果指定了video_id，只处理该视频
            if video_id and current_video_id != video_id:
                continue
            
            # 检查是否有时间戳数据
            if not row['caption_timestamps']:
                print(f"视频 {current_video_id} 没有字幕时间戳，跳过")
                continue
            
            # 解析时间戳
            try:
                timestamps = json.loads(row['caption_timestamps'])
            except json.JSONDecodeError:
                print(f"视频 {current_video_id} 的时间戳数据格式错误，跳过")
                continue
            
            if not timestamps:
                print(f"视频 {current_video_id} 没有时间戳条目，跳过")
                continue
            
            # 查找视频文件
            video_file = find_video_file(video_dir, current_video_id)
            if not video_file:
                print(f"未找到视频文件: {current_video_id}，跳过")
                continue
            
            print(f"\n处理视频: {current_video_id}")
            print(f"找到 {len(timestamps)} 个字幕片段")
            
            # 提取每个片段
            extract_segments(
                video_file=video_file,
                timestamps=timestamps,
                output_dir=output_dir,
                video_id=current_video_id
            )


def find_video_file(video_dir: str, video_id: str) -> str:
    """查找视频文件（支持多种格式）"""
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        return None
    
    # 常见的视频格式
    extensions = ['.mp4', '.mkv', '.webm', '.flv', '.avi']
    
    for ext in extensions:
        video_file = video_dir_path / f"{video_id}{ext}"
        if video_file.exists():
            return str(video_file)
    
    return None


def extract_segments(
    video_file: str,
    timestamps: List[Dict],
    output_dir: str,
    video_id: str,
    padding: float = 0.5
):
    """
    使用ffmpeg提取视频片段
    
    Args:
        video_file: 输入视频文件路径
        timestamps: 时间戳列表，每个条目包含 'start', 'end', 'text'
        output_dir: 输出目录
        video_id: 视频ID
        padding: 每个片段前后的额外时间（秒），用于包含更多上下文
    """
    video_path = Path(video_file)
    output_path = Path(output_dir)
    
    # 为每个视频创建子目录
    video_output_dir = output_path / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, entry in enumerate(timestamps):
        start_time = max(0, entry['start'] - padding)  # 确保不为负数
        end_time = entry['end'] + padding
        duration = end_time - start_time
        
        # 清理文本，用作文件名（移除特殊字符）
        text_snippet = entry['text'][:50].replace('/', '_').replace('\\', '_')
        text_snippet = ''.join(c for c in text_snippet if c.isalnum() or c in (' ', '-', '_')).strip()
        text_snippet = text_snippet.replace(' ', '_')
        
        # 输出文件名
        output_file = video_output_dir / f"segment_{i+1:04d}_{start_time:.2f}s-{end_time:.2f}s_{text_snippet[:30]}.mp4"
        
        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',
            '-i', str(video_file),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # 使用copy避免重新编码，速度更快
            '-avoid_negative_ts', 'make_zero',
            '-y',  # 覆盖已存在的文件
            str(output_file)
        ]
        
        try:
            # 运行ffmpeg命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  ✓ 片段 {i+1}/{len(timestamps)}: {start_time:.2f}s - {end_time:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 片段 {i+1} 提取失败: {e.stderr}")
        except FileNotFoundError:
            print("错误: 未找到ffmpeg。请安装ffmpeg并确保它在PATH中。")
            print("下载地址: https://ffmpeg.org/download.html")
            return


def extract_segments_advanced(
    video_file: str,
    timestamps: List[Dict],
    output_dir: str,
    video_id: str,
    merge_consecutive: bool = True,
    min_gap: float = 1.0
):
    """
    高级版本：可以合并连续的片段
    
    Args:
        merge_consecutive: 是否合并连续或接近的片段
        min_gap: 如果两个片段之间的间隔小于此值（秒），则合并它们
    """
    if not merge_consecutive:
        extract_segments(video_file, timestamps, output_dir, video_id)
        return
    
    # 合并连续的片段
    merged_segments = []
    current_segment = None
    
    for entry in timestamps:
        if current_segment is None:
            current_segment = {
                'start': entry['start'],
                'end': entry['end'],
                'texts': [entry['text']]
            }
        else:
            # 检查是否应该合并（间隔小于min_gap）
            gap = entry['start'] - current_segment['end']
            if gap <= min_gap:
                # 合并片段
                current_segment['end'] = entry['end']
                current_segment['texts'].append(entry['text'])
            else:
                # 保存当前片段，开始新片段
                merged_segments.append(current_segment)
                current_segment = {
                    'start': entry['start'],
                    'end': entry['end'],
                    'texts': [entry['text']]
                }
    
    # 添加最后一个片段
    if current_segment:
        merged_segments.append(current_segment)
    
    print(f"原始片段数: {len(timestamps)}, 合并后: {len(merged_segments)}")
    
    # 转换为标准格式并提取
    standard_timestamps = [
        {
            'start': seg['start'],
            'end': seg['end'],
            'text': ' | '.join(seg['texts'])  # 合并文本
        }
        for seg in merged_segments
    ]
    
    extract_segments(video_file, standard_timestamps, output_dir, video_id)


def batch_extract_all(metadata_file: str, video_dir: str, output_dir: str):
    """批量处理所有视频"""
    extract_segments_from_metadata(
        metadata_file=metadata_file,
        video_dir=video_dir,
        output_dir=output_dir
    )


if __name__ == '__main__':
    import sys
    
    # 示例用法
    if len(sys.argv) < 2:
        print("用法:")
        print("  python extract_video_segments.py <metadata.csv> [video_dir] [output_dir] [video_id]")
        print("\n示例:")
        print("  python extract_video_segments.py youtube-asl_metadata.csv videos segments")
        print("  python extract_video_segments.py youtube-asl_metadata.csv videos segments 4eNt91uV02o")
        sys.exit(1)
    
    metadata_file = sys.argv[1]
    video_dir = sys.argv[2] if len(sys.argv) > 2 else "videos"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "segments"
    video_id = sys.argv[4] if len(sys.argv) > 4 else None
    
    extract_segments_from_metadata(
        metadata_file=metadata_file,
        video_dir=video_dir,
        output_dir=output_dir,
        video_id=video_id
    )



