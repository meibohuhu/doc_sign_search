#!/usr/bin/env python3
"""
视频处理流水线脚本
按照以下顺序处理视频：
1. clip_sign1news_videos.py - 根据 caption timestamps 裁剪视频
2. generate_sign1news_bbox.py - 生成 bbox
3. visualize_bbox.py - 使用 bbox 裁剪视频到 224x224
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n{'='*60}")
    print(f"步骤: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"\n❌ 错误: {description} 失败 (退出码: {result.returncode})")
        return False
    
    print(f"\n✅ 成功: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="视频处理流水线")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入视频目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录（将在此目录下创建子目录）"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="youtube-asl_metadata.csv",
        help="metadata CSV 文件路径（默认: youtube-asl_metadata.csv）"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="最终输出视频的目标尺寸（默认: 224x224）"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="internvl",
        help="Conda 环境名称（默认: internvl）"
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="跳过步骤1（裁剪视频 clips）"
    )
    parser.add_argument(
        "--skip-bbox",
        action="store_true",
        help="跳过步骤2（生成 bbox）"
    )
    parser.add_argument(
        "--skip-crop",
        action="store_true",
        help="跳过步骤3（应用 bbox 裁剪）"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        return 1
    
    # 创建输出目录结构
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    clips_dir = output_base / "clips"
    bbox_file = output_base / "clips_bbox.json"
    cropped_dir = output_base / "clips_cropped_224"
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 检查 metadata 文件
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = project_root / metadata_path
    if not metadata_path.exists():
        print(f"❌ 错误: metadata 文件不存在: {metadata_path}")
        return 1
    
    print(f"\n{'='*60}")
    print("视频处理流水线")
    print(f"{'='*60}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_base}")
    print(f"Metadata: {metadata_path}")
    print(f"目标尺寸: {args.target_size}x{args.target_size}")
    print(f"{'='*60}\n")
    
    # 步骤 1: 裁剪视频 clips
    if not args.skip_clip:
        clips_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "conda", "run", "-n", args.conda_env,
            "python", str(script_dir / "clip_sign1news_videos.py"),
            "--metadata", str(metadata_path),
            "--input-dir", str(input_dir),
            "--output-dir", str(clips_dir)
        ]
        
        if not run_command(cmd, "步骤 1: 根据 caption timestamps 裁剪视频"):
            return 1
        
        # 检查是否有生成的 clips
        clip_count = len(list(clips_dir.glob("*.mp4")))
        if clip_count == 0:
            print(f"⚠️  警告: 没有生成任何 clips，请检查 metadata 和输入视频")
            return 1
        print(f"✅ 生成了 {clip_count} 个 clips")
    else:
        print("⏭️  跳过步骤 1: 裁剪视频 clips")
        if not clips_dir.exists():
            print(f"❌ 错误: clips 目录不存在: {clips_dir}")
            return 1
    
    # 步骤 2: 生成 bbox
    if not args.skip_bbox:
        cmd = [
            "conda", "run", "-n", args.conda_env,
            "python", str(script_dir / "generate_sign1news_bbox.py"),
            "--clips-dir", str(clips_dir),
            "--output", str(bbox_file)
        ]
        
        if not run_command(cmd, "步骤 2: 生成 bbox"):
            return 1
        
        # 检查 bbox 文件
        if not bbox_file.exists():
            print(f"❌ 错误: bbox 文件未生成: {bbox_file}")
            return 1
        print(f"✅ bbox 文件已生成: {bbox_file}")
    else:
        print("⏭️  跳过步骤 2: 生成 bbox")
        if not bbox_file.exists():
            print(f"❌ 错误: bbox 文件不存在: {bbox_file}")
            return 1
    
    # 步骤 3: 应用 bbox 裁剪
    if not args.skip_crop:
        cropped_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "conda", "run", "-n", args.conda_env,
            "python", str(script_dir / "visualize_bbox.py"),
            "--bbox-file", str(bbox_file),
            "--clips-dir", str(clips_dir),
            "--output-dir", str(cropped_dir),
            "--target-size", str(args.target_size)
        ]
        
        if not run_command(cmd, f"步骤 3: 使用 bbox 裁剪视频到 {args.target_size}x{args.target_size}"):
            return 1
        
        # 检查输出
        cropped_count = len(list(cropped_dir.glob("*.mp4")))
        print(f"✅ 生成了 {cropped_count} 个裁剪后的视频")
    else:
        print("⏭️  跳过步骤 3: 应用 bbox 裁剪")
    
    # 总结
    print(f"\n{'='*60}")
    print("✅ 处理完成！")
    print(f"{'='*60}")
    print(f"输出目录结构:")
    print(f"  - Clips: {clips_dir}")
    print(f"  - Bbox: {bbox_file}")
    print(f"  - Cropped videos: {cropped_dir}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

