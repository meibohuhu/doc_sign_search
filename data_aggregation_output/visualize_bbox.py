"""
使用 bbox 裁剪视频，验证 bbox 是否正确
读取 bbox JSON 文件，对每个视频 clip 应用 bbox 裁剪，输出裁剪后的视频
"""

import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def crop_video_with_bbox(input_video, output_video, bbox, width, height, target_size=224):
    """
    使用 bbox 裁剪视频，并调整到目标尺寸
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        bbox: 归一化的 bbox [x0, y0, x1, y1]（相对于图像宽高）
        width: 原始视频宽度
        height: 原始视频高度
        target_size: 目标输出尺寸（默认 224x224）
    """
    # 将归一化的 bbox 转换为像素坐标
    x0 = int(bbox[0] * width)
    y0 = int(bbox[1] * height)
    x1 = int(bbox[2] * width)
    y1 = int(bbox[3] * height)
    
    # 确保坐标有效
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"  错误: 无法打开视频 {input_video}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 创建输出视频写入器，输出尺寸为 target_size x target_size
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_size, target_size))
    
    if not out.isOpened():
        print(f"  错误: 无法创建输出视频 {output_video}")
        cap.release()
        return False
    
    # 计算裁剪区域的宽高
    crop_width = x1 - x0
    crop_height = y1 - y0
    crop_aspect = crop_width / crop_height
    
    # 读取并裁剪每一帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 裁剪帧
        cropped_frame = frame[y0:y1, x0:x1]
        
        # 保持宽高比调整到目标尺寸
        # 计算缩放比例，使较长的边等于 target_size
        if crop_aspect > 1.0:  # 宽度 > 高度
            new_width = target_size
            new_height = int(target_size / crop_aspect)
        else:  # 高度 >= 宽度
            new_height = target_size
            new_width = int(target_size * crop_aspect)
        
        # 先缩放到保持宽高比的尺寸
        resized_frame = cv2.resize(cropped_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 创建 224x224 的黑色背景
        final_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        
        # 将调整后的帧放到中心位置
        final_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        # 写入输出视频
        out.write(final_frame)
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    return frame_count > 0


def main():
    parser = argparse.ArgumentParser(description="使用 bbox 裁剪视频，验证 bbox 是否正确")
    parser.add_argument(
        "--bbox-file",
        type=str,
        required=True,
        help="bbox JSON 文件路径"
    )
    parser.add_argument(
        "--clips-dir",
        type=str,
        required=True,
        help="原始视频 clips 目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出裁剪后的视频目录"
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="只处理指定的 clip ID（用于测试）"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="输出视频的目标尺寸（默认 224x224）"
    )
    
    args = parser.parse_args()
    
    # 读取 bbox 文件
    print(f"正在读取 bbox 文件: {args.bbox_file}")
    with open(args.bbox_file, 'r', encoding='utf-8') as f:
        bbox_dict = json.load(f)
    
    print(f"找到 {len(bbox_dict)} 个 bbox")
    print(f"输入目录: {args.clips_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标尺寸: {args.target_size}x{args.target_size}")
    print("-" * 50)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理视频
    success_count = 0
    fail_count = 0
    
    # 如果指定了 clip_id，只处理该 clip
    clip_ids = [args.clip_id] if args.clip_id else list(bbox_dict.keys())
    
    for clip_id in tqdm(clip_ids, desc="处理视频"):
        if clip_id not in bbox_dict:
            print(f"  ⚠ 跳过: {clip_id} (不在 bbox 文件中)")
            continue
        
        # 查找输入视频
        input_video = os.path.join(args.clips_dir, f"{clip_id}.mp4")
        if not os.path.exists(input_video):
            print(f"  ✗ 找不到视频: {clip_id}")
            fail_count += 1
            continue
        
        # 获取视频尺寸
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"  ✗ 无法打开视频: {clip_id}")
            fail_count += 1
            continue
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # 输出视频路径
        output_video = os.path.join(args.output_dir, f"{clip_id}_cropped.mp4")
        
        # 如果输出已存在，跳过
        if os.path.exists(output_video):
            print(f"  ℹ 跳过（已存在）: {clip_id}")
            success_count += 1
            continue
        
        # 获取 bbox
        bbox = bbox_dict[clip_id]
        
        # 裁剪视频并调整到目标尺寸
        if crop_video_with_bbox(input_video, output_video, bbox, width, height, args.target_size):
            success_count += 1
        else:
            print(f"  ✗ 裁剪失败: {clip_id}")
            fail_count += 1
    
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()


