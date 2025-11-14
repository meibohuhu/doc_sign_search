#!/usr/bin/env python3
"""
从 CSV 文件中筛选 <10秒的测试视频并生成 JSON 格式
"""

import csv
import json
import random
import os
import sys

def load_csv_and_filter(csv_path, max_duration=10.0, sample_size=100, seed=42):
    """
    从 CSV 文件中读取数据，筛选 <10秒的视频，并随机采样
    
    Args:
        csv_path: CSV 文件路径
        max_duration: 最大时长（秒），默认 10.0
        sample_size: 采样数量，默认 100
        seed: 随机种子，默认 42
    """
    # 设置随机种子以确保可复现
    random.seed(seed)
    
    filtered_videos = []
    
    print(f"Reading CSV file: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        # 使用 tab 分隔符（从文件格式看是 TSV）
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            try:
                # 获取字段
                sentence_id = row['SENTENCE_ID']
                sentence_name = row['SENTENCE_NAME']
                start_time = float(row['START_REALIGNED'])
                end_time = float(row['END_REALIGNED'])
                sentence = row['SENTENCE'].strip()
                
                # 计算时长
                duration = end_time - start_time
                
                # 筛选 <10秒的视频
                if duration < max_duration:
                    filtered_videos.append({
                        'sentence_id': sentence_id,
                        'sentence_name': sentence_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'sentence': sentence
                    })
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue
    
    print(f"Found {len(filtered_videos)} videos with duration < {max_duration} seconds")
    
    # 随机采样
    if len(filtered_videos) > sample_size:
        selected_videos = random.sample(filtered_videos, sample_size)
        print(f"Randomly sampled {sample_size} videos")
    else:
        selected_videos = filtered_videos
        print(f"Using all {len(selected_videos)} videos (less than {sample_size})")
    
    return selected_videos


def generate_json_format(videos):
    """
    将筛选的视频转换为 JSON 格式
    
    Args:
        videos: 视频列表，每个元素包含 sentence_id, sentence_name, duration, sentence
    """
    json_data = []
    
    for video in videos:
        sentence_name = video['sentence_name']
        
        # 生成 JSON 格式（与 sample120.json 格式一致）
        json_entry = {
            "id": sentence_name,
            "video": f"{sentence_name}.mp4",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat do the ASL signs in this video mean?"
                },
                {
                    "from": "gpt",
                    "value": video['sentence']
                }
            ]
        }
        
        json_data.append(json_entry)
    
    return json_data


def main():
    # 配置
    csv_path = '/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/how2sign_realigned_test.csv'
    output_path = '/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_test_videos_filtered.sample100.json'
    max_duration = 10.0  # 10秒
    sample_size = 100
    seed = 42
    
    print("=" * 60)
    print("Generating test JSON from CSV")
    print("=" * 60)
    print(f"Input CSV: {csv_path}")
    print(f"Output JSON: {output_path}")
    print(f"Max duration: {max_duration} seconds")
    print(f"Sample size: {sample_size}")
    print(f"Random seed: {seed}")
    print()
    
    # 读取并筛选
    filtered_videos = load_csv_and_filter(csv_path, max_duration, sample_size, seed)
    
    # 生成 JSON 格式
    json_data = generate_json_format(filtered_videos)
    
    # 保存 JSON
    print(f"\nSaving JSON to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully generated JSON with {len(json_data)} videos")
    
    # 打印统计信息
    durations = [v['duration'] for v in filtered_videos]
    print(f"\nStatistics:")
    print(f"  Total videos: {len(json_data)}")
    print(f"  Average duration: {sum(durations) / len(durations):.2f} seconds")
    print(f"  Min duration: {min(durations):.2f} seconds")
    print(f"  Max duration: {max(durations):.2f} seconds")
    
    # 打印前几个示例
    print(f"\nFirst 3 examples:")
    for i, entry in enumerate(json_data[:3], 1):
        print(f"  {i}. {entry['id']} - {entry['conversations'][1]['value'][:60]}...")


if __name__ == '__main__':
    main()
