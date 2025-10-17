#!/usr/bin/env python3
"""
Script to analyze the mismatch between JSON entries and available video files.
"""

import json
import os
from pathlib import Path

def main():
    # Paths
    json_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_filtered.json"
    video_dir = "/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips/"
    
    print("🔍 Loading JSON data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 JSON entries: {len(data)}")
    
    # Get list of available video files
    print("📁 Scanning video directory...")
    available_videos = set()
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            available_videos.add(filename)
    
    print(f"📹 Available videos: {len(available_videos)}")
    
    # Get videos referenced in JSON
    json_videos = set()
    for entry in data:
        video_filename = entry.get('video', '')
        if video_filename:
            json_videos.add(video_filename)
    
    print(f"📝 Videos in JSON: {len(json_videos)}")
    
    # Find mismatches
    missing_from_dir = json_videos - available_videos
    missing_from_json = available_videos - json_videos
    
    print(f"\n📈 Analysis Results:")
    print(f"   Videos in JSON but missing from directory: {len(missing_from_dir)}")
    print(f"   Videos in directory but missing from JSON: {len(missing_from_json)}")
    
    if missing_from_dir:
        print(f"\n❌ First 10 videos missing from directory:")
        for i, video in enumerate(sorted(missing_from_dir)[:10]):
            print(f"   {i+1}. {video}")
    
    if missing_from_json:
        print(f"\n➕ First 10 videos in directory but not in JSON:")
        for i, video in enumerate(sorted(missing_from_json)[:10]):
            print(f"   {i+1}. {video}")
    
    # Check if there are any actual missing videos
    if not missing_from_dir:
        print(f"\n✅ All videos referenced in JSON exist in the directory!")
        print(f"   The dataset is already clean.")
    else:
        print(f"\n⚠️  Found {len(missing_from_dir)} missing videos that need to be removed.")

if __name__ == "__main__":
    main()










