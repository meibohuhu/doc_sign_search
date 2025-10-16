#!/usr/bin/env python3
"""
Script to remove entries from segmented_train_videos_filtered.json 
where the corresponding video files are missing from the video directory.
"""

import json
import os
from pathlib import Path

def main():
    # Paths
    json_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_filtered.json"
    video_dir = "/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips/"
    output_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_cleaned.json"
    
    print("🔍 Loading JSON data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Original dataset size: {len(data)} entries")
    
    # Get list of available video files
    print("📁 Scanning video directory...")
    available_videos = set()
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            available_videos.add(filename)
    
    print(f"📹 Found {len(available_videos)} video files in directory")
    
    # Filter data to keep only entries with existing videos
    cleaned_data = []
    missing_count = 0
    
    print("🧹 Cleaning dataset...")
    for i, entry in enumerate(data):
        if i % 10000 == 0:
            print(f"   Processed {i}/{len(data)} entries...")
        
        video_filename = entry.get('video', '')
        if video_filename in available_videos:
            cleaned_data.append(entry)
        else:
            missing_count += 1
            if missing_count <= 10:  # Show first 10 missing files
                print(f"   ❌ Missing: {video_filename}")
    
    print(f"\n📈 Cleaning Results:")
    print(f"   Original entries: {len(data)}")
    print(f"   Available videos: {len(available_videos)}")
    print(f"   Missing videos: {missing_count}")
    print(f"   Cleaned entries: {len(cleaned_data)}")
    print(f"   Removed entries: {len(data) - len(cleaned_data)}")
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned dataset to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print("✅ Cleaning complete!")
    
    # Show some statistics
    if len(cleaned_data) > 0:
        print(f"\n📊 Sample of cleaned data:")
        for i, entry in enumerate(cleaned_data[:3]):
            print(f"   {i+1}. ID: {entry.get('id', 'N/A')}")
            print(f"      Video: {entry.get('video', 'N/A')}")
            print(f"      Conversations: {len(entry.get('conversations', []))}")

if __name__ == "__main__":
    main()




