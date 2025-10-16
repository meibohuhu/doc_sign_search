#!/usr/bin/env python3
"""
Script to identify corrupted video files that cause training errors.
Based on the error patterns: 'nframes should in interval [2, 1], but got 0' and 'video_fps'
"""

import json
import os
import re
from pathlib import Path

def extract_failed_videos_from_log(log_file):
    """Extract video filenames that failed to load from training log"""
    failed_videos = set()
    
    print(f"🔍 Analyzing log file: {log_file}")
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Failed to load sample" in line and "Error:" in line:
                # Extract video filename from the error message
                match = re.search(r'video: ([^)]+\.mp4)', line)
                if match:
                    video_filename = match.group(1)
                    failed_videos.add(video_filename)
    
    return failed_videos

def main():
    # Paths
    json_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_filtered.json"
    log_file = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_20917630.txt"
    output_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_corrupted_removed.json"
    
    # Extract failed videos from log
    failed_videos = extract_failed_videos_from_log(log_file)
    print(f"❌ Found {len(failed_videos)} videos that failed to load during training")
    
    if failed_videos:
        print("📋 Failed videos:")
        for i, video in enumerate(sorted(failed_videos)[:10]):
            print(f"   {i+1}. {video}")
        if len(failed_videos) > 10:
            print(f"   ... and {len(failed_videos) - 10} more")
    
    # Load JSON data
    print(f"\n🔍 Loading JSON data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Original dataset size: {len(data)} entries")
    
    # Filter out corrupted videos
    cleaned_data = []
    removed_count = 0
    
    print("🧹 Removing corrupted videos...")
    for i, entry in enumerate(data):
        if i % 10000 == 0:
            print(f"   Processed {i}/{len(data)} entries...")
        
        video_filename = entry.get('video', '')
        if video_filename not in failed_videos:
            cleaned_data.append(entry)
        else:
            removed_count += 1
            if removed_count <= 10:
                print(f"   🗑️  Removing: {video_filename}")
    
    print(f"\n📈 Cleaning Results:")
    print(f"   Original entries: {len(data)}")
    print(f"   Corrupted videos: {len(failed_videos)}")
    print(f"   Removed entries: {removed_count}")
    print(f"   Cleaned entries: {len(cleaned_data)}")
    
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
    
    # Also create a list of corrupted videos for reference
    corrupted_list_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/corrupted_videos_list.txt"
    with open(corrupted_list_file, 'w') as f:
        f.write("Corrupted video files that cause training errors:\n")
        f.write("=" * 50 + "\n")
        for video in sorted(failed_videos):
            f.write(f"{video}\n")
    
    print(f"📝 Corrupted videos list saved to: {corrupted_list_file}")

if __name__ == "__main__":
    main()




