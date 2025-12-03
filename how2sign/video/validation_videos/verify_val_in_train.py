#!/usr/bin/env python3
"""
Verify that all validation videos are in the train folder
"""

import os
from pathlib import Path

def main():
    val_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_val_segment_clips_stable_224x224'
    train_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224'
    
    # Get all video files from val directory
    val_videos = set()
    for file in Path(val_dir).glob('*.mp4'):
        val_videos.add(file.name)
    
    # Get all video files from train directory
    train_videos = set()
    for file in Path(train_dir).glob('*.mp4'):
        train_videos.add(file.name)
    
    # Find videos in val but not in train
    missing_videos = val_videos - train_videos
    
    print("=" * 60)
    print("Verification Results")
    print("=" * 60)
    print(f"Validation videos: {len(val_videos)}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos found in train: {len(val_videos & train_videos)}")
    print(f"Val videos NOT in train: {len(missing_videos)}")
    print("=" * 60)
    
    if missing_videos:
        print(f"\n❌ Missing videos ({len(missing_videos)}):")
        for video in sorted(missing_videos)[:20]:  # Show first 20
            print(f"   - {video}")
        if len(missing_videos) > 20:
            print(f"   ... and {len(missing_videos) - 20} more")
    else:
        print("\n✅ All validation videos are in the train folder!")
    
    print("=" * 60)

if __name__ == '__main__':
    main()



