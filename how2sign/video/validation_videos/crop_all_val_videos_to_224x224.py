#!/usr/bin/env python3
"""
Process all how2sign validation videos to crop and resize to 224x224.
Uses center crop method (same as test script).
"""

import cv2
import os
import sys
import glob
from tqdm import tqdm

def process_video(input_path, output_path, target_size=224):
    """Process a single video: read, center crop, resize to target_size"""
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, f"Cannot open video {input_path}"
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate center crop (square)
    size = min(width, height)
    x = (width - size) // 2
    y = (height - size) // 2
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Center crop
        cropped = frame[y:y+size, x:x+size]
        
        # Resize to target size
        resized = cv2.resize(cropped, (target_size, target_size))
        
        # Write frame
        out.write(resized)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Verify output
    if os.path.exists(output_path) and frame_count > 0:
        return True, None
    else:
        return False, f"Output file not created or no frames processed"

def main():
    # Paths
    input_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_val_segment_clips_24fps'
    output_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_val_segment_clips_stable_224x224'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    all_videos = glob.glob(os.path.join(input_dir, '*.mp4'))
    all_videos.sort()
    
    print("🎬 Processing all videos to 224x224 (center crop)")
    print("=" * 60)
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📹 Total videos: {len(all_videos)}")
    print("=" * 60)
    sys.stdout.flush()
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_videos = []
    
    # Process with progress bar
    for input_video in tqdm(all_videos, desc="Processing videos"):
        output_video = os.path.join(output_dir, os.path.basename(input_video))
        
        # Skip if already processed
        if os.path.exists(output_video):
            skipped_count += 1
            continue
        
        success, error_msg = process_video(input_video, output_video, target_size=224)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_videos.append((os.path.basename(input_video), error_msg))
            tqdm.write(f"❌ Failed: {os.path.basename(input_video)} - {error_msg}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Processing complete!")
    print(f"   Success: {success_count}")
    print(f"   Skipped (already exists): {skipped_count}")
    print(f"   Failed: {failed_count}")
    
    if failed_videos:
        print(f"\n❌ Failed videos ({len(failed_videos)}):")
        for video, error in failed_videos[:10]:  # Show first 10
            print(f"   - {video}: {error}")
        if len(failed_videos) > 10:
            print(f"   ... and {len(failed_videos) - 10} more")
    
    print("=" * 60)

if __name__ == '__main__':
    main()



