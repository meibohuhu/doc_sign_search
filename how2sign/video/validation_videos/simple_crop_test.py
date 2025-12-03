#!/usr/bin/env python3
"""
Simple test script to crop and resize two videos to 224x224
Uses center crop for testing (will be replaced with person detection later)
"""

import cv2
import os
import sys

def process_video(input_path, output_path, target_size=224):
    """Process a single video: read, center crop, resize to target_size"""
    print(f"Processing: {os.path.basename(input_path)}")
    sys.stdout.flush()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  ❌ Error: Cannot open video {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Original: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    sys.stdout.flush()
    
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
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
            sys.stdout.flush()
    
    cap.release()
    out.release()
    
    print(f"  ✅ Done! Processed {frame_count} frames")
    print(f"  Output: {output_path}")
    sys.stdout.flush()
    
    # Verify output
    if os.path.exists(output_path):
        cap_check = cv2.VideoCapture(output_path)
        if cap_check.isOpened():
            out_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Verified: {out_width}x{out_height}")
            cap_check.release()
            return True
    
    return False

def main():
    import random
    import glob
    
    # Get all videos from input directory
    input_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_val_segment_clips_24fps'
    all_videos = glob.glob(os.path.join(input_dir, '*.mp4'))
    
    # Randomly select 20 videos
    random.seed(42)  # For reproducibility
    selected_videos = random.sample(all_videos, min(20, len(all_videos)))
    selected_videos.sort()  # Sort for consistent output
    
    output_dir = '/shared/rc/llm-gen-agent/mhu/videos/how2sign_val_segment_clips_stable_224x224_test'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎬 Testing video cropping to 224x224 (center crop)")
    print("=" * 60)
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📹 Total videos found: {len(all_videos)}")
    print(f"🎲 Randomly selected: {len(selected_videos)} videos")
    print("=" * 60)
    sys.stdout.flush()
    
    success_count = 0
    failed_count = 0
    
    for i, input_video in enumerate(selected_videos, 1):
        if not os.path.exists(input_video):
            print(f"❌ Video {i} not found: {input_video}")
            failed_count += 1
            continue
        
        output_video = os.path.join(output_dir, os.path.basename(input_video))
        
        # Skip if already processed
        if os.path.exists(output_video):
            print(f"\n[{i}/{len(selected_videos)}] {os.path.basename(input_video)}")
            print(f"  ⏭️  Already exists, skipping...")
            success_count += 1
            continue
        
        print(f"\n[{i}/{len(selected_videos)}]")
        sys.stdout.flush()
        
        success = process_video(input_video, output_video, target_size=224)
        if success:
            print(f"  ✅ Success!")
            success_count += 1
        else:
            print(f"  ❌ Failed!")
            failed_count += 1
        sys.stdout.flush()
    
    print("\n" + "=" * 60)
    print(f"✅ Test complete!")
    print(f"   Success: {success_count}/{len(selected_videos)}")
    print(f"   Failed: {failed_count}/{len(selected_videos)}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()

