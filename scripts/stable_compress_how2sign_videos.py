#!/usr/bin/env python3
"""
Stable How2Sign Video Compression Script
Fixes shaking and tight cropping issues
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import glob
import numpy as np

def detect_stable_crop_region(video_path, sample_frames=10):
    """
    Detect a stable crop region by analyzing multiple frames
    
    Args:
        video_path: Path to video
        sample_frames: Number of frames to sample
    
    Returns:
        (x, y, w, h) stable crop region
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
    
    face_positions = []
    face_sizes = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        
        if len(faces) > 0:
            # Use largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = face
            face_positions.append((fx, fy))
            face_sizes.append((fw, fh))
    
    cap.release()
    
    if not face_positions:
        # Fallback: use center region - more generous for sign language
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            h, w = frame.shape[:2]
            center_w = int(w * 0.9)  # 90% of width (very generous)
            center_h = int(h * 0.95)  # 95% of height (include almost full body)
            center_x = (w - center_w) // 2
            center_y = int(h * 0.02)  # Start from top 2%, include even more lower body
            return (center_x, center_y, center_w, center_h)
        return None
    
    # Calculate stable region based on face positions
    avg_x = np.mean([pos[0] for pos in face_positions])
    avg_y = np.mean([pos[1] for pos in face_positions])
    avg_w = np.mean([size[0] for size in face_sizes])
    avg_h = np.mean([size[1] for size in face_sizes])
    
    # Expand around average face position - generous for sign language
    expand_factor = 5.2  # Generous expansion for full body gestures
    expanded_w = int(avg_w * expand_factor)
    expanded_h = int(avg_h * expand_factor * 1.8)  # Even more height for full torso and arms
    
    # Center the expanded region around average face position - shift down for more body
    crop_x = int(avg_x + avg_w/2 - expanded_w/2)
    crop_y = int(avg_y + avg_h/2 - expanded_h/6)  # Face in upper sixth, even more body below
    
    # Get frame dimensions
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    
    h, w = frame.shape[:2]
    
    # Ensure within frame bounds
    crop_x = max(0, min(crop_x, w - expanded_w))
    crop_y = max(0, min(crop_y, h - expanded_h))
    crop_w = min(expanded_w, w - crop_x)
    crop_h = min(expanded_h, h - crop_y)
    
    return (crop_x, crop_y, crop_w, crop_h)

def stable_crop_to_square(frame, crop_region, target_size=224):
    """
    Stable crop frame to square format using fixed region
    
    Args:
        frame: Input frame
        crop_region: Fixed crop region (x, y, w, h)
        target_size: Target square size
    
    Returns:
        Square cropped frame
    """
    if crop_region is None:
        # Fallback: center crop
        h, w = frame.shape[:2]
        min_dim = min(w, h)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        return cv2.resize(square_frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    crop_x, crop_y, crop_w, crop_h = crop_region
    
    # Make the crop region square
    crop_size = min(crop_w, crop_h)
    
    # Center the square crop within the region
    square_x = crop_x + (crop_w - crop_size) // 2
    square_y = crop_y + (crop_h - crop_size) // 2
    
    # Ensure within frame bounds
    h, w = frame.shape[:2]
    square_x = max(0, min(square_x, w - crop_size))
    square_y = max(0, min(square_y, h - crop_size))
    
    # Crop to square
    square_frame = frame[square_y:square_y+crop_size, square_x:square_x+crop_size]
    
    # Resize to target size
    resized_frame = cv2.resize(square_frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized_frame

def compress_video_stable(input_path, output_path, target_size=224, fps=24):
    """
    Compress video with stable cropping (no shaking)
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_size: Target square size
        fps: Target FPS
    """
    try:
        # Detect stable crop region first
        print(f"   🔍 Detecting stable crop region...")
        crop_region = detect_stable_crop_region(input_path)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {input_path}")
            return False
        
        # Get original video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))
        
        if not out.isOpened():
            print(f"❌ Failed to create output video: {output_path}")
            cap.release()
            return False
        
        # Process frames with stable cropping
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use stable crop region for all frames
            processed_frame = stable_crop_to_square(frame, crop_region, target_size)
            
            # Write frame to output video
            out.write(processed_frame)
            frame_count += 1
        
        # Clean up
        cap.release()
        out.release()
        
        # Verify output file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            print(f"✅ Stable compression: {os.path.basename(input_path)}")
            print(f"   Resolution: {original_width}x{original_height} → {target_size}x{target_size}")
            print(f"   FPS: {original_fps:.2f} → {fps}")
            print(f"   Size: {original_size:.2f}MB → {compressed_size:.2f}MB ({compression_ratio:.1f}x compression)")
            print(f"   Method: Stable crop region (no shaking)")
            return True
        else:
            print(f"❌ Output file not created properly: {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error compressing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Stable compress How2Sign videos (fixes shaking)")
    parser.add_argument("--input-dir", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips",
                       help="Input directory containing videos")
    parser.add_argument("--output-dir", type=str,
                       default="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224",
                       help="Output directory for compressed videos")
    parser.add_argument("--size", type=int, default=224,
                       help="Target square size (224x224)")
    parser.add_argument("--fps", type=int, default=24,
                       help="Target FPS")
    parser.add_argument("--overwrite", action="store_true", default=False,
                       help="Overwrite existing files")
    parser.add_argument("--test-samples", type=int, default=0,
                       help="Test on first N samples only (0 = all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all MP4 files
    video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    
    if not video_files:
        print(f"❌ No MP4 files found in {args.input_dir}")
        return
    
    # Limit to test samples if specified
    if args.test_samples > 0:
        video_files = video_files[:args.test_samples]
    
    print(f"🎬 Stable How2Sign Video Compression (No Shaking)")
    print("=" * 50)
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Format: {args.size}x{args.size} (square)")
    print(f"Target FPS: {args.fps}")
    print(f"Method: Stable crop region detection")
    print(f"Total Videos: {len(video_files)}")
    if args.test_samples > 0:
        print(f"Test Mode: Processing first {args.test_samples} samples only")
    print()
    
    # Process videos
    successful = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Stable compressing videos"):
        video_name = os.path.basename(video_path)
        output_path = os.path.join(args.output_dir, video_name)
        
        # Skip if output exists and not overwriting
        if os.path.exists(output_path) and not args.overwrite:
            print(f"⏭️  Skipping {video_name} (already exists)")
            continue
        
        # Compress video
        if compress_video_stable(video_path, output_path, args.size, args.fps):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 STABLE COMPRESSION SUMMARY:")
    print(f"   Total videos: {len(video_files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {(successful/len(video_files)*100):.1f}%")
    
    if successful > 0:
        print(f"\n✅ Stable videos saved to: {args.output_dir}")
        print(f"   Format: {args.size}x{args.size} square, {args.fps} FPS")
        print(f"   Fixed: No more shaking or tight cropping!")
    
    if failed > 0:
        print(f"\n⚠️  {failed} videos failed to compress. Check error messages above.")

if __name__ == "__main__":
    main()
