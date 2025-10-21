#!/usr/bin/env python3
"""
DailyMoth-Format How2Sign Video Compression Script
Compress How2Sign videos to match DailyMoth format exactly
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import glob
import numpy as np

def detect_face_and_crop_region(frame):
    """
    Detect face and determine optimal crop region for person
    
    Args:
        frame: Input frame
    
    Returns:
        (x, y, w, h) crop region or None
    """
    h, w = frame.shape[:2]
    
    # Try face detection first
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    
    if len(faces) > 0:
        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = face
        
        # Expand around face to include more of the person
        # Face is usually in upper portion, so expand more downward
        expand_factor = 2.5  # Expand face region by 2.5x
        
        # Calculate expanded region
        expanded_w = int(fw * expand_factor)
        expanded_h = int(fh * expand_factor * 1.5)  # More height for torso
        
        # Center the expanded region around the face
        center_x = fx + fw // 2
        center_y = fy + fh // 2
        
        crop_x = center_x - expanded_w // 2
        crop_y = center_y - expanded_h // 3  # Face is in upper third
        
        # Ensure within frame bounds
        crop_x = max(0, min(crop_x, w - expanded_w))
        crop_y = max(0, min(crop_y, h - expanded_h))
        crop_w = min(expanded_w, w - crop_x)
        crop_h = min(expanded_h, h - crop_y)
        
        return (crop_x, crop_y, crop_w, crop_h)
    
    # Fallback: use center region
    center_w = int(w * 0.5)  # 50% of width
    center_h = int(h * 0.7)  # 70% of height
    center_x = (w - center_w) // 2
    center_y = (h - center_h) // 2
    
    return (center_x, center_y, center_w, center_h)

def crop_to_square(frame, target_size=224):
    """
    Crop frame to square format (same as DailyMoth)
    
    Args:
        frame: Input frame
        target_size: Target square size
    
    Returns:
        Square cropped frame
    """
    h, w = frame.shape[:2]
    
    # Detect optimal crop region
    crop_region = detect_face_and_crop_region(frame)
    
    if crop_region is not None:
        crop_x, crop_y, crop_w, crop_h = crop_region
        
        # Make the crop region square
        crop_size = min(crop_w, crop_h)
        
        # Center the square crop
        square_x = crop_x + (crop_w - crop_size) // 2
        square_y = crop_y + (crop_h - crop_size) // 2
        
        # Ensure within frame bounds
        square_x = max(0, min(square_x, w - crop_size))
        square_y = max(0, min(square_y, h - crop_size))
        
        # Crop to square
        square_frame = frame[square_y:square_y+crop_size, square_x:square_x+crop_size]
        
        # Resize to target size
        resized_frame = cv2.resize(square_frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return resized_frame
    
    # Fallback: center crop to square
    min_dim = min(w, h)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized_frame = cv2.resize(square_frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized_frame

def compress_to_dailymoth_format(input_path, output_path, target_size=224, fps=24):
    """
    Compress video to match DailyMoth format exactly
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_size: Target square size (224x224)
        fps: Target FPS (24)
    """
    try:
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
        
        # Set up video writer with compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec (compatible)
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))
        
        if not out.isOpened():
            print(f"❌ Failed to create output video: {output_path}")
            cap.release()
            return False
        
        # Process frames
        frame_count = 0
        face_detected_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop to square format (same as DailyMoth)
            processed_frame = crop_to_square(frame, target_size)
            
            # Count face detections (only on first few frames for efficiency)
            if frame_count < 5:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(frame, 1.1, 4)
                if len(faces) > 0:
                    face_detected_count += 1
            
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
            
            print(f"✅ DailyMoth-format compression: {os.path.basename(input_path)}")
            print(f"   Resolution: {original_width}x{original_height} → {target_size}x{target_size}")
            print(f"   FPS: {original_fps:.2f} → {fps}")
            print(f"   Size: {original_size:.2f}MB → {compressed_size:.2f}MB ({compression_ratio:.1f}x compression)")
            print(f"   Format: Square (1:1 aspect ratio)")
            print(f"   Face detection: {face_detected_count}/5 frames")
            return True
        else:
            print(f"❌ Output file not created properly: {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error compressing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compress How2Sign videos to DailyMoth format")
    parser.add_argument("--input-dir", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips",
                       help="Input directory containing videos")
    parser.add_argument("--output-dir", type=str,
                       default="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_dailymoth_format",
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
    
    print(f"🎬 DailyMoth-Format How2Sign Video Compression")
    print("=" * 50)
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Format: {args.size}x{args.size} (square)")
    print(f"Target FPS: {args.fps}")
    print(f"Codec: MP4V (compatible)")
    print(f"Total Videos: {len(video_files)}")
    if args.test_samples > 0:
        print(f"Test Mode: Processing first {args.test_samples} samples only")
    print()
    
    # Process videos
    successful = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Compressing to DailyMoth format"):
        video_name = os.path.basename(video_path)
        output_path = os.path.join(args.output_dir, video_name)
        
        # Skip if output exists and not overwriting
        if os.path.exists(output_path) and not args.overwrite:
            print(f"⏭️  Skipping {video_name} (already exists)")
            continue
        
        # Compress video
        if compress_to_dailymoth_format(video_path, output_path, args.size, args.fps):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 DAILYMOTH-FORMAT COMPRESSION SUMMARY:")
    print(f"   Total videos: {len(video_files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {(successful/len(video_files)*100):.1f}%")
    
    if successful > 0:
        print(f"\n✅ DailyMoth-format videos saved to: {args.output_dir}")
        print(f"   Format: {args.size}x{args.size} square, {args.fps} FPS, MP4V codec")
        print(f"   Matches DailyMoth format exactly!")
    
    if failed > 0:
        print(f"\n⚠️  {failed} videos failed to compress. Check error messages above.")

if __name__ == "__main__":
    main()
