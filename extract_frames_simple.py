#!/usr/bin/env python3
"""
Simple script to extract frames from a video using available tools.
"""

import subprocess
import os
import sys
from pathlib import Path


def extract_frames(video_path, num_frames=5):
    """
    Extract evenly spaced frames from a video.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Try to use ffmpeg if available
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        use_ffmpeg = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg/ffprobe not found. Checking for alternative methods...")
        use_ffmpeg = False
    
    if use_ffmpeg:
        print("Using ffmpeg for extraction...")
        _extract_with_ffmpeg(video_path, num_frames)
    else:
        print("\nPlease install one of the following options:")
        print("  Option 1: Install ffmpeg")
        print("    - On Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("    - On RHEL/CentOS: sudo yum install ffmpeg")
        print("    - Or use conda: conda install -c conda-forge ffmpeg")
        print("\n  Option 2: Install Python libraries")
        print("    - pip install opencv-python")
        print("    - Or: pip install imageio imageio-ffmpeg")
        sys.exit(1)


def _extract_with_ffmpeg(video_path, num_frames):
    """Extract frames using ffmpeg."""
    # Get video information
    try:
        total_frames = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-count_frames', '-show_entries', 'stream=nb_read_frames',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
            text=True
        ).strip()
        
        duration = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
            text=True
        ).strip()
        
        total_frames = int(total_frames)
        duration = float(duration)
        
        print(f"\nVideo info:")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f}s")
        
    except subprocess.CalledProcessError as e:
        print(f"Error getting video info: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = video_path.parent / "extracted_frames"
    output_dir.mkdir(exist_ok=True)
    
    base_name = video_path.stem
    print(f"\nExtracting {num_frames} frames...")
    print(f"Output directory: {output_dir}")
    
    # Extract frames evenly spaced
    for i in range(num_frames):
        if num_frames == 1:
            frame_idx = 0
        else:
            frame_idx = int(i * (total_frames - 1) / (num_frames - 1))
        
        timestamp = frame_idx * duration / total_frames
        
        output_file = output_dir / f"{base_name}_frame_{i+1:02d}_{frame_idx}.png"
        
        print(f"  Extracting frame {i+1}/{num_frames} (index: {frame_idx}, time: {timestamp:.2f}s)...")
        
        try:
            subprocess.run(
                ['ffmpeg', '-i', str(video_path), '-ss', str(timestamp), 
                 '-vframes', '1', str(output_file), '-y', '-loglevel', 'error'],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame: {e}")
    
    print(f"\n✓ Extraction complete! Frames saved to: {output_dir}")


if __name__ == "__main__":
    video_path = "/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_320x320/8kAWy2YodzQ_12-1-rgb_front.mp4"
    extract_frames(video_path, num_frames=15)

