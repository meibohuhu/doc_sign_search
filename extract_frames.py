#!/usr/bin/env python3
"""
Script to extract frames from a video.
"""

import cv2
import numpy as np
import os
from pathlib import Path


def extract_frames(video_path, num_frames=5, output_dir=None):
    """
    Extract evenly spaced frames from a video.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: 5)
        output_dir: Directory to save frames (default: same directory as video)
    
    Returns:
        List of paths to extracted frame images
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Calculate frame indices to extract
    if num_frames > total_frames:
        print(f"Warning: Requested {num_frames} frames but video only has {total_frames} frames.")
        frame_indices = list(range(total_frames))
        num_frames = total_frames
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    print(f"\nExtracting {num_frames} frames at indices: {frame_indices}")
    
    # Extract and save frames
    saved_frames = []
    base_name = video_path.stem
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Save frame
        output_filename = f"{base_name}_frame_{i+1:02d}_index_{frame_idx:05d}.png"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), frame)
        
        print(f"  Saved frame {i+1}/{num_frames}: {output_path.name}")
        saved_frames.append(str(output_path))
    
    cap.release()
    print(f"\nExtraction complete! Saved {len(saved_frames)} frames to {output_dir}")
    
    return saved_frames


if __name__ == "__main__":
    video_path = "/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_320x320/fzDHRCKr7wU_4-8-rgb_front.mp4"
    
    extract_frames(video_path, num_frames=5)

