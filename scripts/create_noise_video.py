#!/usr/bin/env python3
"""
Create a noisy video for testing attention visualization.
Generates a video with random noise frames (uniform or Gaussian).

Usage:
    python scripts/create_noise_video.py --output noise_video.mp4 --duration 3 --fps 30
    python scripts/create_noise_video.py --output noise_video.mp4 --duration 3 --fps 30 --noise_type gaussian
"""

import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm


def create_uniform_noise_frame(width, height, channels=3):
    """Create frame with uniform random noise [0, 255]"""
    noise = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
    return noise


def create_gaussian_noise_frame(width, height, channels=3, mean=128, std=50):
    """Create frame with Gaussian random noise"""
    noise = np.random.normal(mean, std, size=(height, width, channels))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return noise


def create_black_frame(width, height, channels=3):
    """Create completely black frame (all zeros)"""
    return np.zeros((height, width, channels), dtype=np.uint8)


def create_white_frame(width, height, channels=3):
    """Create completely white frame (all 255)"""
    return np.ones((height, width, channels), dtype=np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(description='Create noisy video for testing')
    parser.add_argument('--output', type=str, default='noise_video.mp4', 
                       help='Output video path')
    parser.add_argument('--width', type=int, default=224, help='Video width')
    parser.add_argument('--height', type=int, default=224, help='Video height')
    parser.add_argument('--duration', type=float, default=3.0,
                       help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')
    parser.add_argument('--noise_type', type=str, default='uniform',
                        choices=['uniform', 'gaussian', 'black', 'white'],
                        help='Type of noise to generate')
    parser.add_argument('--gaussian_mean', type=float, default=128,
                        help='Mean for Gaussian noise (0-255)')
    parser.add_argument('--gaussian_std', type=float, default=50,
                        help='Standard deviation for Gaussian noise')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='Video codec (mp4v, avc1, x264)')
    
    args = parser.parse_args()
    
    total_frames = int(args.duration * args.fps)
    
    print(f"🎬 Creating {args.noise_type} noise video...")
    print(f"   Size: {args.width}x{args.height}")
    print(f"   Duration: {args.duration}s @ {args.fps} fps")
    print(f"   Total frames: {total_frames}")
    print(f"   Codec: {args.codec}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    
    if not out.isOpened():
        print(f"❌ Failed to initialize video writer")
        return
    
    # Generate and write frames
    print(f"🎥 Generating frames...")
    for frame_idx in tqdm(range(total_frames), desc="Creating frames"):
        if args.noise_type == 'uniform':
            frame = create_uniform_noise_frame(args.width, args.height, 3)
        elif args.noise_type == 'gaussian':
            frame = create_gaussian_noise_frame(args.width, args.height, 3,
                                              args.gaussian_mean, args.gaussian_std)
        elif args.noise_type == 'black':
            frame = create_black_frame(args.width, args.height, 3)
        elif args.noise_type == 'white':
            frame = create_white_frame(args.width, args.height, 3)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # Print statistics from a sample frame
    if args.noise_type == 'uniform':
        sample_frame = create_uniform_noise_frame(args.width, args.height, 3)
    elif args.noise_type == 'gaussian':
        sample_frame = create_gaussian_noise_frame(args.width, args.height, 3,
                                                  args.gaussian_mean, args.gaussian_std)
    elif args.noise_type == 'black':
        sample_frame = create_black_frame(args.width, args.height, 3)
    elif args.noise_type == 'white':
        sample_frame = create_white_frame(args.width, args.height, 3)
    
    print(f"\n📊 Frame statistics:")
    print(f"   Min: {sample_frame.min()}, Max: {sample_frame.max()}")
    print(f"   Mean: {sample_frame.mean():.2f}, Std: {sample_frame.std():.2f}")
    if args.noise_type != 'black' and args.noise_type != 'white':
        print(f"   Mean per channel: R={sample_frame[:,:,0].mean():.2f}, "
              f"G={sample_frame[:,:,1].mean():.2f}, B={sample_frame[:,:,2].mean():.2f}")
    
    print(f"\n✅ Saved: {args.output}")
    if os.path.exists(args.output):
        file_size_mb = os.path.getsize(args.output) / (1024*1024)
        print(f"   File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()

