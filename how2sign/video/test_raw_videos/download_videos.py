#!/usr/bin/env python3
"""
Download Videos for Sign Language LLM Project
============================================

This script downloads video files from the official dataset repository.
Dataset: https://huggingface.co/datasets/DaydayPhoenix/test_raw_videos

Usage:
    python scripts/download_videos.py
    python scripts/download_videos.py --output-dir custom/path/

Author: Sign Language LLM Project
Date: 2025-09-23
"""

import os
import sys
import argparse
from pathlib import Path

def download_videos(output_dir="how2sign/video/test_raw_videos/"):
    """Download videos from the official Hugging Face dataset"""
    try:
        from huggingface_hub import snapshot_download
        
        print("🔄 Downloading videos from official dataset...")
        print("📍 Source: https://huggingface.co/datasets/DaydayPhoenix/test_raw_videos")
        print(f"📁 Target: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        snapshot_download(
            repo_id="DaydayPhoenix/test_raw_videos",
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns=["*.mp4", "*.avi", "*.mov"]  # Video files only
        )
        
        # Count downloaded files
        video_files = list(Path(output_dir).glob("*.mp4")) + \
                     list(Path(output_dir).glob("*.avi")) + \
                     list(Path(output_dir).glob("*.mov"))
        
        print(f"✅ Download completed!")
        print(f"📊 Downloaded {len(video_files)} video files")
        
        if video_files:
            print(f"📝 First few files:")
            for i, file in enumerate(video_files[:5]):
                print(f"   - {file.name}")
            if len(video_files) > 5:
                print(f"   ... and {len(video_files) - 5} more files")
        
    except ImportError:
        print("❌ Please install huggingface_hub:")
        print("   pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Make sure you have internet connection and the dataset is accessible")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Download videos from DaydayPhoenix/test_raw_videos dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_videos.py
    python scripts/download_videos.py --output-dir videos/
    
Dataset URL: https://huggingface.co/datasets/DaydayPhoenix/test_raw_videos
        """
    )
    
    parser.add_argument(
        "--output-dir", 
        default="how2sign/video/test_raw_videos/",
        help="Output directory for downloaded videos (default: how2sign/video/test_raw_videos/)"
    )
    
    args = parser.parse_args()
    
    print("🎬 Sign Language LLM - Video Downloader")
    print("=" * 50)
    
    download_videos(args.output_dir)
    
    print("\n🚀 Ready to run your sign language analysis!")
    print("💡 Next steps:")
    print("   1. Check downloaded videos in:", args.output_dir)
    print("   2. Run your analysis script: python playground/demo/simpleQA_metrics.py")

if __name__ == "__main__":
    main()
