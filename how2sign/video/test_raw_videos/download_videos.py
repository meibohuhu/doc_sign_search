#!/usr/bin/env python3
"""
Download Videos for Sign Language LLM Project
============================================

This script downloads video files from the official dataset repository.
Dataset: https://huggingface.co/datasets/DaydayPhoenix/test_raw_videos

Usage:
    python how2sign/video/test_raw_videos/download_videos.py
    python scripts/download_videos.py --output-dir custom/path/

Author: Sign Language LLM Project
Date: 2025-09-23
"""

import os
import sys
import argparse
import time
from pathlib import Path

def check_existing_videos(output_dir):
    """Check for existing video files in the output directory"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return [], 0
    
    # Find all video files
    video_extensions = ["*.mp4", "*.avi", "*.mov"]
    existing_files = []
    total_size = 0
    
    for ext in video_extensions:
        for file_path in output_path.glob(ext):
            existing_files.append(file_path)
            total_size += file_path.stat().st_size
    
    return existing_files, total_size

def format_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def download_with_retry(repo_id, output_dir, max_retries=3, base_delay=5):
    """Download with retry logic for handling rate limits and network issues"""
    from huggingface_hub import snapshot_download
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Download attempt {attempt + 1}/{max_retries}...")
            
            # Increase delay with each retry (exponential backoff)
            if attempt > 0:
                delay = base_delay * (2 ** attempt)
                print(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)
            
            # Download with timeout and resume capability
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                allow_patterns=["*.mp4", "*.avi", "*.mov"],
                resume_download=True,
                max_workers=2,  # Reduce concurrent downloads to avoid rate limits
                token=None  # Use anonymous access to avoid token issues
            )
            
            print("✅ Download successful!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Attempt {attempt + 1} failed: {error_msg}")
            
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                print("🚫 Rate limit detected - will retry with longer delay")
                continue
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                print("🌐 Network issue detected - will retry")
                continue
            elif attempt == max_retries - 1:
                print("❌ All retry attempts failed")
                raise e
            else:
                print("🔄 Retrying...")
                continue
    
    return False

def download_videos(output_dir="how2sign/video/test_raw_videos/", force_download=False, max_retries=3, retry_delay=10):
    """Download videos from the official Hugging Face dataset"""
    try:
        from huggingface_hub import snapshot_download
        
        print("🔄 Checking existing videos...")
        print("📍 Source: https://huggingface.co/datasets/DaydayPhoenix/test_raw_videos")
        print(f"📁 Target: {output_dir}")
        
        # Check existing files
        existing_files, existing_size = check_existing_videos(output_dir)
        
        if existing_files and not force_download:
            print(f"📊 Found {len(existing_files)} existing video files ({format_size(existing_size)})")
            print("⏭️  Skipping download - files already exist")
            print("💡 Use --force-download to re-download all files")
            
            # Show first few existing files
            print(f"📝 Existing files:")
            for i, file in enumerate(existing_files[:5]):
                file_size = format_size(file.stat().st_size)
                print(f"   - {file.name} ({file_size})")
            if len(existing_files) > 5:
                print(f"   ... and {len(existing_files) - 5} more files")
            
            return existing_files
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if force_download and existing_files:
            print(f"🔄 Force download enabled - will re-download all {len(existing_files)} files")
        
        print("🔄 Downloading videos from official dataset...")
        print("💡 Using retry logic to handle rate limits and network issues...")
        
        # Download dataset with retry logic
        success = download_with_retry(
            repo_id="DaydayPhoenix/test_raw_videos",
            output_dir=output_dir,
            max_retries=max_retries,
            base_delay=retry_delay
        )
        
        if not success:
            print("❌ Download failed after all retry attempts")
            return []
        
        # Count downloaded files
        video_files, total_size = check_existing_videos(output_dir)
        
        print(f"✅ Download completed!")
        print(f"📊 Total video files: {len(video_files)} ({format_size(total_size)})")
        
        if video_files:
            print(f"📝 First few files:")
            for i, file in enumerate(video_files[:5]):
                file_size = format_size(file.stat().st_size)
                print(f"   - {file.name} ({file_size})")
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
    python scripts/download_videos.py --force-download
    
Dataset URL: https://huggingface.co/datasets/DaydayPhoenix/test_raw_videos
        """
    )
    
    parser.add_argument(
        "--output-dir", 
        default="how2sign/video/test_raw_videos/",
        help="Output directory for downloaded videos (default: how2sign/video/test_raw_videos/)"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of all videos, even if they already exist"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts (default: 3)"
    )
    
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        help="Base delay in seconds between retries (default: 10)"
    )
    
    args = parser.parse_args()
    
    print("🎬 Sign Language LLM - Video Downloader")
    print("=" * 50)
    
    download_videos(args.output_dir, args.force_download, args.max_retries, args.retry_delay)
    
    print("\n🚀 Ready to run your sign language analysis!")
    print("💡 Next steps:")
    print("   1. Check downloaded videos in:", args.output_dir)
    print("   2. Run your analysis script: python playground/demo/simpleQA_metrics.py")

if __name__ == "__main__":
    main()
