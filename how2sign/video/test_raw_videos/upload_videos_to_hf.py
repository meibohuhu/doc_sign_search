#!/usr/bin/env python3
"""
Upload Videos to Hugging Face Dataset
====================================

This script uploads video files to the DaydayPhoenix/test_raw_videos dataset on Hugging Face.
It handles both raw videos and segmented clips with proper directory structure.

Usage:
    python scripts/upload_videos_to_hf.py --upload-raw
    python scripts/upload_videos_to_hf.py --upload-segmented
    python scripts/upload_videos_to_hf.py --upload-all

Author: Sign Language LLM Project
Date: 2025-09-23
"""

import os
import sys
import argparse
from pathlib import Path
import time

def upload_directory(local_dir, repo_path, repo_id="DaydayPhoenix/test_raw_videos"):
    """Upload a directory to Hugging Face dataset"""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        print(f"🔄 Uploading {local_dir} to {repo_id}/{repo_path}")
        
        # Check if directory exists and has files
        local_path = Path(local_dir)
        if not local_path.exists():
            print(f"❌ Directory not found: {local_dir}")
            return False
            
        # Count video files
        video_files = list(local_path.glob("*.mp4")) + list(local_path.glob("*.avi"))
        print(f"📊 Found {len(video_files)} video files to upload")
        
        if len(video_files) == 0:
            print("⚠️  No video files found to upload")
            return False
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in video_files)
        print(f"💾 Total size: {total_size / (1024**3):.2f} GB")
        
        # Confirm upload
        response = input(f"Continue with upload? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Upload cancelled")
            return False
        
        # Upload the folder
        print("🚀 Starting upload... This may take a while for large files.")
        
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.mp4", "*.avi", "*.mov"],
            ignore_patterns=["*.log", "*.py", "*.sh", "*.csv", "*.json"]
        )
        
        print(f"✅ Successfully uploaded {repo_path}!")
        return True
        
    except ImportError:
        print("❌ Please install huggingface_hub:")
        print("   pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("💡 Make sure you're logged in: huggingface-cli login")
        return False

def check_auth():
    """Check if user is authenticated with Hugging Face"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"👤 Authenticated as: {user['name']}")
        return True
    except Exception as e:
        print("❌ Not authenticated with Hugging Face")
        print("💡 Please run: huggingface-cli login")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Upload videos to DaydayPhoenix/test_raw_videos dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload raw videos only
    python scripts/upload_videos_to_hf.py --upload-raw
    
    # Upload segmented clips only  
    python scripts/upload_videos_to_hf.py --upload-segmented
    
    # Upload everything
    python scripts/upload_videos_to_hf.py --upload-all
    
Dataset URL: https://huggingface.co/datasets/DaydayPhoenix/test_raw_videos
        """
    )
    
    parser.add_argument("--upload-raw", action="store_true", 
                       help="Upload raw videos directory")
    parser.add_argument("--upload-segmented", action="store_true",
                       help="Upload segmented clips directory")
    parser.add_argument("--upload-all", action="store_true",
                       help="Upload both directories")
    parser.add_argument("--repo-id", default="DaydayPhoenix/test_raw_videos",
                       help="Hugging Face dataset repository ID")
    
    args = parser.parse_args()
    
    print("🎬 Sign Language LLM - Video Uploader")
    print("=" * 50)
    
    # Check authentication
    if not check_auth():
        sys.exit(1)
    
    # Define paths
    raw_videos_dir = "how2sign/video/test_raw_videos/raw_videos"
    segmented_clips_dir = "how2sign/video/test_raw_videos/segmented_clips"
    
    success = True
    
    # Upload based on arguments
    if args.upload_all or args.upload_raw:
        print("\n🔸 Uploading Raw Videos...")
        if not upload_directory(raw_videos_dir, "raw_videos", args.repo_id):
            success = False
    
    if args.upload_all or args.upload_segmented:
        print("\n🔸 Uploading Segmented Clips...")
        if not upload_directory(segmented_clips_dir, "segmented_clips", args.repo_id):
            success = False
    
    if not (args.upload_all or args.upload_raw or args.upload_segmented):
        print("❌ Please specify what to upload:")
        print("   --upload-raw, --upload-segmented, or --upload-all")
        sys.exit(1)
    
    if success:
        print("\n🎉 Upload completed successfully!")
        print(f"🔗 Dataset: https://huggingface.co/datasets/{args.repo_id}")
        print("\n💡 Others can now download with:")
        print("   python scripts/download_videos.py")
    else:
        print("\n❌ Some uploads failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
