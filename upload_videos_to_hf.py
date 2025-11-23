#!/usr/bin/env python3
"""
Upload video files to Hugging Face dataset repository
Supports both individual file upload and archive-based upload (for rate limit issues)
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi, login, create_repo, whoami
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import tarfile
import tempfile

def upload_videos_packaged(
    video_dir: str,
    dataset_name: str,
    token: str = None,
    archive_size_gb: float = 5.0,
    resume: bool = True
):
    """
    Upload videos by packaging them into tar.gz archives first
    This helps avoid rate limits when uploading many small files
    
    Args:
        video_dir: Path to directory containing video files
        dataset_name: Hugging Face dataset name
        token: Hugging Face authentication token
        archive_size_gb: Target size for each archive in GB (default: 5.0)
        resume: If True, skip archives that already exist (default: True)
    """
    # Login to Hugging Face
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Try to get saved token from Hugging Face cache
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    if token:
        print(f"Using Hugging Face token...")
        login(token=token, add_to_git_credential=False)
    else:
        print(f"Checking for saved credentials...")
        try:
            # Try whoami first to check if already authenticated
            user_info = whoami()
            print(f"Already authenticated as: {user_info.get('name', 'Unknown')}")
        except Exception:
            print(f"Error: Could not authenticate. Please run 'huggingface-cli login' or provide --token")
            raise
    
    # Verify authentication
    try:
        user_info = whoami()
        print(f"Logged in as: {user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"Warning: Could not verify authentication: {e}")
    
    # Initialize API
    api = HfApi()
    repo_id = dataset_name
    repo_type = "dataset"
    
    # Check/create repository
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository {repo_id} already exists")
    except Exception:
        print(f"Creating repository {repo_id}...")
        create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=False)
    
    # Get all video files
    video_path = Path(video_dir)
    video_files = sorted(list(video_path.glob("*.mp4")))
    
    print(f"\nFound {len(video_files)} video files")
    total_size_gb = sum(f.stat().st_size for f in video_files) / (1024**3)
    print(f"Total size: {total_size_gb:.2f} GB")
    
    # Check existing archives if resume
    existing_archives = set()
    if resume:
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
            existing_archives = {f for f in repo_files if f.startswith("videos_") and f.endswith(".tar.gz")}
            print(f"Found {len(existing_archives)} existing archive(s), will skip them")
        except Exception as e:
            print(f"Could not check existing files: {e}")
    
    # Create archives
    archive_size_bytes = archive_size_gb * (1024**3)
    archives_to_create = []
    current_archive_files = []
    current_size = 0
    archive_idx = 0
    
    print(f"\nCreating archives (target size: {archive_size_gb:.2f} GB per archive)...")
    
    for video_file in tqdm(video_files, desc="Organizing files"):
        file_size = video_file.stat().st_size
        
        # If adding this file would exceed the size limit, finalize current archive
        if current_size + file_size > archive_size_bytes and current_archive_files:
            archive_name = f"videos_{archive_idx:05d}.tar.gz"
            archives_to_create.append((archive_name, current_archive_files.copy()))
            current_archive_files = []
            current_size = 0
            archive_idx += 1
        
        current_archive_files.append(video_file)
        current_size += file_size
    
    # Don't forget the last archive
    if current_archive_files:
        archive_name = f"videos_{archive_idx:05d}.tar.gz"
        archives_to_create.append((archive_name, current_archive_files.copy()))
    
    print(f"Will create {len(archives_to_create)} archive(s)")
    
    # Filter out existing archives
    archives_to_upload = [
        (name, files) for name, files in archives_to_create 
        if name not in existing_archives
    ]
    
    if not archives_to_upload:
        print("All archives already exist!")
        return
    
    print(f"Need to create and upload {len(archives_to_upload)} archive(s)")
    
    # Create and upload archives
    with tempfile.TemporaryDirectory() as temp_dir:
        uploaded_count = 0
        failed_archives = []
        
        for archive_name, archive_files in tqdm(archives_to_upload, desc="Creating & uploading archives"):
            archive_path = Path(temp_dir) / archive_name
            
            try:
                # Create tar.gz archive
                print(f"\nCreating {archive_name} ({len(archive_files)} files)...")
                with tarfile.open(archive_path, "w:gz") as tar:
                    for video_file in tqdm(archive_files, desc=f"  Adding files to {archive_name}", leave=False):
                        tar.add(video_file, arcname=video_file.name)
                
                archive_size = archive_path.stat().st_size / (1024**2)
                print(f"  Archive size: {archive_size:.2f} MB")
                
                # Upload archive
                print(f"  Uploading {archive_name}...")
                api.upload_file(
                    path_or_fileobj=str(archive_path),
                    path_in_repo=archive_name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                uploaded_count += 1
                print(f"  ✓ Uploaded {archive_name}")
                
                # Small delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"  ✗ Error with {archive_name}: {e}")
                failed_archives.append(archive_name)
            
            # Clean up local archive
            if archive_path.exists():
                archive_path.unlink()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Upload Summary:")
    print(f"  Total archives: {len(archives_to_create)}")
    print(f"  Uploaded: {uploaded_count}")
    print(f"  Already existed: {len(archives_to_create) - len(archives_to_upload)}")
    print(f"  Failed: {len(failed_archives)}")
    if failed_archives:
        print(f"\nFailed archives: {failed_archives}")
    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*70}")


def upload_videos(
    video_dir: str,
    dataset_name: str,
    token: str = None,
    batch_size: int = 5,
    delay_seconds: float = 1.0,
    max_retries: int = 3,
    resume: bool = True
):
    """
    Upload video files individually with rate limiting and retry logic
    
    Args:
        video_dir: Path to directory containing video files
        dataset_name: Hugging Face dataset name
        token: Hugging Face authentication token
        batch_size: Number of files to upload in parallel (default: 5, lower for rate limits)
        delay_seconds: Delay between batches in seconds (default: 1.0)
        max_retries: Maximum retries for failed uploads (default: 3)
        resume: If True, skip files that already exist (default: True)
    """
    # Login to Hugging Face
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Try to get saved token from Hugging Face cache
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    if token:
        print(f"Using Hugging Face token...")
        login(token=token, add_to_git_credential=False)
    else:
        print(f"Checking for saved credentials...")
        try:
            # Try whoami first to check if already authenticated
            user_info = whoami()
            print(f"Already authenticated as: {user_info.get('name', 'Unknown')}")
        except Exception:
            print(f"Error: Could not authenticate. Please run 'huggingface-cli login' or provide --token")
            raise
    
    # Verify authentication
    try:
        user_info = whoami()
        print(f"Logged in as: {user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"Warning: Could not verify authentication: {e}")
    
    # Initialize API
    api = HfApi()
    repo_id = dataset_name
    repo_type = "dataset"
    
    # Check/create repository
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository {repo_id} already exists")
    except Exception:
        print(f"Creating repository {repo_id}...")
        create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=False)
    
    # Get all video files
    video_path = Path(video_dir)
    video_files = sorted(list(video_path.glob("*.mp4")))
    
    print(f"\nFound {len(video_files)} video files")
    total_size_gb = sum(f.stat().st_size for f in video_files) / (1024**3)
    print(f"Total size: {total_size_gb:.2f} GB")
    
    # Get already uploaded files if resume
    uploaded_files = set()
    if resume:
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
            uploaded_files = {f for f in repo_files if f.endswith('.mp4')}
            print(f"Found {len(uploaded_files)} already uploaded files, will skip them")
        except Exception as e:
            print(f"Could not check existing files: {e}")
    
    # Filter out already uploaded files
    files_to_upload = [f for f in video_files if f.name not in uploaded_files]
    
    if not files_to_upload:
        print(f"All files already uploaded!")
        return
    
    print(f"Uploading {len(files_to_upload)} files with {batch_size} parallel workers...")
    print(f"Delay between batches: {delay_seconds}s, Max retries: {max_retries}")
    
    def upload_single_file(video_file, retry_count=0):
        """Upload a single file with retry logic"""
        try:
            api.upload_file(
                path_or_fileobj=str(video_file),
                path_in_repo=video_file.name,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            return (True, video_file.name, None)
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "rate limit" in error_str.lower() or "429" in error_str:
                if retry_count < max_retries:
                    # Exponential backoff for rate limits
                    wait_time = (2 ** retry_count) * delay_seconds
                    time.sleep(wait_time)
                    return upload_single_file(video_file, retry_count + 1)
            return (False, video_file.name, error_str)
    
    # Upload in batches
    uploaded_count = 0
    failed_files = []
    
    for i in range(0, len(files_to_upload), batch_size):
        batch = files_to_upload[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(files_to_upload) + batch_size - 1) // batch_size
        
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} files)...")
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_file = {executor.submit(upload_single_file, f): f for f in batch}
            
            for future in as_completed(future_to_file):
                success, filename, error = future.result()
                if success:
                    uploaded_count += 1
                    print(f"  ✓ {filename}")
                else:
                    failed_files.append((filename, error))
                    print(f"  ✗ {filename}: {error[:100]}")
        
        # Delay between batches to avoid rate limiting
        if i + batch_size < len(files_to_upload):
            time.sleep(delay_seconds)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Upload Summary:")
    print(f"  Total files: {len(video_files)}")
    print(f"  Uploaded: {uploaded_count}")
    print(f"  Already existed: {len(uploaded_files)}")
    print(f"  Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files (showing first 10):")
        for filename, error in failed_files[:10]:
            print(f"  - {filename}: {error[:100]}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload videos to Hugging Face dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Package videos into archives (recommended for rate limits)
  python3 upload_videos_to_hf.py --mode package --archive_size 5.0
  
  # Upload individually with rate limiting
  python3 upload_videos_to_hf.py --mode individual --batch_size 3 --delay 2.0
        """
    )
    
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224",
        help="Path to directory containing video files"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PhoenixHu/sign_mllm_how_224",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["package", "individual"],
        default="package",
        help="Upload mode: 'package' (recommended, fewer files) or 'individual' (separate files)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token (optional if already logged in)"
    )
    
    # Package mode options
    parser.add_argument(
        "--archive_size",
        type=float,
        default=5.0,
        help="Target size for each archive in GB (package mode only, default: 5.0)"
    )
    
    # Individual mode options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of files to upload in parallel (individual mode only, default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between batches in seconds (individual mode only, default: 1.0)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for failed uploads (individual mode only, default: 3)"
    )
    
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't skip already uploaded files/archives"
    )
    
    args = parser.parse_args()
    
    if args.mode == "package":
        upload_videos_packaged(
            video_dir=args.video_dir,
            dataset_name=args.dataset_name,
            token=args.token,
            archive_size_gb=args.archive_size,
            resume=not args.no_resume
        )
    else:
        upload_videos(
            video_dir=args.video_dir,
            dataset_name=args.dataset_name,
            token=args.token,
            batch_size=args.batch_size,
            delay_seconds=args.delay,
            max_retries=args.max_retries,
            resume=not args.no_resume
        )
