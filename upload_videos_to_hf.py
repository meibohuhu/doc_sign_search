#!/usr/bin/env python3
"""
Upload video/mask files to Hugging Face dataset repository
Supports both individual file upload and archive-based upload (for rate limit issues)
Supports .mp4 (videos) and .npz (masks) files
Also supports uploading model checkpoints to Hugging Face model repository

Hugging Face Token Configuration:
==================================
The script supports multiple ways to provide your Hugging Face token:

1. Command line argument (recommended for scripts):
   --token YOUR_HF_TOKEN

2. Environment variable (recommended for security):
   export HF_TOKEN=YOUR_HF_TOKEN
   # or
   export HUGGINGFACE_TOKEN=YOUR_HF_TOKEN

3. Hugging Face CLI login (most convenient):
   huggingface-cli login
   # This will save the token in ~/.cache/huggingface/ and the script will use it automatically

4. Python environment variable:
   import os
   os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

Where to get your token:
- Go to https://huggingface.co/settings/tokens
- Create a new token with "Write" permissions for uploading files
- Copy the token and use one of the methods above

Priority order (script will check in this order):
1. --token command line argument
2. HF_TOKEN environment variable
3. HUGGINGFACE_TOKEN environment variable
4. Saved token from huggingface-cli login
5. If none found, will prompt for authentication
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
import fnmatch

def upload_videos_packaged(
    video_dir: str,
    dataset_name: str,
    token: str = None,
    archive_size_gb: float = 5.0,
    resume: bool = True,
    file_ext: str = "*.npz"
):
    """
    Upload files by packaging them into tar.gz archives first
    This helps avoid rate limits when uploading many small files
    
    Args:
        video_dir: Path to directory containing files
        dataset_name: Hugging Face dataset name
        token: Hugging Face authentication token
        archive_size_gb: Target size for each archive in GB (default: 5.0)
        resume: If True, skip archives that already exist (default: True)
        file_ext: File extension pattern to match (default: "*.npz")
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
    
    # Get all files matching the extension
    file_path = Path(video_dir)
    all_files = sorted(list(file_path.glob(file_ext)))
    
    # Determine archive prefix based on file extension
    if file_ext == "*.npz":
        archive_prefix = "masks_"
        file_type = "mask"
    elif file_ext == "*.mp4":
        archive_prefix = "videos_"
        file_type = "video"
    else:
        archive_prefix = "files_"
        file_type = "file"
    
    print(f"\nFound {len(all_files)} {file_type} files")
    total_size_gb = sum(f.stat().st_size for f in all_files) / (1024**3)
    print(f"Total size: {total_size_gb:.2f} GB")
    
    # Check existing archives if resume
    existing_archives = set()
    if resume:
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
            existing_archives = {f for f in repo_files if f.startswith(archive_prefix) and f.endswith(".tar.gz")}
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
    
    for file in tqdm(all_files, desc="Organizing files"):
        file_size = file.stat().st_size
        
        # If adding this file would exceed the size limit, finalize current archive
        if current_size + file_size > archive_size_bytes and current_archive_files:
            archive_name = f"{archive_prefix}{archive_idx:05d}.tar.gz"
            archives_to_create.append((archive_name, current_archive_files.copy()))
            current_archive_files = []
            current_size = 0
            archive_idx += 1
        
        current_archive_files.append(file)
        current_size += file_size
    
    # Don't forget the last archive
    if current_archive_files:
        archive_name = f"{archive_prefix}{archive_idx:05d}.tar.gz"
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
                    for file in tqdm(archive_files, desc=f"  Adding files to {archive_name}", leave=False):
                        tar.add(file, arcname=file.name)
                
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
    resume: bool = True,
    file_ext: str = "*.npz"
):
    """
    Upload files individually with rate limiting and retry logic
    
    Args:
        video_dir: Path to directory containing files
        dataset_name: Hugging Face dataset name
        token: Hugging Face authentication token
        batch_size: Number of files to upload in parallel (default: 5, lower for rate limits)
        delay_seconds: Delay between batches in seconds (default: 1.0)
        max_retries: Maximum retries for failed uploads (default: 3)
        resume: If True, skip files that already exist (default: True)
        file_ext: File extension pattern to match (default: "*.npz")
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
    
    # Get all files matching the extension
    file_path = Path(video_dir)
    all_files = sorted(list(file_path.glob(file_ext)))
    
    # Determine file type and extension for checking
    ext_suffix = file_ext.replace("*", "")
    
    print(f"\nFound {len(all_files)} files")
    total_size_gb = sum(f.stat().st_size for f in all_files) / (1024**3)
    print(f"Total size: {total_size_gb:.2f} GB")
    
    # Get already uploaded files if resume
    uploaded_files = set()
    if resume:
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
            uploaded_files = {f for f in repo_files if f.endswith(ext_suffix)}
            print(f"Found {len(uploaded_files)} already uploaded files, will skip them")
        except Exception as e:
            print(f"Could not check existing files: {e}")
    
    # Filter out already uploaded files
    files_to_upload = [f for f in all_files if f.name not in uploaded_files]
    
    if not files_to_upload:
        print(f"All files already uploaded!")
        return
    
    print(f"Uploading {len(files_to_upload)} files with {batch_size} parallel workers...")
    print(f"Delay between batches: {delay_seconds}s, Max retries: {max_retries}")
    
    def upload_single_file(file, retry_count=0):
        """Upload a single file with retry logic"""
        try:
            api.upload_file(
                path_or_fileobj=str(file),
                path_in_repo=file.name,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            return (True, file.name, None)
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "rate limit" in error_str.lower() or "429" in error_str:
                if retry_count < max_retries:
                    # Exponential backoff for rate limits
                    wait_time = (2 ** retry_count) * delay_seconds
                    time.sleep(wait_time)
                    return upload_single_file(file, retry_count + 1)
            return (False, file.name, error_str)
    
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
    print(f"  Total files: {len(all_files)}")
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


def upload_multiple_checkpoints(
    checkpoint_path: Path,
    model_name: str,
    api,
    token: str = None,
    resume: bool = True,
    include_patterns: list = None,
    exclude_patterns: list = None,
    subdir_prefix: str = None
):
    """
    Upload multiple checkpoint subdirectories to Hugging Face model repository
    
    Each subdirectory will be uploaded as a separate checkpoint in the same repository,
    preserving the subdirectory structure.
    Also uploads log files from the same folder as the checkpoints.
    """
    # Find all subdirectories that look like checkpoints
    checkpoint_dirs = []
    
    for item in checkpoint_path.iterdir():
        if not item.is_dir():
            continue
        
        # Filter by prefix if specified
        if subdir_prefix and not item.name.startswith(subdir_prefix):
            continue
        
        # Check if directory contains checkpoint-like files
        has_checkpoint_files = False
        for pattern in include_patterns:
            if list(item.rglob(pattern)):
                has_checkpoint_files = True
                break
        
        if has_checkpoint_files:
            checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        print(f"\nNo checkpoint subdirectories found in {checkpoint_path}")
        if subdir_prefix:
            print(f"  (Looking for directories starting with '{subdir_prefix}')")
        return
    
    checkpoint_dirs = sorted(checkpoint_dirs)
    print(f"\nFound {len(checkpoint_dirs)} checkpoint subdirectories:")
    for ckpt_dir in checkpoint_dirs:
        print(f"  - {ckpt_dir.name}")
    
    # Upload each checkpoint directory
    total_uploaded = 0
    total_failed = 0
    
    for idx, ckpt_dir in enumerate(checkpoint_dirs, 1):
        print(f"\n{'='*70}")
        print(f"Uploading checkpoint {idx}/{len(checkpoint_dirs)}: {ckpt_dir.name}")
        print(f"{'='*70}")
        
        try:
            upload_checkpoint_single(
                checkpoint_path=ckpt_dir,
                repo_id=model_name,
                api=api,
                resume=resume,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                subdir_name=ckpt_dir.name
            )
            total_uploaded += 1
        except Exception as e:
            print(f"✗ Failed to upload {ckpt_dir.name}: {e}")
            total_failed += 1
    
    # Upload log files from the parent folder (same folder containing checkpoints)
    print(f"\n{'='*70}")
    print(f"Uploading log files from checkpoint folder...")
    print(f"{'='*70}")
    
    log_files = list(checkpoint_path.glob("*.log"))
    if log_files:
        print(f"Found {len(log_files)} log file(s) in {checkpoint_path}")
        
        # Get already uploaded files if resume
        uploaded_files = set()
        if resume:
            try:
                repo_files = api.list_repo_files(repo_id=model_name, repo_type="model")
                uploaded_files = set(repo_files)
            except Exception as e:
                print(f"Could not check existing files: {e}")
        
        log_uploaded = 0
        log_failed = 0
        
        for log_file in log_files:
            if log_file.name in uploaded_files and resume:
                print(f"  ⊙ {log_file.name} already exists, skipping...")
                continue
            
            try:
                file_size_mb = log_file.stat().st_size / (1024**2)
                print(f"  Uploading {log_file.name} ({file_size_mb:.2f} MB)...")
                
                api.upload_file(
                    path_or_fileobj=str(log_file),
                    path_in_repo=log_file.name,
                    repo_id=model_name,
                    repo_type="model",
                )
                log_uploaded += 1
                print(f"  ✓ Uploaded {log_file.name}")
                time.sleep(0.5)
            except Exception as e:
                log_failed += 1
                print(f"  ✗ Error uploading {log_file.name}: {str(e)[:100]}")
        
        print(f"\nLog files upload summary: {log_uploaded} uploaded, {log_failed} failed")
    else:
        print(f"No log files found in {checkpoint_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"Multiple Checkpoints Upload Summary:")
    print(f"  Total checkpoints: {len(checkpoint_dirs)}")
    print(f"  Successfully uploaded: {total_uploaded}")
    print(f"  Failed: {total_failed}")
    print(f"\nModel URL: https://huggingface.co/{model_name}")
    print(f"{'='*70}")


def upload_checkpoint_single(
    checkpoint_path: Path,
    repo_id: str,
    api,
    resume: bool = True,
    include_patterns: list = None,
    exclude_patterns: list = None,
    subdir_name: str = None
):
    """
    Upload a single checkpoint directory to Hugging Face model repository
    
    This is a helper function that handles the actual file upload logic.
    """
    # Get all files matching patterns
    pattern_matched_files = []
    for pattern in include_patterns:
        matched = list(checkpoint_path.rglob(pattern))
        pattern_matched_files.extend(matched)
    
    # Remove duplicates
    pattern_matched_files = list(set(pattern_matched_files))
    
    # Also get ALL files in the checkpoint directory as a fallback
    all_files_in_dir = [f for f in checkpoint_path.rglob("*") if f.is_file()]
    
    # Use all files if patterns matched fewer than total files
    # This ensures we don't miss files like .jinja, .py, "latest", etc.
    if len(pattern_matched_files) < len(all_files_in_dir):
        print(f"Patterns matched {len(pattern_matched_files)}/{len(all_files_in_dir)} files.")
        print(f"Using all files (will filter by exclude patterns only)...")
        all_files = all_files_in_dir
    else:
        all_files = pattern_matched_files
    
    # Remove duplicates
    all_files = list(set(all_files))
    filtered_files = []
    
    for file in all_files:
        # Skip directories (but directories are already filtered by is_file() check above)
        if not file.is_file():
            continue
            
        # Skip if matches exclude pattern
        should_exclude = False
        file_str = str(file)
        file_name = file.name
        
        for excl_pattern in exclude_patterns:
            # Try matching against full path and filename
            if fnmatch.fnmatch(file_str, excl_pattern) or \
               fnmatch.fnmatch(file_name, excl_pattern) or \
               excl_pattern in file_str:
                should_exclude = True
                break
        
        if not should_exclude:
            filtered_files.append(file)
    
    # Sort files for consistent ordering
    all_files = sorted(filtered_files)
    
    print(f"Found {len(all_files)} files in this checkpoint")
    total_size_gb = sum(f.stat().st_size for f in all_files) / (1024**3)
    print(f"Total size: {total_size_gb:.2f} GB")
    
    # Get already uploaded files if resume
    uploaded_files = set()
    if resume:
        try:
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            # If subdir_name is provided, check files under that subdirectory
            if subdir_name:
                uploaded_files = {f for f in repo_files if f.startswith(f"{subdir_name}/")}
                # Remove subdir_name prefix for comparison
                uploaded_files = {f[len(subdir_name)+1:] for f in uploaded_files}
            else:
                uploaded_files = set(repo_files)
            print(f"Found {len(uploaded_files)} already uploaded files, will skip them")
        except Exception as e:
            print(f"Could not check existing files: {e}")
    
    # Filter out already uploaded files
    files_to_upload = []
    for file in all_files:
        # Get relative path from checkpoint_path
        relative_path = file.relative_to(checkpoint_path)
        path_str = str(relative_path).replace("\\", "/")  # Normalize path separators
        
        # Add subdirectory prefix if uploading multiple checkpoints
        if subdir_name:
            repo_path = f"{subdir_name}/{path_str}"
        else:
            repo_path = path_str
        
        # Check if file already exists (compare without subdir prefix for uploaded_files)
        if path_str not in uploaded_files and repo_path not in uploaded_files:
            files_to_upload.append((file, repo_path))
    
    if not files_to_upload:
        print(f"All files in this checkpoint already uploaded!")
        return
    
    print(f"Uploading {len(files_to_upload)} files...")
    
    # Upload files
    uploaded_count = 0
    failed_files = []
    
    for file, path_in_repo in tqdm(files_to_upload, desc=f"Uploading {subdir_name or 'checkpoint'}"):
        try:
            file_size_mb = file.stat().st_size / (1024**2)
            
            # Show progress for large files
            if file_size_mb > 100:
                print(f"  Uploading {path_in_repo} ({file_size_mb:.2f} MB)...")
            
            api.upload_file(
                path_or_fileobj=str(file),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            uploaded_count += 1
            
            if file_size_mb > 100:
                print(f"  ✓ Uploaded {path_in_repo}")
            
            # Small delay to avoid rate limiting
            if file_size_mb > 50:
                time.sleep(1)
            else:
                time.sleep(0.5)
                
        except Exception as e:
            error_str = str(e)
            failed_files.append((path_in_repo, error_str))
            print(f"  ✗ Error uploading {path_in_repo}: {error_str[:100]}")
    
    # Summary for this checkpoint
    print(f"\nCheckpoint '{subdir_name or checkpoint_path.name}' summary:")
    print(f"  Uploaded: {uploaded_count}/{len(files_to_upload)}")
    if failed_files:
        print(f"  Failed: {len(failed_files)}")


def upload_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    token: str = None,
    resume: bool = True,
    include_patterns: list = None,
    exclude_patterns: list = None,
    upload_subdirs: bool = False,
    subdir_prefix: str = None
):
    """
    Upload model checkpoint files to Hugging Face model repository
    
    Args:
        checkpoint_dir: Path to directory containing checkpoint files or parent directory with multiple checkpoint subdirs
        model_name: Hugging Face model name (e.g., "username/model-name")
        token: Hugging Face authentication token
        resume: If True, skip files that already exist (default: True)
        include_patterns: List of file patterns to include (e.g., ["*.pt", "*.pth", "*.bin", "*.json"])
                         If None, includes common checkpoint files
        exclude_patterns: List of file patterns to exclude (e.g., ["*.log", "*.txt"])
        upload_subdirs: If True, upload each subdirectory as a separate checkpoint (default: False)
        subdir_prefix: If specified, only upload subdirectories matching this prefix (e.g., "checkpoint-")
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
    repo_id = model_name
    repo_type = "model"
    
    # Check/create repository
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository {repo_id} already exists")
    except Exception:
        print(f"Creating repository {repo_id}...")
        create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=False)
    
    # Default include patterns for checkpoints
    if include_patterns is None:
        include_patterns = [
            "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin",
            "*.json", "*.txt", "*.yaml", "*.yml", "*.md", "*.log",
            "*.jinja", "*.py",  # Template and Python scripts
            "tokenizer*", "vocab*", "config*", "model*", "generation*",
            # Training-related files
            "*rng_state*", "*scheduler*", "*training_args*", "*adapter*",
            "*state_dict*", "*_states.pt", "*_states.pth", "*_optim_states.pt",
            "*_optim_states.pth", "*zero_*", "*mp_rank*", "*pp_rank*",
            "*non_lora*", "*lora*", "*optimizer*", "*trainer*",
            # Common checkpoint files
            "*chat_template*", "*zero_to_fp32*", "latest*"
        ]
    
    if exclude_patterns is None:
        exclude_patterns = ["*.swp", "*.tmp", "__pycache__"]
        # Note: *.log files are now included by default to preserve training logs with checkpoints
    
    # Handle multiple checkpoints
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        return
    
    # If upload_subdirs is True, upload each subdirectory separately
    if upload_subdirs:
        upload_multiple_checkpoints(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            api=api,
            token=token,
            resume=resume,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            subdir_prefix=subdir_prefix
        )
        return
    
    # Single checkpoint upload - use the helper function
    upload_checkpoint_single(
        checkpoint_path=checkpoint_path,
        repo_id=repo_id,
        api=api,
        resume=resume,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        subdir_name=None
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Model URL: https://huggingface.co/{repo_id}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload videos/masks to Hugging Face dataset or checkpoints to model repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Package videos into archives (recommended for rate limits)
  python3 upload_videos_to_hf.py --mode package --archive_size 5.0
  
  # Upload individually with rate limiting
  python3 upload_videos_to_hf.py --mode individual --batch_size 3 --delay 2.0
  
  # Upload checkpoint to model repository
  python3 upload_videos_to_hf.py --mode checkpoint --checkpoint_dir ./checkpoints --model_name username/model-name
  
  # Upload multiple checkpoints (all subdirectories)
  python3 upload_videos_to_hf.py --mode checkpoint --checkpoint_dir ./checkpoints --model_name username/model-name --upload_subdirs
  
  # Upload multiple checkpoints with prefix filter
  python3 upload_videos_to_hf.py --mode checkpoint --checkpoint_dir ./checkpoints --model_name username/model-name --upload_subdirs --subdir_prefix checkpoint-
  
  # Using token from command line
  python3 upload_videos_to_hf.py --mode checkpoint --checkpoint_dir ./checkpoints --model_name username/model-name --token YOUR_TOKEN
  
  # Using token from environment variable (recommended)
  export HF_TOKEN=YOUR_TOKEN
  python3 upload_videos_to_hf.py --mode checkpoint --checkpoint_dir ./checkpoints --model_name username/model-name
  
Token Setup:
  Method 1 (CLI login, recommended): huggingface-cli login
  Method 2 (Env var): export HF_TOKEN=your_token_here
  Method 3 (Command line): --token your_token_here
  Get token from: https://huggingface.co/settings/tokens
        """
    )
    
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_720_mask",
        help="Path to directory containing files (for package/individual modes)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to checkpoint directory (for checkpoint mode)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PhoenixHu/sign_mllm_how2sign_720_mask",
        help="Hugging Face dataset name (for package/individual modes)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Hugging Face model name (for checkpoint mode, e.g., username/model-name)"
    )
    parser.add_argument(
        "--file_ext",
        type=str,
        default="*.npz",
        help="File extension pattern to match (default: *.npz for masks, use *.mp4 for videos)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["package", "individual", "checkpoint"],
        default="package",
        help="Upload mode: 'package' (datasets, recommended), 'individual' (datasets), or 'checkpoint' (models)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token (optional if already logged in via 'huggingface-cli login' or HF_TOKEN env var). "
             "Get token from: https://huggingface.co/settings/tokens"
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
    
    # Checkpoint mode options
    parser.add_argument(
        "--include_patterns",
        type=str,
        nargs="+",
        default=None,
        help="File patterns to include for checkpoint mode (e.g., '*.pt' '*.pth' '*.bin'). "
             "Default includes common checkpoint files."
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="+",
        default=None,
        help="File patterns to exclude for checkpoint mode (e.g., '*.log' '*.tmp')"
    )
    parser.add_argument(
        "--upload_subdirs",
        action="store_true",
        help="Upload each subdirectory as a separate checkpoint (for checkpoint mode). "
             "Useful when you have multiple checkpoint directories (e.g., checkpoint-1000, checkpoint-2000)"
    )
    parser.add_argument(
        "--subdir_prefix",
        type=str,
        default=None,
        help="Only upload subdirectories starting with this prefix (e.g., 'checkpoint-' or 'ckpt-'). "
             "Only used when --upload_subdirs is enabled"
    )
    
    args = parser.parse_args()
    
    if args.mode == "checkpoint":
        if args.checkpoint_dir is None:
            print("Error: --checkpoint_dir is required for checkpoint mode")
            sys.exit(1)
        if args.model_name is None:
            print("Error: --model_name is required for checkpoint mode")
            sys.exit(1)
        
        upload_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            token=args.token,
            resume=not args.no_resume,
            include_patterns=args.include_patterns,
            exclude_patterns=args.exclude_patterns,
            upload_subdirs=args.upload_subdirs,
            subdir_prefix=args.subdir_prefix
        )
    elif args.mode == "package":
        upload_videos_packaged(
            video_dir=args.video_dir,
            dataset_name=args.dataset_name,
            token=args.token,
            archive_size_gb=args.archive_size,
            resume=not args.no_resume,
            file_ext=args.file_ext
        )
    else:  # individual mode
        upload_videos(
            video_dir=args.video_dir,
            dataset_name=args.dataset_name,
            token=args.token,
            batch_size=args.batch_size,
            delay_seconds=args.delay,
            max_retries=args.max_retries,
            resume=not args.no_resume,
            file_ext=args.file_ext
        )
