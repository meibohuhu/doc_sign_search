#!/usr/bin/env python3
"""
Download and extract video archives from Hugging Face dataset repository
"""

import os
import sys
import tarfile
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, login, snapshot_download
from tqdm import tqdm

def download_and_extract(
    dataset_name: str,
    output_dir: str,
    token: str = None,
    extract_only: bool = False,
    archive_pattern: str = "videos_*.tar.gz",
    remove_archives: bool = False
):
    """
    Download all video archives from HF dataset and extract them
    
    Args:
        dataset_name: Hugging Face dataset name (e.g., "PhoenixHu/sign_mllm_how_224")
        output_dir: Directory to save extracted videos
        token: Hugging Face authentication token (optional)
        extract_only: If True, assume archives already exist in output_dir and only extract
        archive_pattern: Pattern to match archive files (default: "videos_*.tar.gz")
    """
    # Login if needed
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if token:
        login(token=token)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if extract_only:
        # Only extract existing archives
        print(f"Extracting archives from {output_path}...")
        archive_files = sorted(list(output_path.glob(archive_pattern)))
        
        if not archive_files:
            print(f"No archives found matching pattern '{archive_pattern}' in {output_path}")
            return
        
        print(f"Found {len(archive_files)} archive(s)")
        
        for archive_file in tqdm(archive_files, desc="Extracting archives"):
            print(f"\nExtracting {archive_file.name}...")
            with tarfile.open(archive_file, "r:gz") as tar:
                tar.extractall(path=output_path)
                members = tar.getnames()
                print(f"  Extracted {len(members)} files")
        
        print(f"\n{'='*70}")
        print(f"Extraction complete!")
        print(f"Videos are in: {output_path}")
        print(f"{'='*70}")
        
    else:
        # Download and extract
        print(f"Downloading archives from {dataset_name}...")
        print(f"Output directory: {output_path}")
        
        # Download all tar.gz files
        try:
            # Use snapshot_download to download all files
            print("Downloading all archives...")
            repo_files = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=str(output_path),
                allow_patterns=["*.tar.gz"],
                token=token
            )
        except Exception as e:
            print(f"Error downloading: {e}")
            print("Trying alternative method...")
            
            # Alternative: list and download individually
            from huggingface_hub import HfApi
            api = HfApi()
            try:
                repo_files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
                archive_files = [f for f in repo_files if f.endswith('.tar.gz') and 'videos_' in f]
                
                print(f"Found {len(archive_files)} archive(s) to download")
                
                for archive_file in tqdm(archive_files, desc="Downloading archives"):
                    print(f"\nDownloading {archive_file}...")
                    hf_hub_download(
                        repo_id=dataset_name,
                        filename=archive_file,
                        repo_type="dataset",
                        local_dir=str(output_path),
                        token=token
                    )
            except Exception as e2:
                print(f"Error: {e2}")
                raise
        
        # Extract all downloaded archives
        print(f"\nExtracting archives...")
        archive_files = sorted(list(output_path.glob(archive_pattern)))
        
        if not archive_files:
            print(f"No archives found matching pattern '{archive_pattern}'")
            print(f"Available files: {list(output_path.glob('*'))}")
            return
        
        print(f"Found {len(archive_files)} archive(s)")
        
        total_files = 0
        for archive_file in tqdm(archive_files, desc="Extracting archives"):
            print(f"\nExtracting {archive_file.name}...")
            with tarfile.open(archive_file, "r:gz") as tar:
                members = tar.getnames()
                tar.extractall(path=output_path)
                print(f"  Extracted {len(members)} files")
                total_files += len(members)
        
        # Optionally remove archives after extraction to save space
        if remove_archives:
            print(f"\nRemoving archive files to save space...")
            for archive_file in archive_files:
                archive_file.unlink()
                print(f"  Removed {archive_file.name}")
            print("Archive files removed.")
        
        print(f"\n{'='*70}")
        print(f"Download and extraction complete!")
        print(f"Total videos extracted: {total_files}")
        print(f"Videos are in: {output_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract video archives from Hugging Face dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and extract all archives
  python3 download_and_extract_videos_from_hf.py \\
      --dataset_name PhoenixHu/sign_mllm_how_224 \\
      --output_dir ./videos
  
  # Extract already downloaded archives
  python3 download_and_extract_videos_from_hf.py \\
      --dataset_name PhoenixHu/sign_mllm_how_224 \\
      --output_dir ./videos \\
      --extract_only
        """
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Hugging Face dataset name (e.g., 'PhoenixHu/sign_mllm_how_224')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted videos"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token (optional if already logged in)"
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only extract archives (assume archives already downloaded)"
    )
    parser.add_argument(
        "--archive_pattern",
        type=str,
        default="videos_*.tar.gz",
        help="Pattern to match archive files (default: 'videos_*.tar.gz')"
    )
    parser.add_argument(
        "--remove_archives",
        action="store_true",
        help="Remove archive files after extraction to save space"
    )
    
    args = parser.parse_args()
    
    download_and_extract(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        token=args.token,
        extract_only=args.extract_only,
        archive_pattern=args.archive_pattern,
        remove_archives=args.remove_archives
    )

