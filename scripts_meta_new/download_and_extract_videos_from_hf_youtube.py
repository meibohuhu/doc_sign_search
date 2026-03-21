#!/usr/bin/env python3
"""
Download and extract video archives from Hugging Face dataset repository
Also supports downloading model checkpoints from Hugging Face model repository

# 下载并解压视频（完整流程）
python3 /local1/mhu/sign_language_llm/download_and_extract_videos_from_hf.py \
    --mode dataset \
    --dataset_name PhoenixHu/how2sign_test_videos_224x224 \
    --output_dir /local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224_downloaded

# 如果只想解压已下载的归档文件
python3 /local1/mhu/sign_language_llm/download_and_extract_videos_from_hf.py \
    --mode dataset \
    --dataset_name PhoenixHu/how2sign_test_videos_224x224 \
    --output_dir /local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224_downloaded \
    --extract_only

# 解压后自动删除归档文件以节省空间
python3 /local1/mhu/sign_language_llm/download_and_extract_videos_from_hf.py \
    --mode dataset \
    --dataset_name PhoenixHu/how2sign_test_videos_224x224 \
    --output_dir /local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224_downloaded \
    --remove_archives

python3 /local1/mhu/sign_language_llm/download_and_extract_videos_from_hf.py \
    --mode checkpoint \
    --model_name PhoenixHu/finetune_internvl2_5_openasl_1B_1216_121640 \
    --output_dir /local1/mhu/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_openasl_1B_1216_121640 \
    --checkpoint_subdir checkpoint-6374

    python3 download_and_extract_videos_from_hf.py \
    --mode checkpoint \
    --model_name PhoenixHu/finetune_internvl2_5_how2sign_16fps_1203 \
    --output_dir /shared/rc/llm-gen-agent/mhu/internvl/20260202/finetune_internvl2_5_how2sign_16fps_1203 \
    --checkpoint_subdir checkpoint-2040
        
    python3 download_and_extract_videos_from_hf.py \
    --mode checkpoint \
    --model_name PhoenixHu/finetune_internvl2_5_how2sign_16fps_1203 \
    --output_dir /shared/rc/llm-gen-agent/mhu/internvl/20260202/finetune_internvl2_5_how2sign_16fps_1203 \
    --checkpoint_subdir checkpoint-2555

python3 download_and_extract_videos_from_hf.py \
    --mode checkpoint \
    --model_name PhoenixHu/finetune_internvl2_5_how2sign_1b_16fps_1205 \
    --output_dir /shared/rc/llm-gen-agent/mhu/internvl/20260202/finetune_internvl2_5_how2sign_1b_16fps_1205 \
    --checkpoint_subdir checkpoint-3058



"""

import os
import sys
import tarfile
import argparse
import subprocess
import re
from pathlib import Path
from huggingface_hub import hf_hub_download, login, snapshot_download, HfApi
from tqdm import tqdm
import fnmatch

def merge_and_extract_split_archives(output_path: Path, archive_pattern: str, remove_archives: bool = False):
    """
    Detect split archive files (e.g., *.tar.gz_aa, *.tar.gz_ab, ...),
    merge them with cat, then extract the combined archive.

    Returns:
        total_files: number of files extracted, or -1 if no split archives found
    """
    # Find split files matching pattern
    split_files = sorted(output_path.glob(archive_pattern))
    if not split_files:
        return -1

    # Group by base name (e.g., "foo.tar.gz_aa" -> "foo.tar.gz")
    groups = {}
    for f in split_files:
        # Match pattern like name.tar.gz_xx
        match = re.match(r'^(.+\.tar\.gz)_([a-z]+)$', f.name)
        if match:
            base = match.group(1)
            if base not in groups:
                groups[base] = []
            groups[base].append(f)

    if not groups:
        return -1

    total_files = 0
    for base_name, parts in groups.items():
        parts = sorted(parts)  # ensure alphabetical order (_aa, _ab, ...)
        merged_path = output_path / base_name

        print(f"\nMerging {len(parts)} split files into {base_name}...")
        # Use cat to merge
        cmd = f"cat {' '.join(str(p) for p in parts)} > {merged_path}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"  Merged file size: {merged_path.stat().st_size / (1024**3):.2f} GB")

        # Extract
        print(f"Extracting {base_name}...")
        with tarfile.open(merged_path, "r:gz") as tar:
            tar.extractall(path=output_path)
            members = tar.getnames()
            print(f"  Extracted {len(members)} files")
            total_files += len(members)

        # Cleanup
        if remove_archives:
            merged_path.unlink()
            for p in parts:
                p.unlink()
            print(f"  Removed merged archive and {len(parts)} split files")
        else:
            # Always remove the merged file (it's a duplicate), keep splits
            merged_path.unlink()
            print(f"  Removed merged archive (split files kept)")

    return total_files


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

        # Try split archives first (e.g., *.tar.gz_aa, *.tar.gz_ab)
        split_pattern = archive_pattern + "_*" if not archive_pattern.endswith("_*") else archive_pattern
        total = merge_and_extract_split_archives(output_path, split_pattern, remove_archives)
        if total >= 0:
            print(f"\n{'='*70}")
            print(f"Extraction complete! (split archives)")
            print(f"Total files extracted: {total}")
            print(f"Videos are in: {output_path}")
            print(f"{'='*70}")
            return

        # Fall back to regular archives
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
        
        # Download all tar.gz files (including split archives like .tar.gz_aa)
        try:
            print("Downloading all archives...")
            repo_files = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=str(output_path),
                allow_patterns=["*.tar.gz", "*.tar.gz_*"],
                token=token
            )
        except Exception as e:
            print(f"Error downloading: {e}")
            print("Trying alternative method...")

            api = HfApi()
            try:
                repo_files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
                archive_files = [f for f in repo_files if '.tar.gz' in f]

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

        # Try split archives first
        split_pattern = archive_pattern + "_*" if not archive_pattern.endswith("_*") else archive_pattern
        total_files = merge_and_extract_split_archives(output_path, split_pattern, remove_archives)

        if total_files < 0:
            # Fall back to regular archives
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


def download_checkpoint(
    model_name: str,
    output_dir: str,
    token: str = None,
    checkpoint_subdir: str = None,
    include_patterns: list = None,
    exclude_patterns: list = None,
    download_all_checkpoints: bool = False
):
    """
    Download model checkpoint from Hugging Face model repository
    
    Args:
        model_name: Hugging Face model name (e.g., "username/model-name")
        output_dir: Directory to save checkpoint files
        token: Hugging Face authentication token (optional)
        checkpoint_subdir: Specific checkpoint subdirectory to download (e.g., "checkpoint-1000")
        include_patterns: List of file patterns to include (default: common checkpoint files)
        exclude_patterns: List of file patterns to exclude (default: common temp files)
        download_all_checkpoints: If True, download all checkpoint subdirectories
    """
    # Login if needed
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if token:
        login(token=token)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default include patterns for checkpoints
    if include_patterns is None:
        include_patterns = [
            "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin",
            "*.json", "*.txt", "*.yaml", "*.yml", "*.md",
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
        # When downloading all checkpoints, also include log files
        if download_all_checkpoints:
            include_patterns.extend(["*.log", "*log*", "*.log.*"])
    
    if exclude_patterns is None:
        # When downloading all checkpoints, don't exclude log files
        if download_all_checkpoints:
            exclude_patterns = ["*.swp", "*.tmp", "__pycache__"]
        else:
            exclude_patterns = ["*.log", "*.swp", "*.tmp", "__pycache__"]
    
    api = HfApi()
    repo_type = "model"
    
    print(f"Downloading checkpoint from {model_name}...")
    print(f"Output directory: {output_path}")
    
    # List all files in the repository
    try:
        repo_files = api.list_repo_files(repo_id=model_name, repo_type=repo_type)
        print(f"Found {len(repo_files)} files in repository")
    except Exception as e:
        print(f"Error listing repository files: {e}")
        raise
    
    # Filter files based on patterns and subdirectory
    files_to_download = []
    
    if download_all_checkpoints:
        # Download all checkpoint subdirectories
        checkpoint_dirs = set()
        root_level_files = []
        
        for file in repo_files:
            # Extract subdirectory name (e.g., "checkpoint-1000/model.bin" -> "checkpoint-1000")
            if "/" in file:
                subdir = file.split("/")[0]
                # Check if it looks like a checkpoint directory
                if any(keyword in subdir.lower() for keyword in ["checkpoint", "ckpt", "step", "epoch", "global"]):
                    checkpoint_dirs.add(subdir)
            else:
                # Root level files (not in any subdirectory)
                root_level_files.append(file)
        
        checkpoint_dirs = sorted(checkpoint_dirs)
        print(f"\nFound {len(checkpoint_dirs)} checkpoint subdirectories:")
        for ckpt_dir in checkpoint_dirs:
            print(f"  - {ckpt_dir}")
        
        # Download root level files first (if any)
        if root_level_files:
            print(f"\n{'='*70}")
            print(f"Downloading root level files ({len(root_level_files)} files)...")
            print(f"{'='*70}")
            
            # Filter root files by patterns
            files_to_download = []
            for file in root_level_files:
                file_name = Path(file).name
                
                # Check include patterns
                matches_include = False
                for pattern in include_patterns:
                    if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(file_name, pattern):
                        matches_include = True
                        break
                
                if not matches_include:
                    continue
                
                # Check exclude patterns
                matches_exclude = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(file, pattern) or pattern in file:
                        matches_exclude = True
                        break
                
                if matches_exclude:
                    continue
                
                files_to_download.append(file)
            
            if files_to_download:
                print(f"  Found {len(files_to_download)} root level files to download")
                downloaded_count = 0
                for file in tqdm(files_to_download, desc="  Downloading root files"):
                    try:
                        local_file = hf_hub_download(
                            repo_id=model_name,
                            filename=file,
                            repo_type="model",
                            local_dir=str(output_path),
                            token=token
                        )
                        downloaded_count += 1
                    except Exception as e:
                        print(f"    ✗ Error downloading {file}: {e}")
                
                print(f"  ✓ Downloaded {downloaded_count}/{len(files_to_download)} root level files")
            else:
                print(f"  No root level files match the include patterns")
        
        # Download each checkpoint subdirectory
        for idx, ckpt_dir in enumerate(checkpoint_dirs, 1):
            print(f"\n{'='*70}")
            print(f"Downloading checkpoint {idx}/{len(checkpoint_dirs)}: {ckpt_dir}")
            print(f"{'='*70}")
            
            download_checkpoint_subdir(
                repo_files=repo_files,
                model_name=model_name,
                output_path=output_path,
                api=api,
                checkpoint_subdir=ckpt_dir,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                token=token
            )
        
        print(f"\n{'='*70}")
        print(f"All checkpoints downloaded successfully!")
        print(f"Checkpoints are in: {output_path}")
        print(f"{'='*70}")
        
    elif checkpoint_subdir:
        # Download specific checkpoint subdirectory
        print(f"\nDownloading checkpoint: {checkpoint_subdir}")
        download_checkpoint_subdir(
            repo_files=repo_files,
            model_name=model_name,
            output_path=output_path,
            api=api,
            checkpoint_subdir=checkpoint_subdir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            token=token
        )
        print(f"\n{'='*70}")
        print(f"Checkpoint downloaded successfully!")
        print(f"Checkpoint is in: {output_path}")
        print(f"{'='*70}")
        
    else:
        # Download all files matching patterns (root level)
        print(f"\nDownloading all checkpoint files...")
        
        for file in repo_files:
            # Check if file matches include patterns
            matches_include = False
            for pattern in include_patterns:
                if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(Path(file).name, pattern):
                    matches_include = True
                    break
            
            if not matches_include:
                continue
            
            # Check if file matches exclude patterns
            matches_exclude = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file, pattern) or pattern in file:
                    matches_exclude = True
                    break
            
            if matches_exclude:
                continue
            
            # Skip files in subdirectories
            if "/" in file:
                continue
            
            files_to_download.append(file)
        
        print(f"Found {len(files_to_download)} files to download")
        
        # Download files
        downloaded_count = 0
        for file in tqdm(files_to_download, desc="Downloading files"):
            try:
                local_file = hf_hub_download(
                    repo_id=model_name,
                    filename=file,
                    repo_type=repo_type,
                    local_dir=str(output_path),
                    token=token
                )
                downloaded_count += 1
            except Exception as e:
                print(f"  ✗ Error downloading {file}: {e}")
        
        print(f"\n{'='*70}")
        print(f"Download complete!")
        print(f"Downloaded {downloaded_count}/{len(files_to_download)} files")
        print(f"Files are in: {output_path}")
        print(f"{'='*70}")


def download_checkpoint_subdir(
    repo_files: list,
    model_name: str,
    output_path: Path,
    api: HfApi,
    checkpoint_subdir: str,
    include_patterns: list,
    exclude_patterns: list,
    token: str = None
):
    """Download files from a specific checkpoint subdirectory"""
    # Filter files in this subdirectory
    subdir_files = [f for f in repo_files if f.startswith(f"{checkpoint_subdir}/")]
    
    if not subdir_files:
        print(f"  No files found in {checkpoint_subdir}")
        return
    
    # Filter by patterns
    files_to_download = []
    for file in subdir_files:
        file_name = Path(file).name
        relative_path = file[len(checkpoint_subdir)+1:]  # Remove subdir prefix
        
        # Check include patterns
        matches_include = False
        for pattern in include_patterns:
            if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(file_name, pattern):
                matches_include = True
                break
        
        if not matches_include:
            continue
        
        # Check exclude patterns
        matches_exclude = False
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file, pattern) or pattern in file:
                matches_exclude = True
                break
        
        if matches_exclude:
            continue
        
        files_to_download.append(file)
    
    print(f"  Found {len(files_to_download)} files to download")
    
    # Create subdirectory in output
    checkpoint_output = output_path / checkpoint_subdir
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    
    # Download files
    downloaded_count = 0
    for file in tqdm(files_to_download, desc=f"  Downloading {checkpoint_subdir}", leave=False):
        try:
            local_file = hf_hub_download(
                repo_id=model_name,
                filename=file,
                repo_type="model",
                local_dir=str(output_path),
                token=token
            )
            downloaded_count += 1
        except Exception as e:
            print(f"    ✗ Error downloading {file}: {e}")
    
    print(f"  ✓ Downloaded {downloaded_count}/{len(files_to_download)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract video archives from Hugging Face dataset or download model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and extract video archives from dataset
  python3 download_and_extract_videos_from_hf.py \\
      --mode dataset \\
      --dataset_name PhoenixHu/sign_mllm_how_224 \\
      --output_dir ./videos
  
  # Extract already downloaded archives
  python3 download_and_extract_videos_from_hf.py \\
      --mode dataset \\
      --dataset_name PhoenixHu/sign_mllm_how_224 \\
      --output_dir ./videos \\
      --extract_only
  
  # Download specific checkpoint
  python3 download_and_extract_videos_from_hf.py \\
      --mode checkpoint \\
      --model_name username/model-name \\
      --output_dir ./checkpoints \\
      --checkpoint_subdir checkpoint-1000
  
  # Download all checkpoints
  python3 download_and_extract_videos_from_hf.py \\
      --mode checkpoint \\
      --model_name username/model-name \\
      --output_dir ./checkpoints \\
      --download_all_checkpoints
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dataset", "checkpoint"],
        default="dataset",
        help="Download mode: 'dataset' (videos/archives) or 'checkpoint' (model checkpoints)"
    )
    
    # Dataset mode arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g., 'PhoenixHu/sign_mllm_how_224'). Required for dataset mode"
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only extract archives (assume archives already downloaded). Dataset mode only"
    )
    parser.add_argument(
        "--archive_pattern",
        type=str,
        default="videos_*.tar.gz",
        help="Pattern to match archive files (default: 'videos_*.tar.gz'). Dataset mode only"
    )
    parser.add_argument(
        "--remove_archives",
        action="store_true",
        help="Remove archive files after extraction to save space. Dataset mode only"
    )
    
    # Checkpoint mode arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Hugging Face model name (e.g., 'username/model-name'). Required for checkpoint mode"
    )
    parser.add_argument(
        "--checkpoint_subdir",
        type=str,
        default=None,
        help="Specific checkpoint subdirectory to download (e.g., 'checkpoint-1000'). Checkpoint mode only"
    )
    parser.add_argument(
        "--download_all_checkpoints",
        action="store_true",
        help="Download all checkpoint subdirectories. Checkpoint mode only"
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        nargs="+",
        default=None,
        help="File patterns to include for checkpoint mode. Default includes common checkpoint files"
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="+",
        default=None,
        help="File patterns to exclude for checkpoint mode (e.g., '*.log' '*.tmp')"
    )
    
    # Common arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token (optional if already logged in)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "checkpoint":
        if args.model_name is None:
            print("Error: --model_name is required for checkpoint mode")
            sys.exit(1)
        
        download_checkpoint(
            model_name=args.model_name,
            output_dir=args.output_dir,
            token=args.token,
            checkpoint_subdir=args.checkpoint_subdir,
            include_patterns=args.include_patterns,
            exclude_patterns=args.exclude_patterns,
            download_all_checkpoints=args.download_all_checkpoints
        )
    else:  # dataset mode
        if args.dataset_name is None:
            print("Error: --dataset_name is required for dataset mode")
            sys.exit(1)
        
        download_and_extract(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            token=args.token,
            extract_only=args.extract_only,
            archive_pattern=args.archive_pattern,
            remove_archives=args.remove_archives
        )

