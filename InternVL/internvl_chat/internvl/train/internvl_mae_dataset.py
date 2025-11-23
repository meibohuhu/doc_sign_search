"""
Dataset for MAE pre-training of InternVL vision encoder
"""

import json
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Import InternVL's video loading function
# This uses the same video loading method as InternVL's training code
# Handle both relative and absolute imports
try:
    from .internvl_chat_finetune_local import _load_video_locally
except ImportError:
    # Fallback to absolute import (when run as script)
    # Add the train directory to sys.path
    import sys
    train_dir = os.path.dirname(os.path.abspath(__file__))
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    from internvl_chat_finetune_local import _load_video_locally


class MAEVideoDataset(Dataset):
    """
    Dataset for MAE pre-training on videos
    Only loads video frames (no text/conversation)
    
    Expected data format (JSON file):
    [
        {
            "video": "path/to/video.mp4",  # required: video file path (relative or absolute)
            # OR
            "video_path": "path/to/video.mp4",  # alternative key name
            # Other fields like "id", "conversations" are ignored for MAE
        },
        ...
    ]
    
    OR simple list of video paths:
    [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str | List[str],
        image_size: int = 224,
        min_num_frames: int = 8,
        max_num_frames: int = 32,
        sampling_method: str = 'rand',
        video_base_path: Optional[str] = None,
        transform: Optional[T.Compose] = None,
    ):
        """
        Args:
            data_path: path to JSON file containing video paths
            image_size: target image size (will be resized to square)
            min_num_frames: minimum number of frames to sample
            max_num_frames: maximum number of frames to sample
            sampling_method: sampling strategy ('rand', 'fpsX.X', 'random_start_every2')
            video_base_path: base path for video files (used when JSON contains relative paths)
            transform: optional torchvision transforms
        """
        self.image_size = image_size
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.sampling_method = sampling_method
        self.video_base_path = video_base_path
        
        # Default transform: resize and normalize
        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load data
        if isinstance(data_path, str):
            data_path = [data_path]
        
        self.data_path = data_path  # Store for path resolution
        self.data = []
        for path in data_path:
            with open(path, 'r', encoding='utf-8') as f:
                # Try to load as JSON first (could be config or array)
                try:
                    content = json.load(f)
                    # Check if it's a config file (has 'root' and 'annotation' keys)
                    if isinstance(content, dict):
                        # Check if any value has 'annotation' key (config format)
                        is_config = False
                        jsonl_path = None
                        config_root = None
                        for key, value in content.items():
                            if isinstance(value, dict) and 'annotation' in value:
                                is_config = True
                                jsonl_path = value['annotation']
                                # Use root from config if video_base_path not provided
                                if 'root' in value:
                                    config_root = value['root']
                                break
                        
                        if is_config and jsonl_path:
                            # Load from JSONL file specified in config
                            print(f"Loading data from config file {path}")
                            print(f"  JSONL path: {jsonl_path}")
                            if config_root:
                                if not self.video_base_path:
                                    self.video_base_path = config_root
                                    print(f"  Using video root from config: {config_root}")
                                else:
                                    print(f"  Using provided video_base_path: {self.video_base_path}")
                            
                            with open(jsonl_path, 'r', encoding='utf-8') as jsonl_f:
                                for line in jsonl_f:
                                    line = line.strip()
                                    if line:
                                        self.data.append(json.loads(line))
                        else:
                            # It's a JSON object/dict but not a config file
                            # Check if it looks like a data sample (has 'video' key)
                            if 'video' in content or 'video_path' in content:
                                # Single sample
                                self.data.append(content)
                            else:
                                # Try to treat as array of samples
                                raise ValueError(
                                    f"File {path} is a JSON dict but doesn't match config format "
                                    f"(missing 'annotation' key) or data format (missing 'video' key). "
                                    f"Keys found: {list(content.keys())[:5]}"
                                )
                    elif isinstance(content, list):
                        # JSON array of samples
                        self.data.extend(content)
                    else:
                        raise ValueError(f"Unexpected JSON format in {path}: {type(content)}")
                except json.JSONDecodeError:
                    # Not valid JSON, try as JSONL (one JSON object per line)
                    f.seek(0)  # Reset file pointer
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.data.append(json.loads(line))
                            except json.JSONDecodeError:
                                print(f"Warning: Skipping invalid JSON line in {path}")
                                continue
        
        print(f"Loaded {len(self.data)} samples for MAE pre-training")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            pixel_values_videos: [T, C, H, W] - video frames as tensor
            video_grid_thw: [3] - (T, H, W) for the video
        """
        # Skip logic: if current sample fails, try next sample (max 100 skips)
        max_skip = 100
        for skip_count in range(max_skip):
            actual_idx = (idx + skip_count) % len(self.data)
            sample = self.data[actual_idx]
            
            try:
                # Extract video path
                if isinstance(sample, dict):
                    if 'video' in sample:
                        video_file = sample['video']
                    elif 'video_path' in sample:
                        video_file = sample['video_path']
                    elif 'image' in sample:  # fallback for image data
                        video_file = sample['image']
                    else:
                        raise ValueError(f"Sample {actual_idx} does not contain 'video' or 'video_path' key")
                else:
                    video_file = sample
                
                # Resolve video path
                if not os.path.isabs(video_file):
                    # If video_base_path is provided, use it as base path
                    if self.video_base_path:
                        video_file = os.path.join(self.video_base_path, video_file)
                    else:
                        # Otherwise, try relative to data file directory
                        data_file_path = self.data_path[0] if isinstance(self.data_path, list) else self.data_path
                        video_file = os.path.join(os.path.dirname(data_file_path), video_file)
                
                if not os.path.exists(video_file) and not video_file.startswith("http"):
                    raise FileNotFoundError(f"Video file not found: {video_file}")
                
                # Load video frames using InternVL's _load_video_locally (same as training code)
                image_list = _load_video_locally(
                    video_path=video_file,
                    max_num_frames=self.max_num_frames,
                    min_num_frames=self.min_num_frames,
                    sample=self.sampling_method,
                    clip=None,  # MAE doesn't use clip parameter
                )
                
                if len(image_list) == 0:
                    raise ValueError(f"No frames loaded from {video_file}")
                
                # Apply transforms to each frame
                # image_list is already List[Image.Image] from _load_video_locally
                transformed_frames = []
                for image in image_list:
                    transformed_frame = self.transform(image)  # [C, H, W]
                    transformed_frames.append(transformed_frame)
                
                # Stack frames: [T, C, H, W]
                pixel_values_videos = torch.stack(transformed_frames, dim=0)
                
                # Get video dimensions
                T_dim = pixel_values_videos.shape[0]
                H_dim = pixel_values_videos.shape[2]
                W_dim = pixel_values_videos.shape[3]
                video_grid_thw = torch.tensor([T_dim, H_dim, W_dim], dtype=torch.long)
                
                return {
                    'pixel_values_videos': pixel_values_videos,  # [T, C, H, W]
                    'video_grid_thw': video_grid_thw,  # [3]
                }
                
            except Exception as e:
                # Log the error and skip to next sample
                video_file_info = 'unknown'
                if isinstance(sample, dict):
                    video_file_info = sample.get('video', sample.get('video_path', 'unknown'))
                else:
                    video_file_info = str(sample)[:100]
                    
                error_msg = str(e)[:200]
                print(f"⚠️  Warning: Skipping sample {actual_idx} (video: {video_file_info}). Error: {error_msg}")
                
                # Continue to next sample
                continue
        
        # If we've skipped max_skip samples without success, raise an error
        raise RuntimeError(f"Failed to load any valid sample after skipping {max_skip} samples starting from index {idx}")


class DataCollatorForMAEDataset(object):
    """Collate function for MAE dataset - concatenates videos from batch"""
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function that concatenates all videos in the batch
        
        Args:
            features: list of dicts with 'pixel_values_videos' and 'video_grid_thw'
        
        Returns:
            pixel_values_videos: [total_patches, patch_pixel_dim] - flattened and concatenated patches
            video_grid_thw: [batch_size, 3] - (T, H, W) for each video (in patch units)
        """
        pixel_values_list = []
        video_grid_thw_list = []
        patch_size = 14  # InternVL's patch size
        
        for feat in features:
            # feat['pixel_values_videos'] is [T, C, H, W]
            pv = feat['pixel_values_videos']  # [T, C, H, W]
            T, C, H, W = pv.shape
            
            # Extract patches: [T, C, H, W] -> [T * num_patches_per_frame, C * patch_size * patch_size]
            # Calculate number of patches per frame
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            num_patches_per_frame = num_patches_h * num_patches_w
            
            # Extract patches for each frame
            patches = []
            for t in range(T):
                frame = pv[t]  # [C, H, W]
                # Use unfold to extract patches
                frame_patches = frame.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
                # frame_patches: [C, num_patches_h, num_patches_w, patch_size, patch_size]
                frame_patches = frame_patches.permute(1, 2, 0, 3, 4).contiguous()
                # frame_patches: [num_patches_h, num_patches_w, C, patch_size, patch_size]
                frame_patches = frame_patches.reshape(num_patches_per_frame, C * patch_size * patch_size)
                # frame_patches: [num_patches_per_frame, C * patch_size * patch_size]
                patches.append(frame_patches)
            
            # Concatenate all frames: [T * num_patches_per_frame, C * patch_size * patch_size]
            video_patches = torch.cat(patches, dim=0)
            pixel_values_list.append(video_patches)
            
            # video_grid_thw in patch units: (T, H_patches, W_patches)
            # T is number of frames, H_patches and W_patches are spatial patch dimensions
            # Total patches per video = T * H_patches * W_patches
            H_patches = num_patches_h
            W_patches = num_patches_w
            video_grid_thw_list.append(torch.tensor([T, H_patches, W_patches], dtype=torch.long))
        
        # Stack video_grid_thw
        video_grid_thw = torch.stack(video_grid_thw_list, dim=0)  # [batch_size, 3]
        
        # Concatenate all patches from all videos
        pixel_values_videos = torch.cat(pixel_values_list, dim=0)  # [total_patches, patch_pixel_dim]
        
        return {
            'pixel_values_videos': pixel_values_videos,
            'video_grid_thw': video_grid_thw,
        }


def make_mae_data_module(
    data_path: str | List[str],
    image_size: int = 224,
    min_num_frames: int = 8,
    max_num_frames: int = 32,
    sampling_method: str = 'rand',
    video_base_path: Optional[str] = None,
    transform: Optional[T.Compose] = None,
):
    """
    Create MAE data module
    
    Args:
        data_path: path to JSON file containing video paths
        image_size: target image size
        min_num_frames: minimum number of frames
        max_num_frames: maximum number of frames
        sampling_method: sampling strategy
        video_base_path: base path for video files
        transform: optional transforms
    """
    dataset = MAEVideoDataset(
        data_path=data_path,
        image_size=image_size,
        min_num_frames=min_num_frames,
        max_num_frames=max_num_frames,
        sampling_method=sampling_method,
        video_base_path=video_base_path,
        transform=transform,
    )
    
    data_collator = DataCollatorForMAEDataset()
    
    return {
        'dataset': dataset,
        'data_collator': data_collator,
    }

