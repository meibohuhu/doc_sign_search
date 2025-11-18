"""
Dataset for MAE pre-training of Qwen2-VL vision encoder
"""

import json
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path
import transformers
from transformers import AutoProcessor

from src.dataset.data_utils import get_video_info


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
        processor: transformers.ProcessorMixin,
        video_resized_width: int = 224,
        video_resized_height: int = 224,
        video_min_pixels: int = 224 * 224,
        video_max_pixels: int = 224 * 224,
        fps: Optional[int] = None,
        nframes: Optional[int] = None,
        video_base_path: Optional[str] = None,
    ):
        """
        Args:
            data_path: path to JSON file containing video paths
            processor: Qwen2-VL processor
            video_resized_width: target video width
            video_resized_height: target video height
            video_min_pixels: minimum pixels per frame
            video_max_pixels: maximum pixels per frame
            fps: frames per second (optional)
            nframes: number of frames (optional, deprecated, use fps instead)
            video_base_path: base path for video files (used when JSON contains relative paths)
        """
        self.processor = processor
        self.video_resized_width = video_resized_width
        self.video_resized_height = video_resized_height
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.fps = fps
        self.nframes = nframes
        self.video_base_path = video_base_path
        
        # Load data
        if isinstance(data_path, str):
            data_path = [data_path]
        
        self.data_path = data_path  # Store for path resolution
        self.data = []
        for path in data_path:
            with open(path, 'r', encoding='utf-8') as f:
                self.data.extend(json.load(f))
        
        print(f"Loaded {len(self.data)} samples for MAE pre-training")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            pixel_values_videos: [N, patch_pixel_dim] - flattened video patches
            video_grid_thw: [num_videos, 3] - (T, H, W) for each video
        """
        sample = self.data[idx]
        
        # Extract video path
        if isinstance(sample, dict):
            if 'video' in sample:
                video_file = sample['video']
            elif 'video_path' in sample:
                video_file = sample['video_path']
            elif 'image' in sample:  # fallback for image data
                video_file = sample['image']
            else:
                raise ValueError(f"Sample {idx} does not contain 'video' or 'video_path' key")
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
        
        # Load video using processor
        video_input, video_kwargs = get_video_info(
            video_file, 
            self.video_min_pixels, 
            self.video_max_pixels, 
            self.video_resized_width, 
            self.video_resized_height, 
            self.fps, 
            self.nframes
        )
        
        # Process video to get patches
        # Qwen2-VL processor may require text input, so we pass empty text for MAE
        # MAE is self-supervised and only needs video data (no text labels)
        try:
            # Try without text first (some processors support this)
            inputs = self.processor(
                videos=[video_input],
                padding=False,
                do_resize=False,
                return_tensors='pt',
                **video_kwargs
            )
        except (TypeError, ValueError) as e:
            # If processor requires text, pass empty string
            inputs = self.processor(
                text=[""],  # Empty text for MAE (no text labels needed)
                videos=[video_input],
                padding=False,
                do_resize=False,
                return_tensors='pt',
                **video_kwargs
            )
        
        # Extract pixel_values_videos and video_grid_thw
        pixel_values_videos = inputs['pixel_values_videos']  # [N, patch_pixel_dim]
        video_grid_thw = inputs['video_grid_thw']  # [num_videos, 3] where num_videos=1
        
        return {
            'pixel_values_videos': pixel_values_videos.squeeze(0),  # [N, patch_pixel_dim]
            'video_grid_thw': video_grid_thw.squeeze(0),  # [1, 3] -> [3]
        }


class DataCollatorForMAEDataset(object):
    """Collate function for MAE dataset - concatenates videos from batch"""
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function that concatenates all videos in the batch
        
        Args:
            features: list of dicts with 'pixel_values_videos' and 'video_grid_thw'
        
        Returns:
            pixel_values_videos: [total_patches, patch_pixel_dim] - concatenated patches
            video_grid_thw: [batch_size, 3] - (T, H, W) for each video
        """
        pixel_values_list = []
        video_grid_thw_list = []
        
        for feat in features:
            pixel_values_list.append(feat['pixel_values_videos'])
            # video_grid_thw should be [3], but we need to expand to [1, 3] if needed
            vg_thw = feat['video_grid_thw']
            if vg_thw.dim() == 1:
                vg_thw = vg_thw.unsqueeze(0)  # [3] -> [1, 3]
            video_grid_thw_list.append(vg_thw)
        
        # Concatenate all pixel values
        pixel_values_videos = torch.cat(pixel_values_list, dim=0)  # [total_patches, patch_pixel_dim]
        
        # Stack video_grid_thw
        video_grid_thw = torch.cat(video_grid_thw_list, dim=0)  # [batch_size, 3]
        
        return {
            'pixel_values_videos': pixel_values_videos,
            'video_grid_thw': video_grid_thw,
        }


def make_mae_data_module(
    model_id: str,
    processor: Optional[transformers.ProcessorMixin],
    data_path: str | List[str],
    video_resized_width: int = 224,
    video_resized_height: int = 224,
    video_min_pixels: int = 224 * 224,
    video_max_pixels: int = 224 * 224,
    fps: Optional[int] = None,
    nframes: Optional[int] = None,
    video_base_path: Optional[str] = None,
):
    """
    Create MAE data module
    
    Args:
        video_base_path: base path for video files (used when JSON contains relative paths)
    """
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_id)
    
    dataset = MAEVideoDataset(
        data_path=data_path,
        processor=processor,
        video_resized_width=video_resized_width,
        video_resized_height=video_resized_height,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        fps=fps,
        nframes=nframes,
        video_base_path=video_base_path,
    )
    
    data_collator = DataCollatorForMAEDataset()
    
    return {
        'dataset': dataset,
        'data_collator': data_collator,
    }

