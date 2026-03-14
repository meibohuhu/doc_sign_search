"""
GRPO Dataset for InternVL sign language video translation.
Returns prompt-only data (no tokenization) for GRPO generation loop.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from internvl.train.constants import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIGLIP_MEAN,
    SIGLIP_STD,
)


def _ensure_rgb(img):
    """Convert image to RGB if needed. Defined as a top-level function for pickling."""
    return img.convert('RGB') if img.mode != 'RGB' else img


def build_transform(input_size, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        mean, std = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        mean, std = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise ValueError(f'Unsupported normalize_type: {normalize_type}')
    return T.Compose([
        T.Lambda(_ensure_rgb),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def _compute_frame_indices(
    sample: str,
    vlen: int,
    input_fps: float,
    max_num_frames: int,
    min_num_frames: int,
    start_index: int = 0,
) -> List[int]:
    """Compute frame indices based on sampling strategy."""
    frame_indices: List[int] = []

    if 'fps' in sample:
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps if input_fps > 0 else 0
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int).tolist()
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            indices_to_keep = np.linspace(0, len(frame_indices) - 1, max_num_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices_to_keep]
    elif sample == 'random_start_every2':
        if vlen > 0:
            start_offset = np.random.randint(0, max(1, vlen))
            frame_indices = list(range(start_offset, vlen, 2))
            if len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
    else:
        start_offset = np.random.randint(0, 2)
        frame_indices = list(range(start_offset, vlen, 2))
        if len(frame_indices) > max_num_frames:
            indices_to_keep = np.linspace(0, len(frame_indices) - 1, max_num_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices_to_keep]

    if len(frame_indices) < min_num_frames and vlen >= min_num_frames:
        remaining_indices = [i for i in range(vlen) if i not in frame_indices]
        needed = min_num_frames - len(frame_indices)
        if len(remaining_indices) >= needed:
            if len(remaining_indices) == needed:
                additional = remaining_indices
            else:
                indices_to_add = np.linspace(0, len(remaining_indices) - 1, needed, dtype=int)
                additional = [remaining_indices[i] for i in indices_to_add]
            frame_indices = sorted(frame_indices + additional)

    if start_index > 0:
        frame_indices = [f + start_index for f in frame_indices]

    frame_indices = sorted(list(set(frame_indices)))
    return frame_indices


def _load_video_locally(
    video_path: str,
    max_num_frames: int,
    min_num_frames: int,
    sample: str = 'rand',
) -> List[Image.Image]:
    """Load video frames using decord with fallback to OpenCV."""
    load_errors = []

    # Try decord
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        frame_indices = _compute_frame_indices(
            sample=sample, vlen=total_frames, input_fps=fps,
            max_num_frames=max_num_frames, min_num_frames=min_num_frames,
        )
        frame_indices = [min(max(int(idx), 0), total_frames - 1) for idx in frame_indices]
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices

        frames = vr.get_batch(frame_indices).asnumpy()
        images = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return images
    except Exception as exc:
        load_errors.append(f'decord: {exc}')

    # Fallback to OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('cv2 failed to open')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        if total_frames == 0:
            raise RuntimeError('cv2 zero frames')

        frame_indices = _compute_frame_indices(
            sample=sample, vlen=total_frames, input_fps=fps,
            max_num_frames=max_num_frames, min_num_frames=min_num_frames,
        )
        frame_indices = [min(max(int(idx), 0), total_frames - 1) for idx in frame_indices]
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices

        frames = []
        frame_set = set(frame_indices)
        retrieved = 0
        for frame_idx in range(total_frames):
            success, frame = cap.read()
            if not success:
                break
            if frame_idx in frame_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                retrieved += 1
                if retrieved >= len(frame_indices):
                    break
        cap.release()
        if not frames:
            raise RuntimeError('cv2 no frames')
        return frames
    except Exception as exc:
        load_errors.append(f'opencv: {exc}')

    raise RuntimeError(f'Failed to load video: {"; ".join(load_errors)}')


class GRPOVideoDataset(Dataset):
    """
    Dataset for GRPO training. Returns raw prompt + video pixel_values.
    No tokenization (handled by trainer for left-padding during generation).
    """

    def __init__(
        self,
        jsonl_path: str,
        video_root: str,
        image_size: int = 224,
        max_num_frame: int = 64,
        min_num_frame: int = 8,
        sampling_method: str = 'fps16.0',
        normalize_type: str = 'imagenet',
    ):
        super().__init__()
        self.video_root = video_root
        self.image_size = image_size
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.transform = build_transform(image_size, normalize_type)

        # Load data
        self.data_items = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data_items.append(json.loads(line))

        print(f'[GRPOVideoDataset] Loaded {len(self.data_items)} items from {jsonl_path}')

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx) -> Dict[str, object]:
        item = self.data_items[idx]

        # Extract prompt and ground truth from conversations
        conversations = item['conversations']
        prompt_text = ""
        ground_truth = ""
        for conv in conversations:
            if conv['from'] == 'human':
                # Remove <video> tag - we'll add image tokens in the trainer
                prompt_text = conv['value'].replace('<video>\n', '').replace('<video>', '').strip()
            elif conv['from'] == 'gpt':
                ground_truth = conv['value'].strip()

        # Load video frames
        video_file = item.get('video', '')
        video_path = os.path.join(self.video_root, video_file)

        try:
            images = _load_video_locally(
                video_path,
                max_num_frames=self.max_num_frame,
                min_num_frames=self.min_num_frame,
                sample=self.sampling_method,
            )
        except Exception as e:
            print(f'[GRPOVideoDataset] Failed to load video {video_path}: {e}')
            # Return a single black frame as fallback
            images = [Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))]

        # Apply transforms to each frame
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)  # (num_frames, 3, H, W)
        num_patches = pixel_values.shape[0]
        image_flags = torch.ones(num_patches, dtype=torch.long)

        return {
            'prompt_text': prompt_text,
            'ground_truth': ground_truth,
            'pixel_values': pixel_values,
            'image_flags': image_flags,
            'num_patches': num_patches,
        }
