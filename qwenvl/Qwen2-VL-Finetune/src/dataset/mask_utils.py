import os
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

## mh: 2025-11-15: mask utils
@lru_cache(maxsize=128)  # Reduced from 256 to save memory
def _load_npz(mask_path: str, key: str) -> np.ndarray:
    with np.load(mask_path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"Mask key '{key}' not found in {mask_path}.")
        mask = data[key]
    if mask.ndim == 2:
        mask = mask[None, ...]
    # Convert to float32 and normalize to [0, 1] range
    # Handle both uint8 (0-255) and uint8/float32 (0-1) cases
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0  # Normalize from uint8 [0,255] to [0,1]
    return mask.clip(0.0, 1.0)


def load_mask(mask_path: Optional[str], key: str) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    if not os.path.exists(mask_path):
        return None
    return _load_npz(mask_path, key)


def temporal_resample(mask: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = mask.shape[0]
    if current_frames == target_frames:
        return mask

    if current_frames == 1:
        return mask.repeat(target_frames, 1, 1)

    indices = torch.linspace(0, current_frames - 1, target_frames, dtype=torch.float32)
    idx_low = torch.floor(indices).long()
    idx_high = torch.ceil(indices).long().clamp(max=current_frames - 1)
    weight_high = indices - idx_low.float()
    weight_low = 1.0 - weight_high

    mask_low = mask[idx_low]
    mask_high = mask[idx_high]
    weight_low = weight_low.unsqueeze(-1).unsqueeze(-1)
    weight_high = weight_high.unsqueeze(-1).unsqueeze(-1)
    return mask_low * weight_low + mask_high * weight_high


def prepare_mask_tensor(
    mask_np: np.ndarray,
    target_frames: int,
    target_height: int,
    target_width: int,
    dilation: int = 0,
    blur_kernel: int = 0,
) -> torch.Tensor:
    mask = torch.from_numpy(mask_np)
    
    # Temporal resample: adjust number of frames to match video
    mask = temporal_resample(mask, target_frames)
    
    # Reshape for interpolation: [T, H, W] -> [T, 1, H, W]
    mask = mask.unsqueeze(1)  # [T,1,H,W]
    
    # Spatial resize: resize mask from original size (e.g., 720x720) to target size (e.g., 224x224)
    # This ensures mask matches video resolution
    mask = F.interpolate(
        mask,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    ### mh: 2025-11-15: add dilation and blur kernel   --mask_dilation 3 --mask_blur_kernel 5
    if dilation and dilation > 1:
        pad = dilation // 2
        mask = F.max_pool2d(mask, kernel_size=dilation, stride=1, padding=pad)

    if blur_kernel and blur_kernel > 1:
        pad = blur_kernel // 2
        weight = 1.0 / (blur_kernel * blur_kernel)
        kernel = torch.full(
            (1, 1, blur_kernel, blur_kernel),
            weight,
            device=mask.device,
            dtype=mask.dtype,
        )
        mask = F.conv2d(mask, kernel, padding=pad)

    return mask.clamp(0.0, 1.0)
