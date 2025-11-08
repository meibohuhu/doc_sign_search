#!/usr/bin/env python3
"""
简化的注意力可视化脚本 for Qwen2.5-VL
只保留最核心的功能

Usage:
    python scripts/visualize_attention_simple.py \
        --video_path /path/to/video.mp4 \
        --frame_indices 0 5 10 \
        --output_dir ./attention_output
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import sys
import os
import cv2

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️  MediaPipe not installed. Will use default ROI region.")

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("⚠️  qwen_vl_utils not installed. Will use frame-by-frame processing.")

sys.path.append('/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def minmax_01(x, eps=1e-12):
    """Normalize tensor to [0, 1] range using min-max normalization"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x_min = x.amin()
    x_max = x.amax()
    return (x - x_min) / (x_max - x_min + eps)  # eps avoids div-by-zero


def create_mosaic_mask(attn_1d, num_frames, num_patches_h, num_patches_w):
    """
    Create a mosaic mask from 1D attention values (InternVL style).
    
    Args:
        attn_1d: 1D array of attention values [num_frames * num_patches_h * num_patches_w]
        num_frames: Number of frames
        num_patches_h: Number of patches in height dimension
        num_patches_w: Number of patches in width dimension
    
    Returns:
        mosaic: 2D numpy array [final_height, final_width] with values in [0, 1]
    """
    # Ensure attn_1d is numpy array
    if isinstance(attn_1d, torch.Tensor):
        attn_1d = attn_1d.float().cpu().numpy()
    
    # Normalize to [0, 1]
    attn_1d = minmax_01(attn_1d)
    
    # Reshape to [num_frames, num_patches_h, num_patches_w]
    attn_3d = attn_1d.reshape(num_frames, num_patches_h, num_patches_w)
    
    # Create mosaic: arrange frames in a grid
    # Calculate grid dimensions: try to make it roughly square
    cols = int(np.ceil(np.sqrt(num_frames)))
    rows = int(np.ceil(num_frames / cols))
    
    # Create empty mosaic
    mosaic_height = rows * num_patches_h
    mosaic_width = cols * num_patches_w
    mosaic = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)
    
    # Fill mosaic with frames
    for frame_idx in range(num_frames):
        row = frame_idx // cols
        col = frame_idx % cols
        start_h = row * num_patches_h
        end_h = start_h + num_patches_h
        start_w = col * num_patches_w
        end_w = start_w + num_patches_w
        mosaic[start_h:end_h, start_w:end_w] = attn_3d[frame_idx]
    
    return mosaic


def tokens_to_mosaic_mask(
    tokens_mask_1d,
    grid_t: int, grid_h: int, grid_w: int, 
    temporal_patch_size: int = 2
):
    """
    Convert 1D token importance/attention mask to 2D mosaic mask for video visualization.
    Based on visualize_importance_heatmap.py line 25-49.
    
    Args:
        tokens_mask_1d: 1D tensor/array of shape [grid_t * grid_h * grid_w] with importance/attention scores
        grid_t: Temporal dimension (number of frames)
        grid_h: Height dimension (patches per frame height)
        grid_w: Width dimension (patches per frame width)
        temporal_patch_size: Temporal patch size (default 2 for Qwen2VL)
    
    Returns:
        cell_mask: [grid_t*grid_h, temporal_patch_size*grid_w] (float {0.,1.})
    """
    if isinstance(tokens_mask_1d, np.ndarray):
        tokens_mask_1d = torch.from_numpy(tokens_mask_1d).float()
    elif not isinstance(tokens_mask_1d, torch.Tensor):
        tokens_mask_1d = torch.tensor(tokens_mask_1d, dtype=torch.float32)
    
    tokens_mask_1d = minmax_01(tokens_mask_1d)
    
    # Reshape to 3D: (grid_t, grid_h, grid_w)
    m3 = tokens_mask_1d.reshape(grid_t, grid_h, grid_w)
    
    # Repeat each spatial cell across all temporal slices horizontally
    # (gt, gh, gw, tps) -> (gt, gh, tps, gw) -> (gt*gh, tps*gw)
    cell_mask = (
        m3.unsqueeze(-1).expand(grid_t, grid_h, grid_w, temporal_patch_size)
        .permute(0, 1, 3, 2)
        .reshape(grid_t * grid_h, temporal_patch_size * grid_w)
    )
    
    return cell_mask


def get_vision_model(model):
    """Get vision model from Qwen2.5-VL architecture"""
    # Try base_model.model.model.visual path (for checkpoints with triple nesting)
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        if hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'visual'):
            visual = model.base_model.model.model.visual
            if hasattr(visual, 'vision_model'):
                return visual.vision_model
            return visual
    
    # Try model.model.visual path (for double nesting)
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        if hasattr(model.model.model, 'visual'):
            visual = model.model.model.visual
            if hasattr(visual, 'vision_model'):
                return visual.vision_model
            return visual
    
    # Try model.model.vision_tower path
    if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        vision_tower = model.model.vision_tower
        if isinstance(vision_tower, (list, tuple)) and len(vision_tower) > 0:
            if hasattr(vision_tower[0], 'vision_model'):
                return vision_tower[0].vision_model
            return vision_tower[0]
        elif hasattr(vision_tower, 'vision_model'):
            return vision_tower.vision_model
    
    # Try model.visual path (standard path)
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        visual = model.model.visual
        if hasattr(visual, 'vision_model'):
            return visual.vision_model
        return visual
    
    # Try direct visual path
    if hasattr(model, 'visual'):
        visual = model.visual
        if hasattr(visual, 'vision_model'):
            return visual.vision_model
        return visual
    
    raise AttributeError("Could not find vision model")


def extract_attention(model, pixel_values, image_grid_thw, all_layers=True):
    """
    Extract attention from vision model.
    
    Args:
        model: The Qwen2.5-VL model
        pixel_values: Input pixel values
        image_grid_thw: Image grid dimensions
        all_layers: If True, extract attention from all layers; if False, only last layer
    
    Returns:
        If all_layers=False: Single attention tensor [num_heads, seq_len, seq_len]
        If all_layers=True: List of attention tensors, one per layer
    """
    visual = get_vision_model(model)
    
    with torch.no_grad():
        
        
        # Manual extraction fallback
        print("   Using manual extraction...")
        try:
            if hasattr(visual, 'patch_embed'):
                hidden_states = visual.patch_embed(pixel_values)
                if len(hidden_states.shape) == 3:
                    hidden_states = hidden_states[0]
                elif len(hidden_states.shape) == 4:
                    hidden_states = hidden_states.flatten(2).transpose(1, 2)[0]
                # If shape is 2, already [seq_len, dim] - no conversion needed
                
                if hasattr(visual, 'blocks'):
                    all_attentions = []
                    
                    # Use hooks to capture attention from each layer
                    if all_layers:
                        # Register hooks to capture attention from each block
                        def make_hook(layer_idx):
                            def hook_fn(module, input, output):
                                # For attention modules, we need to intercept the QKV computation
                                # This is a simplified approach - we'll compute attention manually
                                pass
                            return hook_fn
                        
                        # Process each block/layer
                        for layer_idx, block in enumerate(visual.blocks):
                            if not hasattr(block, 'attn'):
                                continue
                            
                            # Get normalized hidden states for this layer
                            if hasattr(block, 'norm1'):
                                x_norm = block.norm1(hidden_states)
                            else:
                                x_norm = hidden_states
                            
                            attn_module = block.attn
                            
                            if hasattr(attn_module, 'qkv'):
                                print(f"   🔍 Extracting attention from layer {layer_idx}")
                                qkv = attn_module.qkv(x_norm)
                                if len(qkv.shape) == 3:
                                    qkv = qkv[0]
                                
                                embed_dim_3 = qkv.shape[-1]
                                embed_dim = embed_dim_3 // 3
                                num_heads = getattr(attn_module, 'num_heads', getattr(attn_module, 'num_attention_heads', 16))
                                head_dim = embed_dim // num_heads
                                seq_len = qkv.shape[0]
                                
                                qkv = qkv.reshape(seq_len, 3, num_heads, head_dim)
                                qkv = qkv.permute(1, 0, 2, 3)  # [3, seq_len, num_heads, head_dim]
                                q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [seq_len, num_heads, head_dim]
                                
                                # Reshape for attention computation
                                # Standard approach: permute to [num_heads, seq_len, head_dim]
                                q_heads = q.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
                                k_heads = k.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
                                v_heads = v.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
                                
                                # Compute attention scores: Q @ K^T
                                # q_heads: [num_heads, seq_len, head_dim]
                                # k_heads: [num_heads, seq_len, head_dim]
                                # k_heads.transpose(-2, -1): [num_heads, head_dim, seq_len]
                                # But we want: [num_heads, seq_len, seq_len]
                                # So: q_heads @ k_heads.transpose(-2, -1) gives wrong shape
                                # Correct: q_heads @ k_heads.transpose(-1, -2) = [num_heads, seq_len, head_dim] @ [num_heads, head_dim, seq_len]
                                attn_scores = torch.bmm(q_heads, k_heads.transpose(-2, -1)) * (head_dim ** -0.5)
                                attn_weights = attn_scores.softmax(dim=-1)  # [num_heads, seq_len, seq_len]
                                
                                all_attentions.append(attn_weights.detach().cpu())
                                
                                # Update hidden_states by properly applying the block
                                # Compute attention output: attn_weights @ v
                                # attn_weights: [num_heads, seq_len, seq_len]
                                # v_heads: [num_heads, seq_len, head_dim]
                                attn_output_heads = torch.bmm(attn_weights, v_heads)  # [num_heads, seq_len, head_dim]
                                # Reshape back: [num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim] -> [seq_len, embed_dim]
                                attn_output = attn_output_heads.permute(1, 0, 2).reshape(seq_len, -1)  # [seq_len, embed_dim]
                                
                                # Apply output projection if exists
                                if hasattr(attn_module, 'proj'):
                                    attn_output = attn_module.proj(attn_output)
                                
                                # Residual connection
                                hidden_states = hidden_states + attn_output
                                
                                # Apply feedforward and norm2 if exists
                                if hasattr(block, 'mlp'):
                                    if hasattr(block, 'norm2'):
                                        mlp_input = block.norm2(hidden_states)
                                    else:
                                        mlp_input = hidden_states
                                    mlp_output = block.mlp(mlp_input)
                                    hidden_states = hidden_states + mlp_output
                                print(f"   ✅Finish Extracted attention from layer {layer_idx}")
                                print(f"   ✅Attention shape: {attn_weights.shape}")
                            else:
                                # If no qkv, try to extract attention differently
                                print(f"   ⚠️  Layer {layer_idx} doesn't have qkv attribute, skipping")
                                # Still need to pass through the block
                                if hasattr(block, '__call__'):
                                    hidden_states = block(hidden_states)
                        
                        if all_attentions:
                            print(f"   ✅ Extracted all attention from {len(all_attentions)} layers")
                            return all_attentions
                    else:
                        print(f"   🔍 Extracting attention from last layer")
                        # Only extract last layer (original behavior)
                        last_block = visual.blocks[-1]
                        if hasattr(last_block, 'attn'):
                            attn_module = last_block.attn
                            if hasattr(last_block, 'norm1'):
                                x_norm = last_block.norm1(hidden_states)
                            else:
                                x_norm = hidden_states
                            
                            if hasattr(attn_module, 'qkv'):
                                qkv = attn_module.qkv(x_norm)
                                if len(qkv.shape) == 3:
                                    qkv = qkv[0]
                                
                                embed_dim_3 = qkv.shape[-1]
                                embed_dim = embed_dim_3 // 3
                                num_heads = getattr(attn_module, 'num_heads', getattr(attn_module, 'num_attention_heads', 16))
                                head_dim = embed_dim // num_heads
                                seq_len = qkv.shape[0]
                                
                                qkv = qkv.reshape(seq_len, 3, num_heads, head_dim)
                                qkv = qkv.permute(1, 0, 2, 3)  # [3, seq_len, num_heads, head_dim]
                                q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [seq_len, num_heads, head_dim]
                                
                                # Reshape for attention computation
                                # Standard approach: permute to [num_heads, seq_len, head_dim]
                                q_heads = q.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
                                k_heads = k.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
                                v_heads = v.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
                                
                                # Compute attention scores: Q @ K^T using batch matrix multiplication
                                # q_heads: [num_heads, seq_len, head_dim]
                                # k_heads: [num_heads, seq_len, head_dim]
                                # k_heads.transpose(-2, -1): [num_heads, head_dim, seq_len]
                                # torch.bmm: [num_heads, seq_len, head_dim] @ [num_heads, head_dim, seq_len] = [num_heads, seq_len, seq_len]
                                attn_scores = torch.bmm(q_heads, k_heads.transpose(-2, -1)) * (head_dim ** -0.5)
                                attn_weights = attn_scores.softmax(dim=-1)  # [num_heads, seq_len, seq_len]
                                return attn_weights.detach().cpu()
                            
        except Exception as e:
            print(f"   Manual extraction failed: {e}")
            import traceback
            traceback.print_exc()
    
    return None


def extract_attention_all_layers(model, pixel_values, image_grid_thw):
    """
    Extract attention from all layers of the vision model.
    Convenience wrapper around extract_attention(all_layers=True).
    
    Returns:
        List of attention tensors, one per layer [num_heads, seq_len, seq_len]
    """
    return extract_attention(model, pixel_values, image_grid_thw, all_layers=True)


def process_attention(attn, num_cls_tokens=0):
    """Convert attention to 1D heatmap"""
    # Average heads: [heads, seq, seq] -> [seq, seq]
    if len(attn.shape) == 3:
        attn = attn.mean(dim=0)
    
    # Remove CLS tokens and average queries
    if num_cls_tokens > 0 and attn.shape[0] > num_cls_tokens:
        attn = attn[num_cls_tokens:, num_cls_tokens:]
    
    attn_1d = attn.mean(dim=0)  # [num_patches]
    
    if isinstance(attn_1d, torch.Tensor):
        return attn_1d.float().cpu().numpy()
    return attn_1d


def create_heatmap(attn_1d, image, num_patches_h=None, num_patches_w=None):
    """
    Create heatmap and upscale to image size
    
    Args:
        attn_1d: 1D attention weights [num_patches] in row-major order (top-to-bottom, left-to-right)
        image: Original image (PIL Image or numpy array)
        num_patches_h: Number of patches in height dimension (if None, will be inferred)
        num_patches_w: Number of patches in width dimension (if None, will be inferred)
    
    Returns:
        2D heatmap array matching image dimensions
    """
    # Convert Tensor to numpy if needed
    if isinstance(attn_1d, torch.Tensor):
        attn_1d = attn_1d.float().cpu().numpy()
    
    if isinstance(image, Image.Image):
        H, W = image.size[1], image.size[0]  # PIL: (width, height)
    else:
        H, W = image.shape[:2]  # numpy: (height, width)
    
    num_patches = len(attn_1d)
    
    # Use provided patch grid dimensions, or fallback to inference
    if num_patches_h is None or num_patches_w is None:
        num_patches_per_side = int(np.sqrt(num_patches))
        if num_patches_per_side * num_patches_per_side == num_patches:
            num_patches_h = num_patches_w = num_patches_per_side
        else:
            # Fallback: use aspect ratio
            aspect_ratio = W / H if H > 0 else 1.0
            num_patches_h = int(np.sqrt(num_patches / aspect_ratio))
            num_patches_w = int(num_patches / num_patches_h) if num_patches_h > 0 else int(np.sqrt(num_patches))
    
    # Verify patch grid matches num_patches
    if num_patches_h * num_patches_w != num_patches:
        print(f"   ⚠️  Warning: patch grid {num_patches_h}x{num_patches_w}={num_patches_h*num_patches_w} != num_patches={num_patches}")
        # Try to fix by adjusting
        num_patches_per_side = int(np.sqrt(num_patches))
        if num_patches_per_side * num_patches_per_side == num_patches:
            num_patches_h = num_patches_w = num_patches_per_side
            print(f"   📐 Using square grid: {num_patches_h}x{num_patches_w}")
    
    # Reshape to 2D grid (row-major order: first dim is height, second is width)
    # ViT outputs patches in row-major: 
    #   [patch_0_0, patch_0_1, ..., patch_0_w-1, patch_1_0, patch_1_1, ...]
    #   i.e., top row left-to-right, then next row left-to-right, etc.
    attn_grid = attn_1d.reshape(num_patches_h, num_patches_w)
    
    # Debug: Check if attention is concentrated in one corner (common issue with black images)
    # For black images, all patches are similar, so normalization can amplify small differences
    corner_values_before_norm = {
        'top-left': float(attn_grid[0, 0]),
        'top-right': float(attn_grid[0, -1]),
        'bottom-left': float(attn_grid[-1, 0]),
        'bottom-right': float(attn_grid[-1, -1])
    }
    
    attn_min = attn_grid.min()
    attn_max = attn_grid.max()
    attn_range = attn_max - attn_min
    attn_mean = attn_grid.mean()
    attn_std = attn_grid.std()
    
    # Check if this is a black/uniform image scenario
    if attn_range < 0.01:  # Very small range indicates uniform attention
        print(f"   ⚠️  Uniform attention detected (range={attn_range:.6f}, std={attn_std:.6f})")
        print(f"   💡 For black/uniform images:")
        print(f"      - All patches are similar → similar embeddings → similar attention values")
        print(f"      - Min-max normalization amplifies tiny differences (noise or numerical artifacts)")
        print(f"      - The 'hot spots' on black background are likely normalization artifacts, not real attention focus")
        print(f"      - Real attention should be nearly uniform for uniform input")
        
        # Show actual corner values to demonstrate they're almost identical
        print(f"   📊 Corner values (before normalization): {corner_values_before_norm}")
        max_corner_diff = max(corner_values_before_norm.values()) - min(corner_values_before_norm.values())
        print(f"   📊 Maximum corner difference: {max_corner_diff:.8f} (this is what gets amplified!)")
    
    # Normalize for visualization
    # CRITICAL: For uniform/black images, min-max normalization creates false patterns
    # The attention values are essentially all the same, but normalization makes tiny differences visible
    if attn_range < 1e-8:
        # Truly uniform (numerical precision limit)
        print(f"   ⚠️  Numerically uniform attention (range < 1e-8)")
        # Just use zeros or a small constant to indicate uniform attention
        attn_grid = np.ones_like(attn_grid) * 0.5  # Uniform gray
    elif attn_range < 0.01:
        # Very uniform - warn that normalization creates artifacts
        attn_grid = (attn_grid - attn_min) / (attn_range + 1e-8)
        print(f"   ⚠️  Normalization artifact warning: {attn_range:.6f} range amplified to [0, 1]")
        print(f"   💡 Hot spots on black background are likely artifacts, not real attention patterns")
    else:
        attn_grid = (attn_grid - attn_min) / (attn_range + 1e-8)
    
    # Upsample to match original image size
    # CRITICAL: The patch grid corresponds to processor input size (e.g., 336x336 or 448x448),
    # but we want to visualize on the original frame size (e.g., 720p). The upsampling
    # handles this mapping correctly as long as we have the right patch grid dimensions.
    # Convert attn_grid to Tensor if it's not already
    if isinstance(attn_grid, torch.Tensor):
        attn_tensor = attn_grid.unsqueeze(0).unsqueeze(0).float()
    else:
        attn_tensor = torch.from_numpy(attn_grid).unsqueeze(0).unsqueeze(0).float()
    upsampled = torch.nn.functional.interpolate(
        attn_tensor, size=(H, W), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    
    return upsampled


def visualize_attention_overlay(frame, attn_map, threshold=0.3, output_path=None):
    """
    Create filtered attention overlay with threshold to remove background bleed
    
    Args:
        frame: RGB frame as a numpy array (H, W, 3)
        attn_map: 2D numpy array of attention weights (h, w)
        threshold: minimum attention value to visualize (0.0-1.0)
        output_path: Optional path to save the visualization
    """
    # --- 1️⃣ Normalize attention map ---
    attn = attn_map - attn_map.min()
    attn = attn / (attn.max() + 1e-8)
    
    # --- 2️⃣ Threshold low attention to remove background bleed ---
    attn[attn < threshold] = 0.0
    
    # --- 3️⃣ Apply a colormap (hot/yellow means strong attention) ---
    # attn_map is already resized to frame size by create_heatmap, but resize again to ensure exact match
    if attn.shape != (frame.shape[0], frame.shape[1]):
        attn = cv2.resize(attn, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # --- 5️⃣ Blend with original frame ---
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    
    # --- 6️⃣ Plot and save ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title(f"Filtered Attention Overlay (threshold={threshold})")
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


def visualize(frame, attn_1d, output_path,
              use_overlay=False, overlay_threshold=0.15, num_patches_h=None, num_patches_w=None):
    """Create separate visualizations for attention heatmap and ROI boxes"""
    import matplotlib.patches as patches
    
    # Create heatmap from 1D attention (using true patch grid if available)
    heatmap = create_heatmap(attn_1d, frame, num_patches_h=num_patches_h, num_patches_w=num_patches_w)
    
    # Always save standard heatmap visualization (overlaid on original frame)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(frame)
    im = ax.imshow(heatmap, cmap='hot', alpha=0.6, interpolation='bilinear')
    ax.set_title('Attention Heatmap')
    ax.axis('off')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Always save pure attention heatmap (without original frame overlay)
    heatmap_only_path = output_path.replace('_attention.png', '_attention_only.png')
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
    ax.set_title('Attention Heatmap (Pure)')
    ax.axis('off')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(heatmap_only_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Optionally save filtered overlay visualization
    if use_overlay:
        overlay_path = output_path.replace('_attention.png', '_attention_overlay.png')
        visualize_attention_overlay(frame, heatmap, threshold=overlay_threshold, output_path=overlay_path)

def main():
    parser = argparse.ArgumentParser(description='Simple attention visualization')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--frame_indices', type=int, nargs='+', default=[0, 5, 10])
    parser.add_argument('--output_dir', type=str, default='./attention_output')
    parser.add_argument('--base_model_name', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--num_cls_tokens', type=int, default=0)
    parser.add_argument('--use_overlay', action='store_true',
                        help='Use filtered overlay visualization (removes low attention background)')
    parser.add_argument('--overlay_threshold', type=float, default=0.10,
                        help='Threshold for overlay visualization (0.0-1.0, higher = more filtered). Default: 0.15')
    parser.add_argument('--processor_input_size', type=int, default=None,
                        help='Processor input size (e.g., 336 or 448). If None, will use processor default.')
    parser.add_argument('--use_video_mode', action='store_true',
                        help='Process video using Qwen2.5-VL video processing (like qwenvl). Processes entire video then extracts attention for specified frames.')
    parser.add_argument('--video_fps', type=float, default=1.0,
                        help='Video FPS for frame sampling (frames per second to extract from video)')
    parser.add_argument('--all_layers', action='store_true',
                        help='Extract attention from all layers instead of just the last layer')
    parser.add_argument('--save_mosaic_mask', action='store_true',
                        help='Save mosaic mask visualization (for video mode, similar to visualize_importance_heatmap.py)')
    parser.add_argument('--temporal_patch_size', type=int, default=2,
                        help='Temporal patch size for mosaic mask (default: 2 for Qwen2VL)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load video frames
    print(f"📹 Loading video: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return
    
    frames = []
    for idx in args.frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    
    if not frames:
        print("❌ No frames loaded")
        return
    
    print(f"✅ Loaded {len(frames)} frames")
    
    # Load model with error handling to prevent segmentation faults
    print(f"🤖 Loading model: {args.base_model_name}")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if args.checkpoint_path:
        print(f"📦 Loading checkpoint: {args.checkpoint_path}")
        try:
            if os.path.exists(os.path.join(args.checkpoint_path, 'non_lora_state_dict.bin')):
                state_dict = torch.load(os.path.join(args.checkpoint_path, 'non_lora_state_dict.bin'), map_location='cpu')
                print(f"   Loading non-LoRA weights ({len(state_dict)} parameters)")
                vision_keys = [k for k in state_dict.keys() if 'visual' in k or 'vision' in k]
                if vision_keys:
                    print(f"   Vision weights found: {len(vision_keys)} keys")
                    print(f"   Sample keys: {vision_keys[:3]}")
                model.load_state_dict(state_dict, strict=False)
                print(f"   ✅ Non-LoRA weights loaded")
            if os.path.exists(os.path.join(args.checkpoint_path, 'adapter_config.json')):
                model = PeftModel.from_pretrained(model, args.checkpoint_path)
                print(f"   ✅ LoRA adapter loaded")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Continuing with base model...")
    
    try:
        processor = AutoProcessor.from_pretrained(args.base_model_name, trust_remote_code=True)
        print(f"   ✅ Processor loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load processor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    model.eval()
    
    # Process video or frames
    if args.use_video_mode and HAS_QWEN_VL_UTILS:
        # Process entire video using Qwen2.5-VL video processing (like qwenvl)
        print(f"📹 Processing video using Qwen2.5-VL video pipeline...")
        print(f"   This matches how qwenvl processes videos during training/inference")
        
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": args.video_path, "fps": args.video_fps},
                    {"type": "text", "text": "Translate the American Sign Language in this video to English."}
                ]
            }
        ]
        
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Step 1: Apply chat template
            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False
            )
            
            # Step 2: Extract vision inputs using process_vision_info
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                conversation, 
                return_video_kwargs=True
            )
            
            print(f"   📹 Video inputs extracted: {len(video_inputs) if video_inputs else 0} video(s)")
            
            # Step 3: Process with processor (official Qwen2.5-VL way)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                **video_kwargs
            )
            
            print(f"   ✅ Video processed successfully")
            
            # Extract video information
            if 'pixel_values_videos' in inputs:
                pixel_values_videos = inputs['pixel_values_videos'].to(model.device)
                video_grid_thw = inputs.get('video_grid_thw', None)
                
                print(f"   📐 Video grid shape: {video_grid_thw.shape if video_grid_thw is not None else 'None'}")
                print(f"   📐 Pixel values shape: {pixel_values_videos.shape}")
                
                # Process video to extract attention for specified frames
                process_video_attention(
                    model, pixel_values_videos, video_grid_thw, 
                    frames, args, conversation, video_inputs, video_kwargs
                )
            else:
                print(f"   ⚠️  No video pixel values found, falling back to frame-by-frame processing")
                # # Fallback to frame-by-frame
                # for frame_idx, frame in frames:
                #     process_single_frame(model, processor, frame_idx, frame, args, original_frame=None)
        except Exception as e:
            print(f"   ❌ Video mode failed: {e}")
            import traceback
            traceback.print_exc()
            # print(f"   📸 Falling back to frame-by-frame image mode")
            # for frame_idx, frame in frames:
            #     process_single_frame(model, processor, frame_idx, frame, args, original_frame=None)
    else:
        # Default: process each frame individually
        print(f"🔄 Processing frames individually...")
        # for frame_idx, frame in frames:
        #     process_single_frame(model, processor, frame_idx, frame, args, original_frame=None)


def process_video_attention(model, pixel_values_videos, video_grid_thw, frames, args, 
                           conversation, video_inputs, video_kwargs):
    """
    Extract attention from video for specified frames.
    
    Args:
        model: The Qwen2.5-VL model
        pixel_values_videos: Processed video pixel values [batch, num_patches, hidden_dim]
        video_grid_thw: Video grid info [num_patches_t, num_patches_h, num_patches_w]
        frames: List of (frame_idx, frame_array) tuples for frames to visualize
        args: Command line arguments
        conversation: Conversation format used for processing
        video_inputs: Video inputs from process_vision_info
        video_kwargs: Video processing kwargs
    """
    print(f"\n🎬 Extracting attention from video for {len(frames)} frames...")
    
    # Prepare video_grid_thw for model
    if not isinstance(video_grid_thw, torch.Tensor):
        video_grid_thw = torch.tensor(video_grid_thw, dtype=torch.long, device=model.device)
    else:
        video_grid_thw = video_grid_thw.to(model.device)
    
    # Extract attention from entire video
    print(f"   🔍 Extracting attention from video...")
    try:
        # Check if all_layers is set (could be in args or default to False for backward compatibility)
        # all_layers = getattr(args, 'all_layers', False)
        all_layers = True
        attn_video = extract_attention(model, pixel_values_videos, video_grid_thw, all_layers=all_layers)
        if attn_video is None:
            print(f"   ❌ Failed to extract attention from video")
            return
        
        # Handle list of attentions (when all_layers=True)
        if isinstance(attn_video, list):
            print(f"   ✅ Video attention list length: {len(attn_video)}")
            print(f"   ✅ Video attention extracted: {len(attn_video)} layers")
            # Store all layers for potential multi-layer visualization
            all_layers_attn = attn_video
            # For now, use the last layer (can be extended to visualize all layers)
            print(f"   📊 Using last layer attention for visualization")
            attn_video = attn_video[-1]  # Use last layer
            print(f"   📊 Last layer shape: {attn_video.shape if isinstance(attn_video, torch.Tensor) else type(attn_video)}")
            # Store reference to all layers in args for later use
            args.all_layers_attn = all_layers_attn
        else:
            print(f"   ✅ Video attention extracted: shape {attn_video.shape if isinstance(attn_video, torch.Tensor) else type(attn_video)}")
            args.all_layers_attn = None
    except Exception as e:
        print(f"   ❌ Error extracting video attention: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Parse video_grid_thw to get dimensions
    if isinstance(video_grid_thw, torch.Tensor):
        grid_list = video_grid_thw[0].tolist() if video_grid_thw.dim() > 1 else video_grid_thw.tolist()
    else:
        grid_list = video_grid_thw
    
    if not isinstance(grid_list, (list, tuple)) or len(grid_list) < 3:
        print(f"   ⚠️  Invalid video_grid_thw format: {grid_list}")
        return
    
    num_patches_t, num_patches_h, num_patches_w = grid_list[0], grid_list[1], grid_list[2]
    print(f"   📐 Video grid: T={num_patches_t}, H={num_patches_h}, W={num_patches_w}")
    
    # Calculate how many patches per frame
    patches_per_frame = num_patches_h * num_patches_w
    print(f"   📐 Patches per frame: {patches_per_frame}")
    
    # Process attention: [heads, seq_len, seq_len] or [seq_len, seq_len]
    # For video: seq_len = num_patches_t * num_patches_h * num_patches_w
    # We need to extract attention for each frame separately
    if isinstance(attn_video, torch.Tensor):
        print(f"   🔍 Raw attention tensor shape: {attn_video.shape}")
        if len(attn_video.shape) == 3:
            # [heads, seq_len, seq_len] - average heads first
            print(f"   🔍 Averaging over {attn_video.shape[0]} heads")
            attn_video = attn_video.mean(dim=0)
            print(f"   🔍 After head averaging: {attn_video.shape}")
        # Now: [seq_len, seq_len]
        # Convert BFloat16 to float32 before converting to numpy (NumPy doesn't support BFloat16)
        if attn_video.dtype == torch.bfloat16:
            attn_video = attn_video.float()
        attn_video = attn_video.cpu().numpy()
    
    total_patches = attn_video.shape[0]
    expected_total = num_patches_t * patches_per_frame
    print(f"   🔍 Attention shape: {attn_video.shape}, expected patches: {expected_total}")
    
    if total_patches != expected_total:
        print(f"   ⚠️  Mismatch: attention has {total_patches} patches, expected {expected_total}")
        # Try to proceed anyway
    
    # For each requested frame, extract its attention
    # Map frame indices to time indices in video_grid_thw
    # This depends on how the video was sampled (video_fps)
    frame_indices_in_video = [frame_idx for frame_idx, _ in frames]
    
    # Get actual video frame count from video_inputs
    # video_inputs should contain information about sampled frames
    try:
        # Try to infer which frames were actually sampled
        # This is approximate - we assume uniform sampling based on video_fps
        cap = cv2.VideoCapture(args.video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps_original = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate sampling rate
        sampling_rate = int(video_fps_original / args.video_fps) if args.video_fps > 0 else 1
        print(f"   📊 Video info: {total_video_frames} frames @ {video_fps_original:.2f} fps")
        print(f"   📊 Sampling: 1 frame every {sampling_rate} frames (target fps: {args.video_fps})")
        
        # Map requested frame indices to sampled video indices
        # The video was sampled, so we need to find which sampled frame index corresponds to each requested frame
        sampled_frame_indices = [i for i in range(0, total_video_frames, sampling_rate)][:num_patches_t]
        
        for requested_frame_idx, frame_array in frames:
            # Find closest sampled frame index
            closest_sampled_idx = min(range(len(sampled_frame_indices)), 
                                     key=lambda i: abs(sampled_frame_indices[i] - requested_frame_idx))
            
            if closest_sampled_idx >= num_patches_t:
                print(f"   ⚠️  Frame {requested_frame_idx} is beyond video time dimension (T={num_patches_t})")
                continue
            
            print(f"\n📸 Frame {requested_frame_idx} (mapped to video index {closest_sampled_idx}):")
            
            # Extract attention for this frame
            # Attention shape is [total_patches, total_patches]
            # We want patches corresponding to frame at time index closest_sampled_idx
            start_patch = closest_sampled_idx * patches_per_frame
            end_patch = start_patch + patches_per_frame
            
            print(f"   🔍 Extracting patches {start_patch}:{end_patch} from attention matrix {attn_video.shape}")
            
            if end_patch > total_patches:
                print(f"   ⚠️  Frame patches {start_patch}:{end_patch} exceed total {total_patches}")
                continue
            
            # Extract attention map for this frame's patches
            # Attention matrix shape: [total_patches, total_patches]
            # Each element [i, j] means: how much patch i attends to patch j
            
            # Extract the attention submatrix for this frame
            # This gives us attention within this frame only
            frame_attn_matrix = attn_video[start_patch:end_patch, start_patch:end_patch]  # [patches_per_frame, patches_per_frame]
            print(f"   🔍 Frame attention submatrix shape: {frame_attn_matrix.shape}, expected: ({patches_per_frame}, {patches_per_frame})")
            
            # Calculate attention map: average over queries (columns) to see which patches are attended to
            # This gives: for each patch j, how much attention it receives from all patches in this frame
            attn_1d = frame_attn_matrix.mean(axis=0)  # [patches_per_frame]
            
            # Alternative interpretation: average over keys (rows) to see how much each patch attends
            # This gives: for each patch i, how much it attends to other patches
            # attn_1d = frame_attn_matrix.mean(axis=1)  # [patches_per_frame]
            
            # Alternative: use attention received (column sum) - which patches receive most attention
            # attn_1d = frame_attn_matrix.sum(axis=0)  # [patches_per_frame]
            
            print(f"   📊 Calculated attention map: shape {attn_1d.shape}, range: [{attn_1d.min():.4f}, {attn_1d.max():.4f}]")
            print(f"   📊 Attention map statistics: mean={attn_1d.mean():.4f}, std={attn_1d.std():.4f}")
            
            # Visualize this frame with attention map
            visualize_frame_attention(
                requested_frame_idx, frame_array, attn_1d, 
                num_patches_h, num_patches_w, args
            )
            
            # If all layers were extracted, optionally visualize other layers
            if hasattr(args, 'all_layers_attn') and args.all_layers_attn is not None:
                print(f"   💡 Note: {len(args.all_layers_attn)} layers extracted. Currently visualizing last layer only.")
                print(f"   💡 To visualize all layers, modify visualize_frame_attention to support multi-layer output.")
        
        # Generate mosaic mask for entire video if requested
        if args.save_mosaic_mask:
            print(f"\n🎨 Generating mosaic mask for entire video...")
            try:
                print(f"   🔍 Attn video shape: {attn_video.shape}")
                # Calculate overall attention map for entire video
                # Average attention over all queries to get per-patch attention
                attn_video_1d = attn_video.mean(axis=0)  # [total_patches]
                # Generate mosaic mask using create_mosaic_mask (InternVL style)
                mask = create_mosaic_mask(
                    attn_video_1d, 
                    num_patches_t, num_patches_h, num_patches_w
                )
                
                # Convert to RGB image with colormap
                # Apply colormap (hot) to make it more visible
                mask_rgb = plt.cm.hot(mask)[:, :, :3]  # Get RGB, ignore alpha
                mask_rgb = (mask_rgb * 255).astype(np.uint8)
                
                # Save mosaic mask
                mosaic_mask_path = os.path.join(args.output_dir, "video_mosaic_mask.png")
                Image.fromarray(mask_rgb).save(mosaic_mask_path)
                print(f"   ✅ Saved mosaic mask to {mosaic_mask_path}")
                print(f"   📊 Mosaic mask shape: {mask_rgb.shape}")
            except Exception as e:
                print(f"   ⚠️  Failed to generate mosaic mask: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"   ❌ Error processing video frames: {e}")
        import traceback
        traceback.print_exc()


def visualize_frame_attention(frame_idx, frame, attn_1d, num_patches_h, num_patches_w, args):
    """Visualize attention for a single frame"""
    
    # Create output path
    checkpoint_name = os.path.basename(args.checkpoint_path) if args.checkpoint_path else "base"
    output_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}_attention.png")
    
    # Visualize
    visualize(
        frame, attn_1d, output_path, 
        use_overlay=args.use_overlay,
        overlay_threshold=args.overlay_threshold,
        num_patches_h=num_patches_h,
        num_patches_w=num_patches_w
    )
    
    print(f"   ✅ Saved to {output_path}")


def process_single_frame(model, processor, frame_idx, frame, args, original_frame=None):
    """Process a single frame and extract attention"""
    if original_frame is None:
        original_frame = frame.copy()
    
    print(f"\n📸 Frame {frame_idx}:")
    
    # Resize frame to processor input size if specified (ensures consistent processing)
    image = Image.fromarray(frame)
    original_frame = frame.copy()  # Keep original for visualization
    
    if args.processor_input_size is not None:
        target_size = args.processor_input_size
        # Resize to square target_size x target_size for consistent processing
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        # Convert back to numpy for frame-based operations
        frame = np.array(image)
        print(f"   🔄 Resized frame to {target_size}x{target_size} for processor (original: {original_frame.shape[0]}x{original_frame.shape[1]})")
    
    # Process as single image
    try:
        inputs = processor(images=[image], text=["<video>\nAnalyze this frame."], return_tensors="pt")
    except Exception as e:
        print(f"   ❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        print(f"   ⏭️  Skipping this frame...")
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    
    # Handle both image and video inputs (Qwen2.5-VL video mode uses pixel_values_videos)
    if 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values'].to(model.device)
        image_grid_thw = inputs.get('image_grid_thw', None)
        grid_key = 'image_grid_thw'
        pixel_key = 'pixel_values'
    elif 'pixel_values_videos' in inputs:
        # Video mode - use video inputs
        pixel_values = inputs['pixel_values_videos'].to(model.device)
        image_grid_thw = inputs.get('video_grid_thw', None)
        grid_key = 'video_grid_thw'
        pixel_key = 'pixel_values_videos'
        print(f"   📹 Using video pixel values and {grid_key}")
    else:
        print(f"   ⚠️  No pixel_values found in inputs")
        return
    
    # Extract patch grid dimensions directly from image_grid_thw or video_grid_thw
    # Format: [num_patches_t, num_patches_h, num_patches_w] or [[num_patches_t, num_patches_h, num_patches_w]]
    num_patches_h = None
    num_patches_w = None
    
    if grid_key in inputs and image_grid_thw is not None:
        try:
            # Handle tensor format: could be [1, 3] or [batch, 3]
            if isinstance(image_grid_thw, torch.Tensor):
                grid_list = image_grid_thw[0].tolist()  # Take first element, convert to list
            else:
                grid_list = image_grid_thw[0] if isinstance(image_grid_thw, (list, tuple)) and len(image_grid_thw) > 0 else image_grid_thw
            
            if isinstance(grid_list, (list, tuple)) and len(grid_list) >= 3:
                num_patches_t, val_h, val_w = grid_list[0], grid_list[1], grid_list[2]
                print(f"   📐 Raw image_grid_thw: [T={num_patches_t}, H={val_h}, W={val_w}]")
                
                # Store raw values - we'll verify after extracting attention
                # H and W might be:
                # 1. Patch grid dimensions (H_patches × W_patches = num_patches)
                # 2. Processor input dimensions in pixels (H_pixels, W_pixels)
                # 3. Something else entirely
                candidate_h, candidate_w = val_h, val_w
                
                # Don't assign to num_patches_h/w yet - wait for verification
            else:
                print(f"   ⚠️  Unexpected image_grid_thw format: {grid_list}")
        except Exception as e:
            print(f"   ⚠️  Failed to parse image_grid_thw: {e}")
    
    # Prepare image_grid_thw for model forward (still needed for attention extraction)
    if image_grid_thw is None:
        pv_shape = pixel_values.shape
        patches_per_side = int(np.sqrt(pv_shape[-2] // pv_shape[0]))
        if patches_per_side * patches_per_side * pv_shape[0] == pv_shape[-2]:
            image_grid_thw = torch.tensor([[pv_shape[0], patches_per_side, patches_per_side]], 
                                         dtype=torch.long, device=model.device)
    
    if not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long, device=model.device)
    else:
        image_grid_thw = image_grid_thw.to(model.device)
        
    # Extract and visualize with error handling to prevent segmentation faults
    try:
        # Check if all_layers is requested (default to False for backward compatibility)
        all_layers = getattr(args, 'all_layers', True)
        attn = extract_attention(model, pixel_values, image_grid_thw, all_layers=all_layers)
        if attn is None:
            print(f"   ⚠️  Failed to extract attention")
            return
        
        # Handle list of attentions (when all_layers=True)
        if isinstance(attn, list):
            print(f"   ✅ Extracted attention from {len(attn)} layers")
            # For now, use the last layer (can be extended to visualize all layers)
            print(f"   📊 Using last layer attention for visualization")
            attn = attn[-1]  # Use last layer
    except Exception as e:
        print(f"   ❌ Error extracting attention: {e}")
        import traceback
        traceback.print_exc()
        print(f"   ⏭️  Skipping this frame...")
        # Clear GPU cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    
    attn_1d = process_attention(attn, args.num_cls_tokens)
    actual_num_patches = attn_1d.shape[0]
    print(f"   Attention shape: {attn_1d.shape}, range: [{attn_1d.min():.4f}, {attn_1d.max():.4f}]")
    print(f"   🔍 Debug: num_patches={actual_num_patches}")
    
    # Analyze attention distribution for black images
    attn_mean = attn_1d.mean()
    attn_std = attn_1d.std()
    attn_range = attn_1d.max() - attn_1d.min()
    print(f"   📊 Attention stats: mean={attn_mean:.6f}, std={attn_std:.6f}, range={attn_range:.6f}")
    
    # Check if attention is very uniform (typical for black images)
    if attn_range < 0.01:  # Very small range
        print(f"   ⚠️  Very uniform attention distribution (range={attn_range:.6f})")
        print(f"   💡 This suggests all patches have similar attention values")
        print(f"   💡 Min-max normalization will amplify tiny differences into visible patterns")
        print(f"   💡 The 'hot spots' you see may be normalization artifacts, not real attention")
    
    # Now verify image_grid_thw values with actual num_patches (after we know num_patches)
    if grid_key in inputs and image_grid_thw is not None and num_patches_h is None:
        try:
            if isinstance(image_grid_thw, torch.Tensor):
                grid_list = image_grid_thw[0].tolist()
            else:
                grid_list = image_grid_thw[0] if isinstance(image_grid_thw, (list, tuple)) else image_grid_thw
            
            if isinstance(grid_list, (list, tuple)) and len(grid_list) >= 3:
                _, candidate_h, candidate_w = grid_list[0], grid_list[1], grid_list[2]
                
                # Try Option 1: H and W are patch grid dimensions
                if candidate_h * candidate_w == actual_num_patches:
                    num_patches_h, num_patches_w = candidate_h, candidate_w
                    print(f"   ✅ Interpreting as patch grid: {num_patches_h}x{num_patches_w}")
                # Try Option 2: H and W are processor input pixels (divide by patch_size=14)
                else:
                    PATCH_SIZE = 14
                    if candidate_h % PATCH_SIZE == 0 and candidate_w % PATCH_SIZE == 0:
                        patches_h = candidate_h // PATCH_SIZE
                        patches_w = candidate_w // PATCH_SIZE
                        if patches_h * patches_w == actual_num_patches:
                            num_patches_h, num_patches_w = patches_h, patches_w
                            print(f"   ✅ Interpreting as processor input pixels: {candidate_h}x{candidate_w} → patch grid: {patches_h}x{patches_w}")
                        else:
                            print(f"   ⚠️  Cannot interpret {candidate_h}x{candidate_w}: pixels→{patches_h}x{patches_w}={patches_h*patches_w} != {actual_num_patches}")
                    else:
                        print(f"   ⚠️  Cannot interpret {candidate_h}x{candidate_w} (not divisible by {PATCH_SIZE})")
        except Exception as e:
            print(f"   ⚠️  Failed to verify {grid_key}: {e}")
    
    # Fallback if extraction failed or verify patch grid matches actual num_patches
    if num_patches_h is None or num_patches_w is None:
        num_patches_per_side = int(np.sqrt(len(attn_1d)))
        if num_patches_per_side * num_patches_per_side == len(attn_1d):
            num_patches_h = num_patches_w = num_patches_per_side
            print(f"   📐 Using fallback square grid: {num_patches_h}x{num_patches_w}")
        else:
            # Use aspect ratio-based fallback
            aspect_ratio = frame.shape[1] / frame.shape[0] if frame.shape[0] > 0 else 1.0
            num_patches_h = int(np.sqrt(len(attn_1d) / aspect_ratio))
            num_patches_w = int(len(attn_1d) / num_patches_h) if num_patches_h > 0 else int(np.sqrt(len(attn_1d)))
            print(f"   📐 Using fallback aspect-ratio grid: {num_patches_h}x{num_patches_w}")
    else:
        # Verify patch grid matches actual num_patches
        if num_patches_h * num_patches_w != len(attn_1d):
            print(f"   ⚠️  Patch grid mismatch! {num_patches_h}x{num_patches_w}={num_patches_h*num_patches_w} != {len(attn_1d)}")
            # Recalculate using fallback
            num_patches_per_side = int(np.sqrt(len(attn_1d)))
            if num_patches_per_side * num_patches_per_side == len(attn_1d):
                num_patches_h = num_patches_w = num_patches_per_side
                print(f"   📐 Recalculated using square grid: {num_patches_h}x{num_patches_w}")
    
    # Debug output: print final patch grid and num_patches
    print(f"   🔍 Final patch grid: ({num_patches_h}, {num_patches_w}), num_patches: {attn_1d.shape[0]}")
    
    # Debug: Check if this is a mostly black image
    frame_mean = np.array(frame).mean() if isinstance(frame, Image.Image) else frame.mean()
    if frame_mean < 20:  # Very dark frame (threshold: 20/255 ≈ 8% brightness)
        print(f"   ⚠️  Detected mostly black frame (mean intensity: {frame_mean:.1f})")
        print(f"   💡 For black images, attention distribution may appear unusual:")
        print(f"      - Uniform pixel values → similar patch embeddings → similar attention")
        print(f"      - Min-max normalization amplifies tiny attention differences")
        print(f"      - This can cause 'hot spots' in corners (like bottom-right)")
    
    # Print processor input size info
    PATCH_SIZE = 14  # Qwen2.5-VL default
    H_proc_est = num_patches_h * PATCH_SIZE
    W_proc_est = num_patches_w * PATCH_SIZE
    print(f"   📐 Estimated processor input size: {H_proc_est}x{W_proc_est} (patch_size={PATCH_SIZE})")
    print(f"   📐 Original frame size: {frame.shape[0]}x{frame.shape[1]}")
    aspect_ratio_frame = frame.shape[1] / frame.shape[0]
    aspect_ratio_proc = W_proc_est / H_proc_est
    if abs(aspect_ratio_frame - aspect_ratio_proc) > 0.1:
        print(f"   ⚠️  Aspect ratio mismatch! Frame: {aspect_ratio_frame:.2f}, Processor: {aspect_ratio_proc:.2f}")
        print(f"   💡 Processor will resize/pad, affecting patch-to-image alignment")
    
    output_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}_attention.png")
    # Use original_frame for visualization if we resized for processor
    # This ensures visualization matches the original video frame
    viz_frame = original_frame if args.processor_input_size is not None else frame
    # Pass num_patches_h and num_patches_w directly to ensure correct patch-to-image alignment
    visualize(viz_frame, attn_1d, output_path,
             use_overlay=args.use_overlay, overlay_threshold=args.overlay_threshold,
             num_patches_h=num_patches_h, num_patches_w=num_patches_w)
    
    print(f"   ✅ Saved: {os.path.basename(output_path)}")
    if args.use_overlay:
        overlay_path = output_path.replace('_attention.png', '_attention_overlay.png')
        print(f"   ✅ Saved overlay: {os.path.basename(overlay_path)}")
    
    print(f"\n✅ Complete! Results in {args.output_dir}")


if __name__ == "__main__":
    main()

