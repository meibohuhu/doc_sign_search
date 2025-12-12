#!/usr/bin/env python3
"""
InternVL2 Processing Script with Attention Visualization
Uses hook to intercept projector output and feed to InternLM2Model.forward() for attention extraction.

Usage:
    # Using HuggingFace model:
    python internlm2/internvl2_evaluation_attention.py \
        --model-path OpenGVLab/InternVL2_5-2B \
        --video-path /local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224/g0fgci8L_rc_18-8-rgb_front.mp4 \
        --out-dir /local1/mhu/sign_language_llm/internlm2 \
        --save-attention \
        --image-size 224 \
        --num-segments 2

    # Using local LoRA checkpoint:
    python internlm2/internvl2_evaluation_attention.py \
        --model-path /local1/mhu/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_how2sign_16fps_1130/checkpoint-2399 \
        --base-model-name OpenGVLab/InternVL2_5-2B \
        --video-path /local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224/g0fgci8L_rc_18-8-rgb_front.mp4 \
        --out-dir /local1/mhu/sign_language_llm/internlm2 \
        --save-attention \
        --image-size 224 \
        --num-segments 4
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")
os.environ["DISABLE_FLASH_ATTN"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from internlm2.modeling_internlm2 import InternLM2Model
from scripts.visualize_attention_simple_correct_final import (
    visualize,
    minmax_01
)


def build_transform(input_size=448):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def visualize_text_to_visual_heatmap(layer_attn, layer_idx, output_path, text_tokens=None, visual_tokens_per_frame=None):
    """
    Visualize text-to-visual attention heatmap for a single layer.
    
    Args:
        layer_attn: [text_len, visual_len] attention matrix
        layer_idx: Layer index
        output_path: Path to save the heatmap
        text_tokens: Optional list of text token strings for y-axis labels
        visual_tokens_per_frame: Optional number of visual tokens per frame for x-axis grouping
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Convert to numpy if tensor
    if isinstance(layer_attn, torch.Tensor):
        layer_attn = layer_attn.float().cpu().numpy()
    
    # Debug: print attention statistics
    print(f"      Layer {layer_idx} attention stats:")
    print(f"         Shape: {layer_attn.shape}")
    print(f"         Min: {layer_attn.min():.6f}, Max: {layer_attn.max():.6f}")
    print(f"         Mean: {layer_attn.mean():.6f}, Std: {layer_attn.std():.6f}")
    print(f"         Non-zero ratio: {(layer_attn > 1e-6).sum() / layer_attn.size:.4f}")
    
    # Check if attention values are too uniform
    if layer_attn.std() < 1e-6:
        print(f"      ⚠️  Warning: Attention values are too uniform (std={layer_attn.std():.6f})")
    
    # Normalize to [0, 1] with better handling of edge cases
    attn_min = layer_attn.min()
    attn_max = layer_attn.max()
    
    if attn_max - attn_min < 1e-6:
        # All values are the same, set to 0.5
        attn_norm = np.full_like(layer_attn, 0.5)
        print(f"      ⚠️  Warning: All attention values are the same, setting to 0.5")
    else:
        attn_norm = (layer_attn - attn_min) / (attn_max - attn_min)
    
    # Apply percentile-based normalization for better contrast (optional)
    # Use 1st and 99th percentile to reduce impact of outliers
    p1, p99 = np.percentile(layer_attn, [1, 99])
    if p99 - p1 > 1e-6:
        attn_norm_robust = np.clip((layer_attn - p1) / (p99 - p1), 0, 1)
        # Use robust normalization if it provides better contrast
        if attn_norm_robust.std() > attn_norm.std():
            attn_norm = attn_norm_robust
            print(f"      Using robust normalization (percentile 1-99)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, layer_attn.shape[1] // 10), max(8, layer_attn.shape[0] // 5)))
    
    # Plot heatmap
    im = ax.imshow(attn_norm, cmap='hot', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Set labels
    ax.set_xlabel('Visual Tokens', fontsize=12)
    ax.set_ylabel('Text Tokens', fontsize=12)
    ax.set_title(f'Text-to-Visual Attention Heatmap - Layer {layer_idx}', fontsize=14, fontweight='bold')
    
    # Add frame boundaries if visual_tokens_per_frame is provided
    if visual_tokens_per_frame is not None:
        num_frames = layer_attn.shape[1] // visual_tokens_per_frame
        for frame_idx in range(1, num_frames):
            x_pos = frame_idx * visual_tokens_per_frame
            ax.axvline(x=x_pos - 0.5, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
            ax.text(x_pos + visual_tokens_per_frame // 2, -layer_attn.shape[0] * 0.05, 
                   f'Frame {frame_idx}', ha='center', va='top', fontsize=8, color='cyan')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved layer {layer_idx} text-to-visual heatmap: {output_path}")


def visualize_per_layer_visual_attention(layer_attn_1d, layer_idx, output_path, num_frames=None, num_patches_h=None, num_patches_w=None):
    """
    Visualize per-layer attention to visual tokens (1D vector, same format as aggregated attention).
    
    Args:
        layer_attn_1d: [visual_len] 1D attention vector
        layer_idx: Layer index
        output_path: Path to save the heatmap
        num_frames: Number of frames
        num_patches_h: Number of patches in height dimension
        num_patches_w: Number of patches in width dimension
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy if tensor
    if isinstance(layer_attn_1d, torch.Tensor):
        layer_attn_1d = layer_attn_1d.float().cpu().numpy()
    
    # Ensure it's 1D
    if len(layer_attn_1d.shape) > 1:
        layer_attn_1d = layer_attn_1d.flatten()
    
    visual_len = len(layer_attn_1d)
    
    # Debug: print attention statistics
    print(f"      Layer {layer_idx} attention stats:")
    print(f"         Shape: {layer_attn_1d.shape}")
    print(f"         Min: {layer_attn_1d.min():.6f}, Max: {layer_attn_1d.max():.6f}")
    print(f"         Mean: {layer_attn_1d.mean():.6f}, Std: {layer_attn_1d.std():.6f}")
    print(f"         Non-zero ratio: {(layer_attn_1d > 1e-6).sum() / layer_attn_1d.size:.4f}")
    
    # Calculate patches per frame if not provided
    if num_frames is None or num_patches_h is None or num_patches_w is None:
        # Try to infer from visual_len
        # Assume square patches: visual_len = num_frames * num_patches_h * num_patches_w
        # Try common configurations
        if visual_len == 64:
            # 1 frame, 8x8 patches
            num_frames = 1
            num_patches_h = num_patches_w = 8
        else:
            # Try to find a reasonable factorization
            num_frames = 1
            patches_per_frame = visual_len
            num_patches_h = num_patches_w = int(np.sqrt(patches_per_frame))
            if num_patches_h * num_patches_w != patches_per_frame:
                # Not a perfect square, use approximate
                num_patches_h = int(np.sqrt(patches_per_frame))
                num_patches_w = patches_per_frame // num_patches_h
    
    print(f"   📐 num_frames: {num_frames}, num_patches_h: {num_patches_h}, num_patches_w: {num_patches_w}")
    # Create mosaic mask (same as aggregated attention visualization)
    mosaic_mask = create_mosaic_mask(layer_attn_1d, num_frames, num_patches_h, num_patches_w)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    im = ax.imshow(mosaic_mask, cmap='hot', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Set labels
    ax.set_xlabel('Patches (Width)', fontsize=12)
    ax.set_ylabel('Patches (Height)', fontsize=12)
    ax.set_title(f'Per-Layer Visual Attention - Layer {layer_idx}', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved layer {layer_idx} visual attention: {output_path}")


def visualize_frame_across_layers(per_layer_attention, frame_idx, frame_idx_in_segments, 
                                  patches_per_frame, num_patches_h, num_patches_w,
                                  video_file, output_dir):
    """
    Visualize a single frame across all layers (mosaic for each layer).
    
    Args:
        per_layer_attention: List of per-layer attention vectors [visual_len] for each layer
        frame_idx: Original frame index in video
        frame_idx_in_segments: Frame index in the sampled segments
        patches_per_frame: Number of patches per frame
        num_patches_h: Number of patches in height dimension
        num_patches_w: Number of patches in width dimension
        video_file: Video filename (without extension)
        output_dir: Output directory
    """
    import matplotlib.pyplot as plt
    
    # Extract frame attention from each layer
    frame_attentions = []
    valid_layers = []
    
    for layer_idx, layer_attn in enumerate(per_layer_attention):
        if layer_attn is None:
            continue
        
        # Convert to numpy if tensor
        if isinstance(layer_attn, torch.Tensor):
            layer_attn_np = layer_attn.float().cpu().numpy()
        else:
            layer_attn_np = layer_attn
        
        if layer_attn_np.size == 0:
            continue
        
        # Ensure it's 1D
        if len(layer_attn_np.shape) > 1:
            layer_attn_np = layer_attn_np.flatten()
        
        # Extract attention for this specific frame
        start_patch = frame_idx_in_segments * patches_per_frame
        end_patch = start_patch + patches_per_frame
        
        if end_patch > len(layer_attn_np):
            print(f"      ⚠️  Layer {layer_idx}: frame patches {start_patch}:{end_patch} exceed total {len(layer_attn_np)}")
            continue
        
        frame_attn = layer_attn_np[start_patch:end_patch]
        frame_attentions.append(frame_attn)
        valid_layers.append(layer_idx)
    
    if len(frame_attentions) == 0:
        print(f"   ⚠️  No valid layer attention found for frame {frame_idx}")
        return
    
    # Create figure with subplots for all layers
    num_layers = len(frame_attentions)
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for plot_idx, (layer_idx, frame_attn) in enumerate(zip(valid_layers, frame_attentions)):
        # Normalize attention for this frame
        frame_attn_norm = minmax_01(frame_attn)
        
        # Reshape to 2D grid for single frame: [num_patches_h, num_patches_w]
        frame_attn_2d = frame_attn_norm.reshape(num_patches_h, num_patches_w)
        
        # Plot
        ax = axes[plot_idx]
        im = ax.imshow(frame_attn_2d, cmap='hot', aspect='auto', interpolation='nearest')
        ax.set_title(f'Layer {layer_idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Patches (Width)', fontsize=8)
        ax.set_ylabel('Patches (Height)', fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Frame {frame_idx:04d} Attention Across All Layers\n{video_file}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    frame_output_dir = os.path.join(output_dir, 'per_frame_layers')
    os.makedirs(frame_output_dir, exist_ok=True)
    output_path = os.path.join(frame_output_dir, f"{video_file}_frame_{frame_idx:04d}_all_layers.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved frame {frame_idx} across all layers: {output_path}")


def create_mosaic_mask(attn_1d, num_frames, num_patches_h, num_patches_w):
    """
    Create a mosaic mask from 1D attention values.
    
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


def load_video(video_path, input_size=448, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    
    seg_size = float(max_frame) / num_segments
    frame_indices = np.array([
        int((seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        pixel_values = transform(img).unsqueeze(0)
        num_patches_list.append(1)
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, frame_indices


def _insert_visual_tokens(text_embeds, visual_embeds, visual_token_index):
    """
    Insert visual tokens into text embeddings at the positions specified by visual_token_index.
    
    Args:
        text_embeds: Text embeddings tensor [batch, text_len, hidden_dim]
        visual_embeds: Visual embeddings tensor [batch, visual_len, hidden_dim]
        visual_token_index: Tensor [start_idx, end_idx] indicating where to insert visual tokens
    
    Returns:
        Full inputs_embeds with visual tokens inserted [batch, total_len, hidden_dim]
    """
    visual_start = visual_token_index[0].item()
    visual_end = visual_token_index[1].item() + 1
    
    batch_size, text_len, hidden_dim = text_embeds.shape
    visual_len = visual_embeds.shape[1]
    visual_token_count = visual_end - visual_start
    
    # Calculate total length needed
    total_len = text_len + visual_token_count
    
    # Create full inputs_embeds
    full_inputs_embeds = torch.zeros(
        batch_size, total_len, hidden_dim,
        dtype=text_embeds.dtype, device=text_embeds.device
    )
    
    # Insert text embeddings before visual tokens
    full_inputs_embeds[:, :visual_start] = text_embeds[:, :visual_start]
    
    # Insert visual embeddings
    if visual_embeds.shape[1] >= visual_token_count:
        full_inputs_embeds[:, visual_start:visual_end] = visual_embeds[:, :visual_token_count]
    else:
        # If visual embeddings are shorter, pad or truncate
        full_inputs_embeds[:, visual_start:visual_start+visual_embeds.shape[1]] = visual_embeds
    
    # Insert remaining text embeddings after visual tokens
    if visual_start < text_len:
        remaining_text = text_embeds[:, visual_start:]
        remaining_len = remaining_text.shape[1]
        if remaining_len > 0:
            full_inputs_embeds[:, visual_end:visual_end+remaining_len] = remaining_text
    
    return full_inputs_embeds


def extract_attention_with_hook(model, tokenizer, pixel_values, num_patches_list, question):
    """Extract attention using hook to intercept projector output"""
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Prepare prompt
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    full_question = video_prefix + question
    
    # Tokenize
    inputs = tokenizer(full_question, return_tensors='pt').to(model.device)
    input_ids = inputs.input_ids
    
    # Find <image> token positions
    image_token_ids = tokenizer.encode('<image>', add_special_tokens=False)
    if len(image_token_ids) == 0:
        image_token_id = getattr(tokenizer, 'IMAGE_TOKEN_ID', getattr(model, 'IMAGE_TOKEN_ID', None))
    else:
        image_token_id = image_token_ids[0]
    
    if image_token_id is not None:
        image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        visual_token_start = image_positions[0].item() if len(image_positions) > 0 else input_ids.shape[1]
    else:
        visual_token_start = input_ids.shape[1]
    
    # Note: num_patches_list might not reflect actual visual tokens after projection
    # We'll update visual_token_index after we get the actual visual_embeds shape
    total_visual_patches = sum(num_patches_list)  # This is just an initial estimate
    visual_token_end = visual_token_start + total_visual_patches - 1
    visual_token_index = torch.tensor([visual_token_start, visual_token_end], dtype=torch.long, device=model.device)
    print(f"   📐 Initial visual_token_index (before correction): {visual_token_index.tolist()}")
    
    # Find language model
    language_model = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model
    
    # Detect model type
    model_type_name = type(language_model).__name__
    print(f"   🔍 Language model type: {model_type_name}")
    
    # Monkey patch InternLM2Model.forward to use local version (only for InternLM2Model)
    if 'InternLM2Model' in model_type_name or 'InternLM2' in model_type_name:
        import importlib.util
        modeling_path = os.path.join(current_dir, 'modeling_internlm2.py')
        spec = importlib.util.spec_from_file_location("internlm2.modeling_internlm2_local", modeling_path)
        local_modeling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_modeling)
        LocalInternLM2Model = local_modeling.InternLM2Model
        language_model.forward = lambda *args, **kwargs: LocalInternLM2Model.forward(language_model, *args, **kwargs)
        print(f"   ✅ Applied InternLM2Model monkey patch")
    else:
        print(f"   ℹ️  Skipping monkey patch (not InternLM2Model)")
    
    # Hook projector layer to intercept visual embeddings after projection
    intercepted_visual_embeds = []
    intercepted_text_embeds = [None]  # Use list to allow modification in closure
    
    # Find projector (mlp1 or projector)
    projector = (getattr(model, 'mlp1', None) or 
                 getattr(model, 'projector', None) or
                 getattr(getattr(model, 'model', None), 'mlp1', None) or
                 getattr(getattr(model, 'model', None), 'projector', None))
    
    if projector is None:
        raise ValueError("Could not find projector layer in model")
    
    def projector_hook(module, input, output):
        intercepted_visual_embeds.append(output.detach().clone())
        print(f"   ✅ Hooked projector output: {output.shape}")
    
    handle_projector = projector.register_forward_hook(projector_hook)
    
    # Hook text embeddings to get text part
    def text_emb_hook(module, input, output):
        intercepted_text_embeds[0] = output.detach().clone()
        print(f"   ✅ Hooked text embeddings: {output.shape}")
    
    # Find token embeddings layer (support different model architectures)
    # Try different possible attribute names
    token_embeddings = None
    if hasattr(language_model, 'tok_embeddings'):
        token_embeddings = language_model.tok_embeddings
    elif hasattr(language_model, 'embed_tokens'):
        token_embeddings = language_model.embed_tokens
    elif hasattr(language_model, 'get_input_embeddings'):
        token_embeddings = language_model.get_input_embeddings()
    else:
        # Try to find embeddings in model attributes
        for attr_name in ['tok_embeddings', 'embed_tokens', 'embeddings']:
            if hasattr(language_model, attr_name):
                token_embeddings = getattr(language_model, attr_name)
                break
    
    if token_embeddings is None:
        raise ValueError(
            f"Could not find token embeddings layer in language model. "
            f"Model type: {type(language_model).__name__}, "
            f"Available attributes: {[attr for attr in dir(language_model) if 'embed' in attr.lower() or 'token' in attr.lower()]}"
        )
    
    print(f"   🔍 Found token embeddings: {type(token_embeddings).__name__}")
    handle_text_emb = token_embeddings.register_forward_hook(text_emb_hook)
    
    # Store handles for cleanup
    handles = [handle_projector, handle_text_emb]
    
    # Call model.chat() to trigger hook (chat internally calls forward)
    # Note: chat() may fail due to shape mismatch, but hooks should have captured embeddings
    generation_config = {
        'max_new_tokens': 128,
        'do_sample': False,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }
    
    generated_text = None
    with torch.no_grad():
        try:
            response, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )
            generated_text = response.strip()
        except (RuntimeError, AttributeError, Exception) as e:
            # Chat failed, but hooks should have captured embeddings
            # This is expected when we manually construct inputs_embeds
            error_msg = str(e)[:200] if len(str(e)) > 200 else str(e)
            print(f"   ⚠️  Chat method failed (expected): {error_msg}")
            # Continue - we'll generate text manually if needed
            # The hooks should have already captured the embeddings we need
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Combine visual and text embeddings
    text_embeds = intercepted_text_embeds[0]
    visual_embeds = intercepted_visual_embeds[-1]
    
    if text_embeds is None or visual_embeds is None:
        raise ValueError("Failed to capture text or visual embeddings")
    
    print(f"   📐 Visual embeds shape: {visual_embeds.shape}")
    print(f"   📐 Text embeds shape: {text_embeds.shape}")
    
    # Reshape visual_embeds if needed and ALWAYS update visual_token_index based on actual shape
    batch_size = text_embeds.shape[0]
    visual_token_start = visual_token_index[0].item()  # Get the start position
    
    # Determine total_visual_tokens from actual visual_embeds shape
    if len(visual_embeds.shape) == 3:
        # visual_embeds shape: [num_frames, patches_per_frame, hidden_dim] or [batch, seq_len, hidden_dim]
        if visual_embeds.shape[0] != batch_size:
            # Reshape from [num_frames, patches_per_frame, hidden_dim] to [batch, total_visual_tokens, hidden_dim]
            num_frames, patches_per_frame, hidden_dim = visual_embeds.shape
            total_visual_tokens = num_frames * patches_per_frame
            visual_embeds = visual_embeds.view(1, total_visual_tokens, hidden_dim)
            print(f"   📐 Reshaped visual_embeds from 3D: {visual_embeds.shape}")
        else:
            # Already in [batch, seq_len, hidden_dim] format
            total_visual_tokens = visual_embeds.shape[1]
            print(f"   📐 Visual embeds already in [batch, seq_len, hidden_dim] format: {visual_embeds.shape}")
    elif len(visual_embeds.shape) == 2:
        # Flattened: [batch*seq_len, hidden_dim] -> [batch, seq_len, hidden_dim]
        visual_seq_len = visual_embeds.shape[0] // batch_size if visual_embeds.shape[0] > batch_size else visual_embeds.shape[0]
        visual_embeds = visual_embeds.view(batch_size, visual_seq_len, -1)
        total_visual_tokens = visual_seq_len
        print(f"   📐 Reshaped visual_embeds from 2D: {visual_embeds.shape}")
    else:
        # Should not happen, but handle it
        total_visual_tokens = visual_embeds.shape[-2] if len(visual_embeds.shape) >= 2 else 1
        print(f"   ⚠️  Unexpected visual_embeds shape: {visual_embeds.shape}, using {total_visual_tokens} tokens")
    
    # ALWAYS update visual_token_index based on actual visual_embeds shape
    visual_token_end = visual_token_start + total_visual_tokens - 1
    visual_token_index = torch.tensor([visual_token_start, visual_token_end], dtype=torch.long, device=model.device)
    print(f"   ✅ Corrected visual_token_index: {visual_token_index.tolist()} (total visual tokens: {total_visual_tokens})")
    
    # Insert visual tokens into text embeddings at visual_token_index positions
    inputs_embeds = _insert_visual_tokens(text_embeds, visual_embeds, visual_token_index)
    print(f"   ✅ Combined inputs_embeds shape: {inputs_embeds.shape}")
    
    # Debug: check inputs_embeds shape and visual_token_index
    print(f"   📐 inputs_embeds shape: {inputs_embeds.shape}")
    print(f"   📐 visual_token_index: {visual_token_index.tolist()}")
    print(f"   📐 input_ids length: {input_ids.shape[1]}")
    
    # Call language model forward with attention extraction
    # For models that support visual_token_index (InternLM2), use it
    # For other models (Phi3), extract attention manually
    is_internlm2 = 'InternLM2Model' in model_type_name or 'InternLM2' in model_type_name
    
    # Extract attention from single forward pass
    aggregated_attention = None
# ============ 在调用 forward 之前添加 ============
    print(f"   🔍 DEBUG - inputs_embeds has NaN: {torch.isnan(inputs_embeds).any().item()}")
    print(f"   🔍 DEBUG - inputs_embeds has Inf: {torch.isinf(inputs_embeds).any().item()}")
    print(f"   🔍 DEBUG - visual_embeds has NaN: {torch.isnan(visual_embeds).any().item()}")
    print(f"   🔍 DEBUG - text_embeds has NaN: {torch.isnan(text_embeds).any().item()}")
    
    # Extract attention from single forward pass
    aggregated_attention = None
    with torch.no_grad():
        if is_internlm2:
            outputs = language_model.forward(
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                visual_token_index=visual_token_index,
                return_dict=True
            )
            
            # ============ 检查 outputs ============
            print(f"   🔍 DEBUG - last_hidden_state has NaN: {torch.isnan(outputs.last_hidden_state).any().item()}")
            
            # 检查 attentions
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                print(f"   🔍 DEBUG - attentions is tuple of length: {len(outputs.attentions)}")
                if outputs.attentions[0] is not None:
                    print(f"   🔍 DEBUG - First layer attention has NaN: {torch.isnan(outputs.attentions[0]).any().item()}")
                    print(f"   🔍 DEBUG - Last layer attention has NaN: {torch.isnan(outputs.attentions[-1]).any().item()}")
            else:
                print(f"   🔍 DEBUG - attentions is None or empty!")
            
            # 检查 aggregated attention
            aggregated_attention = outputs.aggregated_viusal_token_attention
            if aggregated_attention is not None:
                print(f"   🔍 DEBUG - aggregated_attention type: {type(aggregated_attention)}")
                if isinstance(aggregated_attention, torch.Tensor):
                    print(f"   🔍 DEBUG - aggregated_attention has NaN: {torch.isnan(aggregated_attention).any().item()}")
                else:
                    print(f"   🔍 DEBUG - aggregated_attention value: {aggregated_attention}")
            
            # 获取每一层的 text-to-visual attention
            per_layer_text_to_visual_attention = getattr(outputs, 'per_layer_text_to_visual_attention', None)
            if per_layer_text_to_visual_attention is not None:
                print(f"   📊 Per-layer text-to-visual attention: {len(per_layer_text_to_visual_attention)} layers")
                for layer_idx, layer_attn in enumerate(per_layer_text_to_visual_attention):
                    if isinstance(layer_attn, torch.Tensor):
                        print(f"      Layer {layer_idx}: shape {layer_attn.shape}")
    
    if aggregated_attention is not None:
        print(f"   📊 Final aggregated attention shape: {aggregated_attention.shape}")
        print(f"   📊 Attention numel: {aggregated_attention.numel()}")
                # ============ 添加调试代码 ============
        attn_debug = aggregated_attention.float().cpu().numpy()
        print(f"   🔍 DEBUG Attention stats:")
        print(f"      Min: {attn_debug.min():.8f}")
        print(f"      Max: {attn_debug.max():.8f}")
        print(f"      Mean: {attn_debug.mean():.8f}")
        print(f"      Std: {attn_debug.std():.8f}")
        print(f"      Sum: {attn_debug.sum():.8f}")
        print(f"      First 10: {attn_debug[:10]}")
        # ============ 调试代码结束 ============

    # Set default text if generation failed
    if generated_text is None:
        generated_text = "[Generation failed, but attention extracted successfully]"
    
    # Return both aggregated and per-layer attention
    per_layer_attention = getattr(outputs, 'per_layer_text_to_visual_attention', None) if is_internlm2 else None
    
    return aggregated_attention, visual_token_index, generated_text, per_layer_attention


def process_video(model, tokenizer, video_path, question, args, output_dir, video_file):
    print(f"\n📹 Processing video: {video_file}")
    
    # Load video
    pixel_values, num_patches_list, frame_indices = load_video(
        video_path, 
        num_segments=args.num_segments,
        input_size=args.image_size
    )
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    print(f"   ✅ Loaded {len(num_patches_list)} frames")
    
    # Extract attention
    aggregated_attention, visual_token_idx, generated_text, per_layer_attention = extract_attention_with_hook(
        model, tokenizer, pixel_values, num_patches_list, question
    )
    
    # Visualize attention
    if aggregated_attention is not None and args.save_attention:
        print(f"   🎨 Generating attention visualizations...")
        
        attn_1d = aggregated_attention.float().cpu().numpy()
        
        if len(attn_1d) == 0 or attn_1d.size == 0:
            raise ValueError("Attention tensor is empty")
        
        attn_1d = minmax_01(attn_1d)
        
        # Calculate patches per frame based on actual attention length
        total_attention_tokens = len(attn_1d)
        num_frames = len(num_patches_list)
        patches_per_frame = total_attention_tokens // num_frames
        num_patches_h = num_patches_w = int(np.sqrt(patches_per_frame))
        
        print(f"   📐 Total attention tokens: {total_attention_tokens}")
        print(f"   📐 Number of frames: {num_frames}")
        print(f"   📐 Patches per frame: {patches_per_frame}")
        print(f"   📐 Grid size: {num_patches_h}x{num_patches_w}")
        
        # Generate mosaic mask
        if args.save_mosaic_mask:
            mosaic_mask = create_mosaic_mask(
                attn_1d, num_frames, num_patches_h, num_patches_w
            )
            # Convert to RGB image with colormap
            # Apply colormap (hot) to make it more visible
            mosaic_rgb = plt.cm.hot(mosaic_mask)[:, :, :3]  # Get RGB, ignore alpha
            mosaic_rgb = (mosaic_rgb * 255).astype(np.uint8)
            
            mosaic_path = os.path.join(output_dir, f"{video_file}_mosaic_mask.png")
            Image.fromarray(mosaic_rgb).save(mosaic_path)
            print(f"   ✅ Saved mosaic mask: {mosaic_path}")
        # Visualize specific frames
        # Use frame_indices from load_video if not specified by user
        frames_to_visualize = args.frame_indices if args.frame_indices else frame_indices.tolist()
        
        if frames_to_visualize:
            # Create directory for frame attention visualizations
            frame_attention_dir = os.path.join(output_dir, 'frame_attention')
            os.makedirs(frame_attention_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for frame_idx in frames_to_visualize:
                if frame_idx >= total_frames:
                    continue
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find corresponding frame in attention
                # If frame_idx is from load_video's frame_indices, directly use its index
                frame_idx_in_segments = None
                if args.frame_indices:
                    # User specified frames: need to find closest match
                    for i, orig_idx in enumerate(frame_indices):
                        if abs(orig_idx - frame_idx) < 5:
                            frame_idx_in_segments = i
                            break
                else:
                    # Using sampled frames: direct mapping
                    for i, orig_idx in enumerate(frame_indices):
                        if orig_idx == frame_idx:
                            frame_idx_in_segments = i
                            break
                
                if frame_idx_in_segments is None:
                    continue
                
                start_patch = frame_idx_in_segments * patches_per_frame
                end_patch = start_patch + patches_per_frame
                frame_attn = attn_1d[start_patch:end_patch]
                
                # Save both attention.png and attention_only.png in the same folder
                output_path = os.path.join(frame_attention_dir, f"{video_file}_frame_{frame_idx:04d}_attention.png")
                visualize(
                    frame_rgb, frame_attn, output_path,
                    use_overlay=args.use_overlay,
                    overlay_threshold=args.overlay_threshold,
                    num_patches_h=num_patches_h,
                    num_patches_w=num_patches_w
                )
                print(f"   ✅ Saved frame {frame_idx} attention: {output_path}")
            
            cap.release()
    
    # Visualize per-layer text-to-visual attention heatmaps
    if per_layer_attention is not None and args.save_attention:
        print(f"   🎨 Generating per-layer text-to-visual attention heatmaps...")
        print(f"   📊 Total layers: {len(per_layer_attention)}")
        
        # Calculate visual tokens per frame
        total_visual_tokens = visual_token_idx[1].item() - visual_token_idx[0].item() + 1
        num_frames = len(num_patches_list)
        visual_tokens_per_frame = total_visual_tokens // num_frames if num_frames > 0 else total_visual_tokens
        
        print(f"   📐 Visual tokens: {total_visual_tokens}, Frames: {num_frames}, Tokens per frame: {visual_tokens_per_frame}")
        print(f"   📝 Note: Each layer's heatmap shows text tokens (rows) attending to visual tokens (columns)")
        print(f"   📝 Visual tokens include all frames, with frame boundaries marked by cyan lines")
        
        # Create directory for per-layer heatmaps
        per_layer_dir = os.path.join(output_dir, 'per_layer_heatmaps')
        os.makedirs(per_layer_dir, exist_ok=True)
        
        for layer_idx, layer_attn in enumerate(per_layer_attention):
            if layer_attn is None:
                continue
            
            # Convert to numpy if tensor
            if isinstance(layer_attn, torch.Tensor):
                layer_attn_np = layer_attn.float().cpu().numpy()
            else:
                layer_attn_np = layer_attn
            
            # Skip if empty
            if layer_attn_np.size == 0:
                continue
            
            # Save individual layer heatmap
            layer_heatmap_path = os.path.join(per_layer_dir, f"{video_file}_layer_{layer_idx:02d}_visual_attention.png")
            # Calculate patches per frame (same as aggregated attention)
            patches_per_frame = visual_tokens_per_frame
            num_patches_h = num_patches_w = int(np.sqrt(patches_per_frame)) if patches_per_frame > 0 else 8
            
            visualize_per_layer_visual_attention(
                layer_attn_np, 
                layer_idx, 
                layer_heatmap_path,
                num_frames=num_frames,
                num_patches_h=num_patches_h,
                num_patches_w=num_patches_w
            )
        
        # Create a combined figure with all layers (subplot grid)
        if len(per_layer_attention) > 0:
            print(f"   🎨 Creating combined heatmap for all layers...")
            
            # Calculate grid size
            num_layers = len([attn for attn in per_layer_attention if attn is not None])
            cols = min(4, num_layers)
            rows = (num_layers + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
            if num_layers == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Calculate patches per frame for mosaic creation
            patches_per_frame = visual_tokens_per_frame
            num_patches_h = num_patches_w = int(np.sqrt(patches_per_frame)) if patches_per_frame > 0 else 8
            
            layer_idx = 0
            for idx, layer_attn in enumerate(per_layer_attention):
                if layer_attn is None:
                    continue
                
                if isinstance(layer_attn, torch.Tensor):
                    layer_attn_np = layer_attn.float().cpu().numpy()
                else:
                    layer_attn_np = layer_attn
                
                if layer_attn_np.size == 0:
                    continue
                
                # Ensure it's 1D
                if len(layer_attn_np.shape) > 1:
                    layer_attn_np = layer_attn_np.flatten()
                
                # Convert 1D attention to 2D mosaic (same as individual layer visualization)
                mosaic_mask = create_mosaic_mask(layer_attn_np, num_frames, num_patches_h, num_patches_w)
                
                # Plot
                ax = axes[layer_idx]
                im = ax.imshow(mosaic_mask, cmap='hot', aspect='auto', interpolation='nearest')
                ax.set_title(f'Layer {idx}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Patches (Width)', fontsize=8)
                ax.set_ylabel('Patches (Height)', fontsize=8)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                layer_idx += 1
            
            # Hide unused subplots
            for idx in range(layer_idx, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'Text-to-Visual Attention Heatmaps - All Layers\n{video_file}', 
                         fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            combined_path = os.path.join(output_dir, f"{video_file}_all_layers_text_to_visual.png")
            plt.savefig(combined_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved combined heatmap: {combined_path}")
        
        # Generate per-frame across all layers visualizations
        if per_layer_attention is not None and len(per_layer_attention) > 0:
            print(f"   🎨 Generating per-frame across all layers visualizations...")
            
            # Use frame_indices from load_video
            frames_to_visualize = args.frame_indices if args.frame_indices else frame_indices.tolist()
            
            if frames_to_visualize:
                patches_per_frame = visual_tokens_per_frame
                num_patches_h = num_patches_w = int(np.sqrt(patches_per_frame)) if patches_per_frame > 0 else 8
                
                for frame_idx in frames_to_visualize:
                    # Find corresponding frame index in segments
                    frame_idx_in_segments = None
                    if args.frame_indices:
                        # User specified frames: need to find closest match
                        for i, orig_idx in enumerate(frame_indices):
                            if abs(orig_idx - frame_idx) < 5:
                                frame_idx_in_segments = i
                                break
                    else:
                        # Using sampled frames: direct mapping
                        for i, orig_idx in enumerate(frame_indices):
                            if orig_idx == frame_idx:
                                frame_idx_in_segments = i
                                break
                    
                    if frame_idx_in_segments is None:
                        continue
                    
                    visualize_frame_across_layers(
                        per_layer_attention,
                        frame_idx,
                        frame_idx_in_segments,
                        patches_per_frame,
                        num_patches_h,
                        num_patches_w,
                        video_file,
                        output_dir
                    )
    
    return generated_text, aggregated_attention, per_layer_attention


def load_model_from_checkpoint(checkpoint_path, base_model_name=None):
    """
    Load model from checkpoint, supporting both full model and LoRA checkpoints.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        base_model_name: Base model name (required for LoRA checkpoints)
    
    Returns:
        model, tokenizer
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Check if it's a LoRA checkpoint
    # Method 1: Check for adapter_config.json
    adapter_config_path = os.path.join(checkpoint_path, 'adapter_config.json')
    is_lora = os.path.exists(adapter_config_path)
    
    # Method 2: If no adapter_config.json, check model weights for LoRA keys
    if not is_lora:
        try:
            # Check safetensors file
            model_safetensors = os.path.join(checkpoint_path, 'model.safetensors')
            if os.path.exists(model_safetensors):
                from safetensors import safe_open
                with safe_open(model_safetensors, framework='pt') as f:
                    keys = list(f.keys())
                    # Check if any key contains LoRA indicators
                    lora_indicators = ['base_layer', 'lora_A', 'lora_B']
                    has_lora_keys = any(any(indicator in k for indicator in lora_indicators) for k in keys)
                    if has_lora_keys:
                        is_lora = True
                        print(f"   🔍 Detected LoRA format in model.safetensors (found LoRA keys)")
            else:
                # Check pytorch_model.bin
                model_bin = os.path.join(checkpoint_path, 'pytorch_model.bin')
                if os.path.exists(model_bin):
                    state_dict = torch.load(model_bin, map_location='cpu')
                    keys = list(state_dict.keys())
                    lora_indicators = ['base_layer', 'lora_A', 'lora_B']
                    has_lora_keys = any(any(indicator in k for indicator in lora_indicators) for k in keys)
                    if has_lora_keys:
                        is_lora = True
                        print(f"   🔍 Detected LoRA format in pytorch_model.bin (found LoRA keys)")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not check for LoRA keys: {e}")
    
    if is_lora:
        print(f"   🔍 Detected LoRA checkpoint")
        if base_model_name is None:
            # Try to infer from checkpoint config
            if os.path.exists(adapter_config_path):
                try:
                    import json
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get('base_model_name_or_path', None)
                except:
                    pass
            
            # Try to infer from config.json
            if base_model_name is None:
                config_path = os.path.join(checkpoint_path, 'config.json')
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            # Check for base model name in various possible locations
                            base_model_name = (config.get('_name_or_path') or 
                                             config.get('base_model_name_or_path') or
                                             config.get('model_type'))
                    except:
                        pass
            
            if base_model_name is None:
                raise ValueError(
                    "LoRA checkpoint detected but base_model_name not provided. "
                    "Please specify --base-model-name (e.g., 'OpenGVLab/InternVL2.5-2B')"
                )
        
        print(f"   📦 Loading base model: {base_model_name}")
        
        # Try to read checkpoint config to get image_size
        checkpoint_config_path = os.path.join(checkpoint_path, 'config.json')
        checkpoint_image_size = None
        if os.path.exists(checkpoint_config_path):
            try:
                import json
                with open(checkpoint_config_path, 'r') as f:
                    checkpoint_config = json.load(f)
                    checkpoint_image_size = checkpoint_config.get('vision_tower_image_size') or checkpoint_config.get('image_size')
                    if checkpoint_image_size:
                        print(f"   📐 Checkpoint image_size: {checkpoint_image_size}")
            except:
                pass
        
        # Load base model first
        model = AutoModel.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation='eager'
        )
        
        # If checkpoint has different image_size, we'll skip vision weights that don't match
        if checkpoint_image_size and hasattr(model, 'config'):
            model_image_size = getattr(model.config, 'vision_tower_image_size', None) or getattr(model.config, 'image_size', None)
            if model_image_size != checkpoint_image_size:
                print(f"   ⚠️  Image size mismatch: base model={model_image_size}, checkpoint={checkpoint_image_size}")
                print(f"   📝 Will skip vision weights with shape mismatches")
        
        # Load non-LoRA trainable weights if available
        non_lora_path = os.path.join(checkpoint_path, 'non_lora_state_dict.bin')
        if os.path.exists(non_lora_path):
            print(f"   📦 Loading non-LoRA weights (vision tower + merger)...")
            non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
            # Clean up state dict keys (handle different prefix patterns)
            cleaned_state = {}
            for key, value in non_lora_trainables.items():
                # Remove common prefixes
                cleaned_key = key
                if cleaned_key.startswith('base_model.'):
                    cleaned_key = cleaned_key[11:]
                if cleaned_key.startswith('model.model.'):
                    cleaned_key = cleaned_key[6:]
                cleaned_state[cleaned_key] = value
            model.load_state_dict(cleaned_state, strict=False)
            print(f"   ✅ Loaded {len(cleaned_state)} non-LoRA weights")
        
        # Load LoRA weights
        print(f"   📦 Loading LoRA weights from {checkpoint_path}...")
        try:
            from peft import PeftModel
            # Try to load with PEFT if adapter_config.json exists
            if os.path.exists(adapter_config_path):
                model = PeftModel.from_pretrained(model, checkpoint_path)
                # Merge LoRA weights for inference
                print(f"   🔄 Merging LoRA weights...")
                model = model.merge_and_unload()
                print(f"   ✅ LoRA weights merged")
            else:
                # Manual LoRA loading: load from model.safetensors or pytorch_model.bin
                print(f"   📦 Loading LoRA weights manually (no adapter_config.json)...")
                model_safetensors = os.path.join(checkpoint_path, 'model.safetensors')
                model_bin = os.path.join(checkpoint_path, 'pytorch_model.bin')
                
                if os.path.exists(model_safetensors):
                    from safetensors import safe_open
                    state_dict = {}
                    with safe_open(model_safetensors, framework='pt') as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                elif os.path.exists(model_bin):
                    state_dict = torch.load(model_bin, map_location='cpu')
                else:
                    raise FileNotFoundError("Could not find model.safetensors or pytorch_model.bin")
                
                # Extract LoRA weights and merge manually
                print(f"   🔄 Merging LoRA weights manually...")
                merged_state_dict = {}
                lora_keys = [k for k in state_dict.keys() if 'base_layer' in k or 'lora_A' in k or 'lora_B' in k]
                normal_keys = [k for k in state_dict.keys() if k not in lora_keys]
                
                # First, load normal (non-LoRA) weights
                for key in normal_keys:
                    # Remove common prefixes
                    cleaned_key = key
                    if cleaned_key.startswith('base_model.'):
                        cleaned_key = cleaned_key[11:]
                    if cleaned_key.startswith('model.model.'):
                        cleaned_key = cleaned_key[6:]
                    merged_state_dict[cleaned_key] = state_dict[key]
                
                # Then, merge LoRA weights
                # Group LoRA weights by base layer
                lora_groups = {}
                for key in lora_keys:
                    # Extract base layer name (e.g., 'language_model.base_model.model.model.layers.0.attention.wqkv')
                    if 'base_layer' in key:
                        base_key = key.replace('.base_layer', '')
                        if base_key not in lora_groups:
                            lora_groups[base_key] = {}
                        lora_groups[base_key]['base'] = state_dict[key]
                    elif 'lora_A' in key:
                        base_key = key.replace('.lora_A.default', '')
                        if base_key not in lora_groups:
                            lora_groups[base_key] = {}
                        lora_groups[base_key]['lora_A'] = state_dict[key]
                    elif 'lora_B' in key:
                        base_key = key.replace('.lora_B.default', '')
                        if base_key not in lora_groups:
                            lora_groups[base_key] = {}
                        lora_groups[base_key]['lora_B'] = state_dict[key]
                
                # Merge LoRA: W = W_base + alpha * (lora_B @ lora_A) / r
                # Default alpha/r ratio is usually 1.0 (alpha=r), so W = W_base + lora_B @ lora_A
                merged_count = 0
                for base_key, lora_group in lora_groups.items():
                    if 'base' in lora_group and 'lora_A' in lora_group and 'lora_B' in lora_group:
                        base_weight = lora_group['base']
                        lora_A = lora_group['lora_A']
                        lora_B = lora_group['lora_B']
                        
                        # Ensure weights are on CPU and in float32 for merging
                        base_weight = base_weight.float().cpu()
                        lora_A = lora_A.float().cpu()
                        lora_B = lora_B.float().cpu()
                        
                        # Merge: W = W_base + lora_B @ lora_A
                        # lora_B shape: [out_features, r], lora_A shape: [r, in_features]
                        try:
                            merged_weight = base_weight + torch.matmul(lora_B, lora_A)
                            
                            # Remove common prefixes
                            cleaned_key = base_key
                            if cleaned_key.startswith('base_model.'):
                                cleaned_key = cleaned_key[11:]
                            if cleaned_key.startswith('model.model.'):
                                cleaned_key = cleaned_key[6:]
                            merged_state_dict[cleaned_key] = merged_weight
                            merged_count += 1
                        except Exception as e:
                            print(f"   ⚠️  Warning: Failed to merge LoRA for {base_key}: {e}")
                            print(f"      base_weight shape: {base_weight.shape}, lora_A shape: {lora_A.shape}, lora_B shape: {lora_B.shape}")
                
                print(f"   ✅ Successfully merged {merged_count}/{len(lora_groups)} LoRA layers")
                
                # Load merged weights with shape checking
                # Filter out weights that have shape mismatches
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                skipped_keys = []
                
                for key, value in merged_state_dict.items():
                    if key in model_state_dict:
                        model_shape = model_state_dict[key].shape
                        checkpoint_shape = value.shape
                        if model_shape == checkpoint_shape:
                            filtered_state_dict[key] = value
                        else:
                            skipped_keys.append(f"{key}: {checkpoint_shape} -> {model_shape}")
                    else:
                        # Key not in model, skip it
                        skipped_keys.append(f"{key}: (not in model)")
                
                if skipped_keys:
                    print(f"   ⚠️  Skipped {len(skipped_keys)} weights due to shape mismatches or missing keys")
                    if len(skipped_keys) <= 10:
                        for skip_key in skipped_keys:
                            print(f"      - {skip_key}")
                    else:
                        for skip_key in skipped_keys[:5]:
                            print(f"      - {skip_key}")
                        print(f"      ... and {len(skipped_keys) - 5} more")
                
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"   ✅ Loaded {len(filtered_state_dict)} matching weights (merged {merged_count} LoRA layers)")
        except ImportError:
            raise ImportError("peft library is required for LoRA checkpoints. Install with: pip install peft")
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to load LoRA weights: {e}")
            import traceback
            traceback.print_exc()
            print(f"   📦 Falling back to base model only")
        
        # Load tokenizer from checkpoint (prefer checkpoint, fallback to base)
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        print(f"   🔍 Detected full model checkpoint")
        # Load full model directly
        model = AutoModel.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation='eager'  # ← 添加这行
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    return model, tokenizer


def process_model(args):
    print(f"🤖 Loading model: {args.model_path}")
    
    # Check if it's a local checkpoint path
    if os.path.isdir(args.model_path) or (os.path.isfile(args.model_path) and args.model_path.endswith('.json')):
        # Local checkpoint path
        model, tokenizer = load_model_from_checkpoint(
            args.model_path,
            base_model_name=args.base_model_name
        )
    else:
        # HuggingFace model name
        model = AutoModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation='eager'  # ← 添加这行
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    model = model.eval().cuda()
    
    # Force eager attention to enable attention extraction
    # Flash Attention doesn't support output_attentions=True
# Force eager attention to enable attention extraction
# Force eager attention to enable attention extraction
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        lang_model = model.language_model.model
        if hasattr(lang_model, 'config') and hasattr(lang_model, 'layers'):
            
            first_attn_type = type(lang_model.layers[0].attention).__name__
            if 'Flash' in first_attn_type:
                print(f"   🔧 Rebuilding attention layers from '{first_attn_type}' to Eager...")
                lang_model.config.attn_implementation = 'eager'
                
                from internlm2.modeling_internlm2 import InternLM2Attention
                
                device = next(lang_model.parameters()).device
                dtype = next(lang_model.parameters()).dtype
                print(f"   📐 Model device: {device}, dtype: {dtype}")
                
                for idx, layer in enumerate(lang_model.layers):
                    old_attn = layer.attention
                    
                    new_attn = InternLM2Attention(lang_model.config)
                    
                    # 手动复制所有权重
                    new_attn.wqkv.weight.data.copy_(old_attn.wqkv.weight.data)
                    if old_attn.wqkv.bias is not None:
                        new_attn.wqkv.bias.data.copy_(old_attn.wqkv.bias.data)
                    
                    new_attn.wo.weight.data.copy_(old_attn.wo.weight.data)
                    if old_attn.wo.bias is not None:
                        new_attn.wo.bias.data.copy_(old_attn.wo.bias.data)
                    
                    # 关键：手动复制 rotary embedding 的 inv_freq
                    if hasattr(old_attn, 'rotary_emb') and hasattr(new_attn, 'rotary_emb'):
                        # 复制 inv_freq buffer
                        if hasattr(old_attn.rotary_emb, 'inv_freq'):
                            new_attn.rotary_emb.register_buffer(
                                'inv_freq', 
                                old_attn.rotary_emb.inv_freq.clone()
                            )
                        
                        # 复制其他可能的 buffer
                        if hasattr(old_attn.rotary_emb, 'cos_cached'):
                            new_attn.rotary_emb.register_buffer(
                                'cos_cached',
                                old_attn.rotary_emb.cos_cached.clone()
                            )
                        if hasattr(old_attn.rotary_emb, 'sin_cached'):
                            new_attn.rotary_emb.register_buffer(
                                'sin_cached',
                                old_attn.rotary_emb.sin_cached.clone()
                            )
                    
                    # 移动到正确的 device 和 dtype
                    new_attn = new_attn.to(device=device, dtype=dtype)
                    
                    layer.attention = new_attn
                
                print(f"   ✅ Rebuilt {len(lang_model.layers)} layers with eager attention")
                
                # 验证 rotary embedding
                test_rotary = lang_model.layers[0].attention.rotary_emb
                print(f"   🔍 Verification - inv_freq shape: {test_rotary.inv_freq.shape}")
                print(f"   🔍 Verification - inv_freq dtype: {test_rotary.inv_freq.dtype}")
                print(f"   🔍 Verification - inv_freq device: {test_rotary.inv_freq.device}")
                print(f"   🔍 Verification - inv_freq has NaN: {torch.isnan(test_rotary.inv_freq).any().item()}")
            else:
                print(f"   ✅ Already using Eager Attention: {first_attn_type}")
    
    # Resolve image size
    image_size = args.image_size
    if image_size is None:
            image_size = getattr(model.config, 'vision_tower_image_size', 448)
    args.image_size = image_size
    
    # Fixed question
    # question = "Translate the American Sign Language in this video to English. Pay close attention to the person's hand movement and facial expressions."
    question = "Translate the American Sign Language in this video to English."
    # question = "How many people are in this video?"

    # Prepare video paths
    video_paths = []
    if os.path.isfile(args.video_path):
        video_paths = [args.video_path]
    elif os.path.isdir(args.video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        for root, dirs, files in os.walk(args.video_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(root, file))
    else:
        print(f"   ⚠️  Video path not found: {args.video_path}")
        return
    
    if args.max_samples:
        video_paths = video_paths[:args.max_samples]
    
    print(f"   Total videos: {len(video_paths)}\n")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    attention_dir = os.path.join(args.out_dir, 'attention_visualizations')
    if args.save_attention:
        os.makedirs(attention_dir, exist_ok=True)
    
    results = []
    
    for idx, video_path in enumerate(tqdm(video_paths, desc="Processing"), 1):
        video_name = os.path.basename(video_path)
        generated_text, aggregated_attention, per_layer_attention = process_video(
                model, tokenizer, video_path, question, args,
            attention_dir, video_name
            )
            
        results.append({
        "video": video_name,
        "video_path": video_path,
            "model_output": generated_text,
            "has_attention": aggregated_attention is not None,
            "has_per_layer_attention": per_layer_attention is not None and len(per_layer_attention) > 0,
            "num_layers": len(per_layer_attention) if per_layer_attention is not None else 0
        })
        
        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(video_paths)}] {video_name}")
        print(f"Prediction:   {generated_text}")
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"internvl2_results_{timestamp}.json"
    output_path = os.path.join(args.out_dir, output_file)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint (local directory or HuggingFace model name)")
    parser.add_argument("--base-model-name", type=str, default=None,
                        help="Base model name for LoRA checkpoints (e.g., 'OpenGVLab/InternVL2.5-2B'). "
                             "If not provided, will try to infer from adapter_config.json")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=8)
    
    parser.add_argument("--save-attention", action="store_true")
    parser.add_argument("--frame-indices", type=int, nargs='+', default=None,
                        help="Frame indices to visualize. If not specified, uses frames sampled by load_video")
    parser.add_argument("--save-mosaic-mask", action="store_true")
    parser.add_argument("--temporal-patch-size", type=int, default=2)
    parser.add_argument("--use-overlay", action="store_true")
    parser.add_argument("--overlay-threshold", type=float, default=0.15)
    
    args = parser.parse_args()
    process_model(args)


if __name__ == "__main__":
    main()
