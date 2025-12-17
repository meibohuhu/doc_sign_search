#!/usr/bin/env python3
"""
InternVL Evaluation Script for how2sign
Based on qwen2vl_evaluation_how2sign_claude.py evaluation approach
"""

import os
import sys
import json
import torch
import warnings
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional, Iterable
import numpy as np

os.environ["DISABLE_FLASH_ATTN"] = "1"
warnings.filterwarnings("ignore")

# Add InternVL paths - auto-detect project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
internvl_chat_path = os.path.join(project_root, 'InternVL/internvl_chat')
internvl_path = os.path.join(project_root, 'InternVL')
if os.path.exists(internvl_chat_path):
    sys.path.insert(0, internvl_chat_path)
if os.path.exists(internvl_path):
    sys.path.insert(0, internvl_path)
# Fallback to hardcoded paths if auto-detection fails
if '/code/doc_sign_search/InternVL/internvl_chat' not in sys.path:
    sys.path.insert(0, '/code/doc_sign_search/InternVL/internvl_chat')
if '/code/doc_sign_search/InternVL' not in sys.path:
    sys.path.insert(0, '/code/doc_sign_search/InternVL')

from internvl.model.internvl_chat import InternVLChatModel, InternVLChatConfig
# CRITICAL: Import querygate_withbias version (uses q_proj/k_proj/v_proj format)
try:
    # Priority 1: Use modeling_internlm2_querygate_withbias.py (used by internvl_chat_finetune_querygate_withbias.py)
    from internvl.model.internlm2.modeling_internlm2_querygate_withbias import InternLM2ForCausalLM as InternLM2ForCausalLM_QueryGateWithBias
    import internvl.model.internvl_chat.modeling_internvl_chat as modeling_internvl_chat_module
    modeling_internvl_chat_module.InternLM2ForCausalLM = InternLM2ForCausalLM_QueryGateWithBias
    print("✅ Patched modeling_internvl_chat to use InternLM2ForCausalLM from modeling_internlm2_querygate_withbias")
    USE_GATE_VERSION = True
    USE_WQKV_VERSION = False  # querygate_withbias uses q_proj/k_proj/v_proj format
    print("   🔍 Detected: modeling_internlm2_querygate_withbias uses q_proj/k_proj/v_proj format")
except ImportError:
    try:
        # Fallback 1: Try modeling_internlm2_gate.py (may use wqkv or q_proj/k_proj/v_proj)
        from internvl.model.internlm2.modeling_internlm2_gate import InternLM2ForCausalLM as InternLM2ForCausalLM_Gate
        import internvl.model.internvl_chat.modeling_internvl_chat as modeling_internvl_chat_module
        modeling_internvl_chat_module.InternLM2ForCausalLM = InternLM2ForCausalLM_Gate
        print("✅ Patched modeling_internvl_chat to use InternLM2ForCausalLM from modeling_internlm2_gate")
        
        # Detect whether the gate version uses wqkv or q_proj/k_proj/v_proj
        from internvl.model.internlm2.modeling_internlm2_gate import InternLM2Attention, InternLM2FlashAttention2
        from internvl.model.internlm2.configuration_internlm2 import InternLM2Config
        dummy_config = InternLM2Config(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            headwise_attn_output_gate=True,
        )
        dummy_attn = InternLM2Attention(dummy_config)
        
        if hasattr(dummy_attn, 'wqkv') and not hasattr(dummy_attn, 'q_proj'):
            USE_WQKV_VERSION = True
            print("   🔍 Detected: modeling_internlm2_gate uses wqkv format")
        elif hasattr(dummy_attn, 'q_proj') and hasattr(dummy_attn, 'k_proj') and hasattr(dummy_attn, 'v_proj'):
            USE_WQKV_VERSION = False
            print("   🔍 Detected: modeling_internlm2_gate uses q_proj/k_proj/v_proj format")
        else:
            USE_WQKV_VERSION = hasattr(dummy_attn, 'wqkv')
            print(f"   🔍 Auto-detected: USE_WQKV_VERSION={USE_WQKV_VERSION}")
        
        USE_GATE_VERSION = True
        del dummy_attn, dummy_config
    except ImportError:
        try:
            # Fallback 2: Try wqkv gate version (used by internvl_chat_finetune_wqkv.py)
            from internvl.model.internlm2.modeling_internlm2_gate_wqkv import InternLM2ForCausalLM as InternLM2ForCausalLM_GateWQKV
            import internvl.model.internvl_chat.modeling_internvl_chat as modeling_internvl_chat_module
            modeling_internvl_chat_module.InternLM2ForCausalLM = InternLM2ForCausalLM_GateWQKV
            print("✅ Patched modeling_internvl_chat to use InternLM2ForCausalLM from modeling_internlm2_gate_wqkv")
            USE_GATE_VERSION = True
            USE_WQKV_VERSION = True
        except ImportError:
            from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
            print("⚠️  Gate version not available, using standard InternLM2ForCausalLM")
            USE_GATE_VERSION = False
            USE_WQKV_VERSION = False

from transformers import AutoTokenizer, AutoConfig
from internvl.train.dataset import build_transform
from PIL import Image
import cv2
from decord import VideoReader, cpu

def _compute_frame_indices(
    sample: str,
    vlen: int,
    input_fps: float,
    max_num_frames: int,
    min_num_frames: int,
    start_index: int = 0,
) -> List[int]:
    """
    Compute frame indices based on sampling strategy (same as training code).
    
    Args:
        sample: Sampling strategy. Supports:
            - 'fpsX.X': FPS-based sampling (e.g., 'fps2.0', 'fps12.0')
            - 'random_start_every2': Random start frame, then sample every 2 frames
            - Other strategies can be added here
        vlen: Video length (number of frames in the clip)
        input_fps: Original video FPS
        max_num_frames: Maximum number of frames to sample
        min_num_frames: Minimum number of frames to sample
        start_index: Starting frame index offset (for clip parameter)
    
    Returns:
        List of frame indices (relative to start_index)
    """
    frame_indices: List[int] = []
    
    if 'fps' in sample:
        # FPS-based sampling (same as InternVL's get_frame_indices)
        # Format: 'fpsX.X' where X.X is the target FPS
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps if input_fps > 0 else 0
        delta = 1 / output_fps  # gap between frames
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int).tolist()
        frame_indices = [e for e in frame_indices if e < vlen]
        
        # Apply max_num_frames limit: uniformly drop some to maintain even distribution
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            indices_to_keep = np.linspace(0, len(frame_indices) - 1, max_num_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices_to_keep]
    
    elif sample == 'random_start_every2':
        # Randomly choose a start frame, then sample every 2 frames
        if vlen > 0:
            # Randomly choose starting position from [0, vlen-1]
            start_offset = np.random.randint(0, max(1, vlen))
            # Sample every 2 frames starting from start_offset
            frame_indices = list(range(start_offset, vlen, 2))
            
            # If we have more frames than max_num_frames, truncate from the end
            if len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
    
    else:
        # Default: Sample every other frame (take 1 frame out of every 2 frames)
        # Randomly choose starting position (0 or 1) for better diversity
        start_offset = np.random.randint(0, 2)  # 0 or 1
        frame_indices = list(range(start_offset, vlen, 2))  # [0,2,4,...] or [1,3,5,...]
        
        # If we have more frames than max_num_frames, uniformly drop some to maintain even distribution
        if len(frame_indices) > max_num_frames:
            indices_to_keep = np.linspace(0, len(frame_indices) - 1, max_num_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices_to_keep]
    
    # Ensure we have at least min_num_frames if video has enough frames
    if len(frame_indices) < min_num_frames and vlen >= min_num_frames:
        # If we don't have enough, uniformly sample additional frames from the remaining frames
        remaining_indices = [i for i in range(vlen) if i not in frame_indices]
        needed = min_num_frames - len(frame_indices)
        if len(remaining_indices) >= needed:
            # Uniformly sample from remaining frames
            if len(remaining_indices) == needed:
                additional = remaining_indices
            else:
                indices_to_add = np.linspace(0, len(remaining_indices) - 1, needed, dtype=int)
                additional = [remaining_indices[i] for i in indices_to_add]
            frame_indices = sorted(frame_indices + additional)
    
    # Adjust indices if start_index is specified (for clip parameter)
    if start_index > 0:
        frame_indices = [f + start_index for f in frame_indices]
    
    # Remove duplicates and sort
    frame_indices = sorted(list(set(frame_indices)))
    return frame_indices


def _load_video_locally(
    video_path: str,
    max_num_frames: int,
    min_num_frames: int,
    sample: str = 'rand',
    clip: Optional[Iterable[int]] = None,
) -> List[Image.Image]:
    """
    Load video frames locally using decord or OpenCV (same as training code).
    Supports multiple sampling strategies including fps-based sampling.
    """
    load_errors: List[str] = []

    # Try decord first
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / float(fps) if fps > 0 else 0
        
        if clip and len(clip) == 2:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps) if fps > 0 else total_frames
            start_index = int(start * fps) if fps > 0 else 0
        else:
            vlen = total_frames
            start_index = 0

        frame_indices = _compute_frame_indices(
            sample=sample,
            vlen=vlen,
            input_fps=fps,
            max_num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            start_index=start_index,
        )
        
        frame_indices = [min(max(int(idx), 0), total_frames - 1) for idx in frame_indices]
        
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices
        
        # Print sampled frame indices for debugging
        print(f"   📊 Sampled {len(frame_indices)} frames: {frame_indices} (video total: {total_frames}, fps: {fps:.2f})")

        frames = vr.get_batch(frame_indices).asnumpy()
        images = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return images
    except Exception as exc:
        load_errors.append(f'decord: {exc}')

    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('cv2.VideoCapture failed to open file')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        if total_frames == 0:
            raise RuntimeError('cv2 reported zero frames')

        if clip and len(clip) == 2:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps)
            start_index = int(start * fps)
        else:
            vlen = total_frames
            start_index = 0

        frame_indices = _compute_frame_indices(
            sample=sample,
            vlen=vlen,
            input_fps=fps,
            max_num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            start_index=start_index,
        )
        
        frame_indices = [min(max(int(idx), 0), total_frames - 1) for idx in frame_indices]
        
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices
        
        # Print sampled frame indices for debugging
        print(f"   📊 Sampled {len(frame_indices)} frames: {frame_indices} (video total: {total_frames}, fps: {fps:.2f})")

        frames: List[Image.Image] = []
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

        if len(frames) == 0:
            raise RuntimeError('cv2 did not return any frames')

        return frames
    except Exception as exc:
        load_errors.append(f'opencv: {exc}')

    error_msgs = '; '.join(load_errors) or 'unknown'
    raise RuntimeError(f'Failed to load video locally ({error_msgs})')

def extract_and_export_frames(
    video_path: str,
    output_folder: str,
    max_num_frames: int,
    min_num_frames: int,
    sample: str = 'rand',
    clip: Optional[Iterable[int]] = None,
    video_name: Optional[str] = None,
) -> str:
    """
    Extract frames from video and export them to a folder.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted frames
        max_num_frames: Maximum number of frames to extract
        min_num_frames: Minimum number of frames to extract
        sample: Sampling method (same as _load_video_locally)
        clip: Optional clip parameter (start, end) in seconds
        video_name: Optional video name for folder naming (if None, uses video filename)
    
    Returns:
        Path to the folder containing extracted frames
    """
    # Generate output folder name
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Sanitize video_name for filesystem (remove invalid characters)
    video_name = "".join(c for c in video_name if c.isalnum() or c in ('-', '_', '.'))
    
    frame_output_dir = os.path.join(output_folder, video_name)
    os.makedirs(frame_output_dir, exist_ok=True)
    
    # Load video frames
    try:
        image_list = _load_video_locally(
            video_path,
            max_num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            sample=sample,
            clip=clip
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load video frames: {e}")
    
    # Save frames to folder
    for idx, image in enumerate(image_list):
        frame_filename = f"frame_{idx:04d}.jpg"
        frame_path = os.path.join(frame_output_dir, frame_filename)
        image.save(frame_path, quality=95)
    
    return frame_output_dir

def load_base_model(base_model_name="OpenGVLab/InternVL2_5-2B"):
    """
    Load InternVL base model (without checkpoint)
    """
    print("🚀 Loading InternVL base model...")
    print(f"   Model: {base_model_name}")
    
    # Step 1: Load base model config
    print("\n1️⃣ Loading base model config...")
    config = InternVLChatConfig.from_pretrained(base_model_name, trust_remote_code=True)
    print("   ✅ Config loaded")
    
    # Step 2: Load model
    print(f"\n2️⃣ Loading base model...")
    model = InternVLChatModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    print("   ✅ Model loaded from base model")
    
    # Step 2.5: Update model config to match training settings
    # Training uses: force_image_size=224, down_sample_ratio=0.5
    # This ensures num_image_token matches training (64 for 224x224 with ratio 0.5)
    force_image_size = 224
    down_sample_ratio = 0.5
    patch_size = model.patch_size
    
    # Resize vision model if needed
    if hasattr(model.config, 'force_image_size') and model.config.force_image_size != force_image_size:
        print(f"   🔧 Resizing vision model from {model.config.vision_config.image_size} to {force_image_size}")
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=force_image_size,
            patch_size=patch_size
        )
        model.config.vision_config.image_size = force_image_size
    
    # Update config and recalculate num_image_token
    model.config.force_image_size = force_image_size
    model.config.downsample_ratio = down_sample_ratio
    model.downsample_ratio = down_sample_ratio
    model.num_image_token = int((force_image_size // patch_size) ** 2 * (down_sample_ratio ** 2))
    print(f"   🔧 Updated config: force_image_size={force_image_size}, down_sample_ratio={down_sample_ratio}")
    print(f"   🔧 num_image_token={model.num_image_token}")
    
    # Step 3: Load tokenizer
    print(f"\n3️⃣ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    # Set model_max_length to match training (12288 for InternVL2.5-2B)
    # This ensures evaluation uses the same sequence length limit as training
    tokenizer.model_max_length = 12288
    print(f"   ✅ Tokenizer loaded (model_max_length: {tokenizer.model_max_length})")
    
    model.eval()
    
    # Disable gradients for all parameters to save memory
    for param in model.parameters():
        param.requires_grad = False
    
    # CRITICAL FIX: Explicitly move model to GPU to ensure all components are on CUDA
    # device_map="auto" may leave some components on CPU, causing "Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)" error
    print(f"\n4️⃣ Moving model to GPU...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
        # Explicitly move vision_model to GPU (this is often the problematic component)
        if hasattr(model, 'vision_model'):
            model.vision_model = model.vision_model.to(device)
        print(f"   ✅ Model moved to {device}")
    else:
        print(f"   ⚠️  CUDA not available, model remains on CPU")
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE BASE MODEL LOADED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    
    return model, tokenizer

def _load_and_convert_lora_weights(model, checkpoint_weights, checkpoint_path):
    """
    Load LoRA weights from checkpoint and merge into q_proj/k_proj/v_proj/wo for gated attention.
    Supports both q_proj/k_proj/v_proj format (internvl_chat_finetune_gate.py) and wqkv format (internvl_chat_finetune_wqkv.py).
    
    Args:
        model: InternVLChatModel with gate version
        checkpoint_weights: Dictionary of weights from model.safetensors
        checkpoint_path: Path to checkpoint directory
    """
    # Detect which version based on global USE_WQKV_VERSION flag
    try:
        # Priority 1: Try querygate_withbias version (uses q_proj/k_proj/v_proj)
        from internvl.model.internlm2.modeling_internlm2_querygate_withbias import InternLM2Attention as InternLM2Attention_QueryGateWithBias, InternLM2FlashAttention2 as InternLM2FlashAttention2_QueryGateWithBias
        InternLM2Attention_Class = InternLM2Attention_QueryGateWithBias
        InternLM2FlashAttention2_Class = InternLM2FlashAttention2_QueryGateWithBias
    except ImportError:
        try:
            # Fallback 1: Try gate version (may use wqkv or q_proj/k_proj/v_proj)
            from internvl.model.internlm2.modeling_internlm2_gate import InternLM2Attention, InternLM2FlashAttention2
            from internvl.model.internlm2.modeling_internlm2_gate_wqkv import InternLM2Attention as InternLM2AttentionWQKV, InternLM2FlashAttention2 as InternLM2FlashAttention2WQKV
            # Use global flag to determine which version
            if USE_WQKV_VERSION:
                InternLM2Attention_Class = InternLM2AttentionWQKV
                InternLM2FlashAttention2_Class = InternLM2FlashAttention2WQKV
            else:
                InternLM2Attention_Class = InternLM2Attention
                InternLM2FlashAttention2_Class = InternLM2FlashAttention2
        except ImportError:
            try:
                from internvl.model.internlm2.modeling_internlm2_gate import InternLM2Attention, InternLM2FlashAttention2
                InternLM2Attention_Class = InternLM2Attention
                InternLM2FlashAttention2_Class = InternLM2FlashAttention2
            except ImportError:
                print("   ⚠️  Gate version not available")
                return
    
    import contextlib
    try:
        from deepspeed import zero
        gather_context = zero.GatheredParameters
    except ImportError:
        gather_context = contextlib.nullcontext
    
    loaded_wqkv_count = 0
    loaded_wo_count = 0
    
    # Get the model that contains layers
    # Path options:
    # 1. model.language_model (PeftModel) -> .model or .base_model (InternLM2ForCausalLM) -> .model (InternLM2Model) -> .layers
    # 2. model.language_model (InternLM2ForCausalLM) -> .model (InternLM2Model) -> .layers
    try:
        # Step 1: Get InternLM2ForCausalLM from PeftModel (if wrapped)
        if hasattr(model.language_model, 'model'):
            causal_lm_model = getattr(model.language_model, 'model')
        elif hasattr(model.language_model, 'base_model'):
            # PeftModel also has base_model as alias for model
            causal_lm_model = getattr(model.language_model, 'base_model')
        else:
            # Fallback: maybe language_model is already InternLM2ForCausalLM?
            causal_lm_model = model.language_model
        
        # Step 2: Get InternLM2Model from InternLM2ForCausalLM
        if hasattr(causal_lm_model, 'model'):
            layers_model = getattr(causal_lm_model, 'model')
            if hasattr(layers_model, 'layers'):
                layers = layers_model.layers
            else:
                raise AttributeError(f"layers_model ({type(layers_model)}) does not have 'layers' attribute")
        else:
            # Fallback: maybe causal_lm_model is already InternLM2Model?
            if hasattr(causal_lm_model, 'layers'):
                layers = causal_lm_model.layers
            else:
                raise AttributeError(f"causal_lm_model ({type(causal_lm_model)}) does not have 'model' or 'layers' attribute")
    except AttributeError as e:
        print(f"   ❌ Error accessing layers: {e}")
        print(f"   🔍 Debug: model.language_model type: {type(model.language_model)}")
        print(f"   🔍 Debug: hasattr(model.language_model, 'model'): {hasattr(model.language_model, 'model')}")
        print(f"   🔍 Debug: hasattr(model.language_model, 'base_model'): {hasattr(model.language_model, 'base_model')}")
        if hasattr(model.language_model, 'model'):
            causal_lm_model = getattr(model.language_model, 'model')
            print(f"   🔍 Debug: causal_lm_model type: {type(causal_lm_model)}")
            print(f"   🔍 Debug: hasattr(causal_lm_model, 'model'): {hasattr(causal_lm_model, 'model')}")
            if hasattr(causal_lm_model, 'model'):
                layers_model = getattr(causal_lm_model, 'model')
                print(f"   🔍 Debug: layers_model type: {type(layers_model)}")
                print(f"   🔍 Debug: hasattr(layers_model, 'layers'): {hasattr(layers_model, 'layers')}")
        return
    
    # Process each layer
    loaded_q_proj_count = 0
    loaded_k_proj_count = 0
    loaded_v_proj_count = 0
    loaded_wqkv_count = 0
    loaded_wo_count = 0
    
    for i, layer in enumerate(layers):
        attn = layer.attention
        if not isinstance(attn, (InternLM2Attention_Class, InternLM2FlashAttention2_Class)):
            continue
        
        # Process wqkv weights (for wqkv version checkpoints) - check this first
        if USE_WQKV_VERSION and hasattr(attn, 'wqkv'):
            base_key_prefix = f"language_model.base_model.model.model.layers.{i}.attention.wqkv"
            base_layer_key = f"{base_key_prefix}.base_layer.weight"
            lora_a_key = f"{base_key_prefix}.lora_A.default.weight"
            lora_b_key = f"{base_key_prefix}.lora_B.default.weight"
            
            # Also check for q_proj/k_proj/v_proj format (in case checkpoint is q_proj/k_proj/v_proj format)
            q_proj_base_key = f"language_model.base_model.model.model.layers.{i}.attention.q_proj.base_layer.weight"
            k_proj_base_key = f"language_model.base_model.model.model.layers.{i}.attention.k_proj.base_layer.weight"
            v_proj_base_key = f"language_model.base_model.model.model.layers.{i}.attention.v_proj.base_layer.weight"
            
            if base_layer_key in checkpoint_weights:
                try:
                    # Load base layer weight
                    base_wqkv = checkpoint_weights[base_layer_key].to(dtype=torch.bfloat16)
                    
                    # Load and merge LoRA weights if they exist
                    if lora_a_key in checkpoint_weights and lora_b_key in checkpoint_weights:
                        lora_a = checkpoint_weights[lora_a_key].to(dtype=torch.bfloat16)
                        lora_b = checkpoint_weights[lora_b_key].to(dtype=torch.bfloat16)
                        # LoRA merge: base + lora_B @ lora_A (scaled by alpha/rank)
                        # Default LoRA scaling: alpha/rank = 1.0 (assuming alpha=rank, as in training script)
                        lora_delta = torch.matmul(lora_b, lora_a)
                        merged_wqkv = base_wqkv + lora_delta
                    else:
                        merged_wqkv = base_wqkv
                    
                    # Check for NaN/Inf
                    if torch.isnan(merged_wqkv).any() or torch.isinf(merged_wqkv).any():
                        print(f"   ⚠️  Layer {i}: Merged wqkv contains NaN/Inf! Skipping...")
                    else:
                        # Verify shape matches wqkv (base_qkv_dim + gate_dim)
                        expected_shape = attn.wqkv.weight.shape
                        if merged_wqkv.shape == expected_shape:
                            # Load directly into wqkv
                            with gather_context([attn.wqkv.weight]):
                                with torch.no_grad():
                                    attn.wqkv.weight.copy_(merged_wqkv)
                                    
                                    # Verify and fix gate part if needed (same as training script)
                                    headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                                    elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                                    
                                    if headwise_gate or elementwise_gate:
                                        base_qkv_dim = (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim
                                        gate_dim = attn.num_heads if headwise_gate else attn.num_heads * attn.head_dim
                                        gate_part = attn.wqkv.weight[base_qkv_dim:, :]
                                        
                                        # Check if gate part needs initialization
                                        gate_mean = gate_part.mean().item()
                                        gate_std = gate_part.std().item()
                                        has_nan = torch.isnan(gate_part).any().item()
                                        has_inf = torch.isinf(gate_part).any().item()
                                        
                                        if has_nan or has_inf or (abs(gate_mean) < 1e-6 and gate_std < 1e-6) or abs(gate_mean) > 0.1 or gate_std > 0.1:
                                            # Initialize gate part
                                            initializer_range = model.config.llm_config.initializer_range
                                            gate_part.normal_(mean=0.0, std=initializer_range)
                                    
                                    loaded_wqkv_count += 1
                        else:
                            print(f"   ⚠️  Layer {i}: wqkv shape mismatch: expected {expected_shape}, got {merged_wqkv.shape}")
                except Exception as e:
                    print(f"   ⚠️  Layer {i}: Failed to load wqkv LoRA weights: {e}")
                    import traceback
                    traceback.print_exc()
            # Fallback: Try to merge from q_proj/k_proj/v_proj if wqkv not found
            elif q_proj_base_key in checkpoint_weights and k_proj_base_key in checkpoint_weights and v_proj_base_key in checkpoint_weights:
                try:
                    print(f"   🔄 Layer {i}: wqkv not found, merging from q_proj/k_proj/v_proj...")
                    # Load q_proj/k_proj/v_proj weights
                    base_q_proj = checkpoint_weights[q_proj_base_key].to(dtype=torch.bfloat16)
                    base_k_proj = checkpoint_weights[k_proj_base_key].to(dtype=torch.bfloat16)
                    base_v_proj = checkpoint_weights[v_proj_base_key].to(dtype=torch.bfloat16)
                    
                    # Load and merge LoRA if exists
                    q_proj_lora_a_key = f"language_model.base_model.model.model.layers.{i}.attention.q_proj.lora_A.default.weight"
                    q_proj_lora_b_key = f"language_model.base_model.model.model.layers.{i}.attention.q_proj.lora_B.default.weight"
                    k_proj_lora_a_key = f"language_model.base_model.model.model.layers.{i}.attention.k_proj.lora_A.default.weight"
                    k_proj_lora_b_key = f"language_model.base_model.model.model.layers.{i}.attention.k_proj.lora_B.default.weight"
                    v_proj_lora_a_key = f"language_model.base_model.model.model.layers.{i}.attention.v_proj.lora_A.default.weight"
                    v_proj_lora_b_key = f"language_model.base_model.model.model.layers.{i}.attention.v_proj.lora_B.default.weight"
                    
                    if q_proj_lora_a_key in checkpoint_weights and q_proj_lora_b_key in checkpoint_weights:
                        q_lora_a = checkpoint_weights[q_proj_lora_a_key].to(dtype=torch.bfloat16)
                        q_lora_b = checkpoint_weights[q_proj_lora_b_key].to(dtype=torch.bfloat16)
                        base_q_proj = base_q_proj + torch.matmul(q_lora_b, q_lora_a)
                    if k_proj_lora_a_key in checkpoint_weights and k_proj_lora_b_key in checkpoint_weights:
                        k_lora_a = checkpoint_weights[k_proj_lora_a_key].to(dtype=torch.bfloat16)
                        k_lora_b = checkpoint_weights[k_proj_lora_b_key].to(dtype=torch.bfloat16)
                        base_k_proj = base_k_proj + torch.matmul(k_lora_b, k_lora_a)
                    if v_proj_lora_a_key in checkpoint_weights and v_proj_lora_b_key in checkpoint_weights:
                        v_lora_a = checkpoint_weights[v_proj_lora_a_key].to(dtype=torch.bfloat16)
                        v_lora_b = checkpoint_weights[v_proj_lora_b_key].to(dtype=torch.bfloat16)
                        base_v_proj = base_v_proj + torch.matmul(v_lora_b, v_lora_a)
                    
                    # Extract gate part from q_proj if gating is enabled
                    headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                    elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                    base_qkv_dim = (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim
                    
                    if headwise_gate or elementwise_gate:
                        # q_proj contains query + gate, need to separate
                        q_dim = attn.num_heads * attn.head_dim
                        gate_dim = attn.num_heads if headwise_gate else attn.num_heads * attn.head_dim
                        merged_q_proj = base_q_proj[:q_dim, :]  # Only query part
                        gate_part = base_q_proj[q_dim:, :]  # Gate part
                    else:
                        merged_q_proj = base_q_proj
                        gate_part = None
                    
                    # Merge q_proj/k_proj/v_proj into wqkv format (InternLM2's interleaved organization)
                    # InternLM2 wqkv organization: [KV_head0: Q_groups, K, V, KV_head1: Q_groups, K, V, ...]
                    num_key_value_groups = attn.num_key_value_groups
                    gs = 2 + num_key_value_groups
                    
                    q_weights_list = []
                    k_weights_list = []
                    v_weights_list = []
                    
                    # Split q_proj by key_value_head groups
                    q_per_kv_head = attn.num_key_value_groups * attn.head_dim
                    for kv_head in range(attn.num_key_value_heads):
                        q_start = kv_head * q_per_kv_head
                        q_end = q_start + q_per_kv_head
                        q_weights_list.append(merged_q_proj[q_start:q_end, :])
                        k_weights_list.append(base_k_proj[kv_head * attn.head_dim:(kv_head + 1) * attn.head_dim, :])
                        v_weights_list.append(base_v_proj[kv_head * attn.head_dim:(kv_head + 1) * attn.head_dim, :])
                    
                    # Interleave Q, K, V for each key_value_head
                    wqkv_parts = []
                    for kv_head in range(attn.num_key_value_heads):
                        # For each Q group
                        for q_group in range(num_key_value_groups):
                            wqkv_parts.append(q_weights_list[kv_head][q_group * attn.head_dim:(q_group + 1) * attn.head_dim, :])
                        # Then K and V
                        wqkv_parts.append(k_weights_list[kv_head])
                        wqkv_parts.append(v_weights_list[kv_head])
                    
                    merged_wqkv_base = torch.cat(wqkv_parts, dim=0)
                    
                    # Add gate part if exists
                    if gate_part is not None:
                        merged_wqkv = torch.cat([merged_wqkv_base, gate_part], dim=0)
                    else:
                        merged_wqkv = merged_wqkv_base
                    
                    # Check for NaN/Inf
                    if torch.isnan(merged_wqkv).any() or torch.isinf(merged_wqkv).any():
                        print(f"   ⚠️  Layer {i}: Merged wqkv (from q/k/v) contains NaN/Inf! Skipping...")
                    else:
                        expected_shape = attn.wqkv.weight.shape
                        if merged_wqkv.shape == expected_shape:
                            with gather_context([attn.wqkv.weight]):
                                with torch.no_grad():
                                    attn.wqkv.weight.copy_(merged_wqkv)
                                    loaded_wqkv_count += 1
                                    print(f"   ✅ Layer {i}: Successfully merged q_proj/k_proj/v_proj into wqkv")
                        else:
                            print(f"   ⚠️  Layer {i}: Merged wqkv shape mismatch: expected {expected_shape}, got {merged_wqkv.shape}")
                except Exception as e:
                    print(f"   ⚠️  Layer {i}: Failed to merge q_proj/k_proj/v_proj into wqkv: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Process q_proj/k_proj/v_proj weights (for internvl_chat_finetune_gate.py checkpoints)
        elif not USE_WQKV_VERSION and hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
            # Process q_proj
            q_proj_key_prefix = f"language_model.base_model.model.model.layers.{i}.attention.q_proj"
            q_proj_base_key = f"{q_proj_key_prefix}.base_layer.weight"
            q_proj_lora_a_key = f"{q_proj_key_prefix}.lora_A.default.weight"
            q_proj_lora_b_key = f"{q_proj_key_prefix}.lora_B.default.weight"
            
            if q_proj_base_key in checkpoint_weights:
                try:
                    base_q_proj = checkpoint_weights[q_proj_base_key].to(dtype=torch.bfloat16)
                    
                    if q_proj_lora_a_key in checkpoint_weights and q_proj_lora_b_key in checkpoint_weights:
                        q_proj_lora_a = checkpoint_weights[q_proj_lora_a_key].to(dtype=torch.bfloat16)
                        q_proj_lora_b = checkpoint_weights[q_proj_lora_b_key].to(dtype=torch.bfloat16)
                        q_proj_lora_delta = torch.matmul(q_proj_lora_b, q_proj_lora_a)
                        merged_q_proj = base_q_proj + q_proj_lora_delta
                    else:
                        merged_q_proj = base_q_proj
                    
                    if not (torch.isnan(merged_q_proj).any() or torch.isinf(merged_q_proj).any()):
                        if merged_q_proj.shape == attn.q_proj.weight.shape:
                            with gather_context([attn.q_proj.weight]):
                                with torch.no_grad():
                                    attn.q_proj.weight.copy_(merged_q_proj)
                                    loaded_q_proj_count += 1
                except Exception as e:
                    print(f"   ⚠️  Layer {i}: Failed to load q_proj LoRA weights: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Process k_proj
            k_proj_key_prefix = f"language_model.base_model.model.model.layers.{i}.attention.k_proj"
            k_proj_base_key = f"{k_proj_key_prefix}.base_layer.weight"
            k_proj_lora_a_key = f"{k_proj_key_prefix}.lora_A.default.weight"
            k_proj_lora_b_key = f"{k_proj_key_prefix}.lora_B.default.weight"
            
            if k_proj_base_key in checkpoint_weights:
                try:
                    base_k_proj = checkpoint_weights[k_proj_base_key].to(dtype=torch.bfloat16)
                    
                    if k_proj_lora_a_key in checkpoint_weights and k_proj_lora_b_key in checkpoint_weights:
                        k_proj_lora_a = checkpoint_weights[k_proj_lora_a_key].to(dtype=torch.bfloat16)
                        k_proj_lora_b = checkpoint_weights[k_proj_lora_b_key].to(dtype=torch.bfloat16)
                        k_proj_lora_delta = torch.matmul(k_proj_lora_b, k_proj_lora_a)
                        merged_k_proj = base_k_proj + k_proj_lora_delta
                    else:
                        merged_k_proj = base_k_proj
                    
                    if not (torch.isnan(merged_k_proj).any() or torch.isinf(merged_k_proj).any()):
                        if merged_k_proj.shape == attn.k_proj.weight.shape:
                            with gather_context([attn.k_proj.weight]):
                                with torch.no_grad():
                                    attn.k_proj.weight.copy_(merged_k_proj)
                                    loaded_k_proj_count += 1
                except Exception as e:
                    print(f"   ⚠️  Layer {i}: Failed to load k_proj LoRA weights: {e}")
            
            # Process v_proj
            v_proj_key_prefix = f"language_model.base_model.model.model.layers.{i}.attention.v_proj"
            v_proj_base_key = f"{v_proj_key_prefix}.base_layer.weight"
            v_proj_lora_a_key = f"{v_proj_key_prefix}.lora_A.default.weight"
            v_proj_lora_b_key = f"{v_proj_key_prefix}.lora_B.default.weight"
            
            if v_proj_base_key in checkpoint_weights:
                try:
                    base_v_proj = checkpoint_weights[v_proj_base_key].to(dtype=torch.bfloat16)
                    
                    if v_proj_lora_a_key in checkpoint_weights and v_proj_lora_b_key in checkpoint_weights:
                        v_proj_lora_a = checkpoint_weights[v_proj_lora_a_key].to(dtype=torch.bfloat16)
                        v_proj_lora_b = checkpoint_weights[v_proj_lora_b_key].to(dtype=torch.bfloat16)
                        v_proj_lora_delta = torch.matmul(v_proj_lora_b, v_proj_lora_a)
                        merged_v_proj = base_v_proj + v_proj_lora_delta
                    else:
                        merged_v_proj = base_v_proj
                    
                    if not (torch.isnan(merged_v_proj).any() or torch.isinf(merged_v_proj).any()):
                        if merged_v_proj.shape == attn.v_proj.weight.shape:
                            with gather_context([attn.v_proj.weight]):
                                with torch.no_grad():
                                    attn.v_proj.weight.copy_(merged_v_proj)
                                    loaded_v_proj_count += 1
                except Exception as e:
                    print(f"   ⚠️  Layer {i}: Failed to load v_proj LoRA weights: {e}")
        
        # Process wo (output projection) weights
        wo_key_prefix = f"language_model.base_model.model.model.layers.{i}.attention.wo"
        wo_base_key = f"{wo_key_prefix}.base_layer.weight"
        wo_lora_a_key = f"{wo_key_prefix}.lora_A.default.weight"
        wo_lora_b_key = f"{wo_key_prefix}.lora_B.default.weight"
            
        if wo_base_key in checkpoint_weights and hasattr(attn, 'wo'):
            try:
                base_wo = checkpoint_weights[wo_base_key].to(dtype=torch.bfloat16)
                
                if wo_lora_a_key in checkpoint_weights and wo_lora_b_key in checkpoint_weights:
                    wo_lora_a = checkpoint_weights[wo_lora_a_key].to(dtype=torch.bfloat16)
                    wo_lora_b = checkpoint_weights[wo_lora_b_key].to(dtype=torch.bfloat16)
                    wo_lora_delta = torch.matmul(wo_lora_b, wo_lora_a)
                    merged_wo = base_wo + wo_lora_delta
                else:
                    merged_wo = base_wo
                
                if not (torch.isnan(merged_wo).any() or torch.isinf(merged_wo).any()):
                    if merged_wo.shape == attn.wo.weight.shape:
                        with gather_context([attn.wo.weight]):
                            with torch.no_grad():
                                attn.wo.weight.copy_(merged_wo)
                                loaded_wo_count += 1
            except Exception as e:
                print(f"   ⚠️  Layer {i}: Failed to load wo LoRA weights: {e}")
    
    # Report loading results
    if USE_WQKV_VERSION:
        if loaded_wqkv_count > 0:
            print(f"   ✅ Successfully loaded wqkv LoRA weights for {loaded_wqkv_count} layers")
    else:
        if loaded_q_proj_count > 0:
            print(f"   ✅ Successfully loaded q_proj LoRA weights for {loaded_q_proj_count} layers")
        if loaded_k_proj_count > 0:
            print(f"   ✅ Successfully loaded k_proj LoRA weights for {loaded_k_proj_count} layers")
        if loaded_v_proj_count > 0:
            print(f"   ✅ Successfully loaded v_proj LoRA weights for {loaded_v_proj_count} layers")
    
    if loaded_wo_count > 0:
        print(f"   ✅ Successfully loaded wo LoRA weights for {loaded_wo_count} layers")
    
    total_loaded = (loaded_wqkv_count if USE_WQKV_VERSION else (loaded_q_proj_count + loaded_k_proj_count + loaded_v_proj_count)) + loaded_wo_count
    if total_loaded == 0:
        print(f"   ⚠️  No LoRA weights were loaded")


def _load_direct_weights(model, checkpoint_weights, checkpoint_path):
    """
    Load direct weights (non-LoRA) from checkpoint.
    Supports both q_proj/k_proj/v_proj format and wqkv format.
    If checkpoint is q_proj/k_proj/v_proj format but model is wqkv format, merge them.
    
    Args:
        model: InternVLChatModel with gate version
        checkpoint_weights: Dictionary of weights from model.safetensors
        checkpoint_path: Path to checkpoint directory
    """
    try:
        # Priority 1: Try querygate_withbias version (uses q_proj/k_proj/v_proj)
        from internvl.model.internlm2.modeling_internlm2_querygate_withbias import InternLM2Attention as InternLM2Attention_QueryGateWithBias, InternLM2FlashAttention2 as InternLM2FlashAttention2_QueryGateWithBias
        InternLM2Attention_Class = InternLM2Attention_QueryGateWithBias
        InternLM2FlashAttention2_Class = InternLM2FlashAttention2_QueryGateWithBias
    except ImportError:
        try:
            # Fallback: Try gate version
            from internvl.model.internlm2.modeling_internlm2_gate import InternLM2Attention, InternLM2FlashAttention2
            InternLM2Attention_Class = InternLM2Attention
            InternLM2FlashAttention2_Class = InternLM2FlashAttention2
        except ImportError:
            print("   ⚠️  Gate version not available")
            return
    
    import contextlib
    try:
        from deepspeed import zero
        gather_context = zero.GatheredParameters
    except ImportError:
        gather_context = contextlib.nullcontext
    
    # Get layers
    try:
        if hasattr(model.language_model, 'model'):
            causal_lm_model = getattr(model.language_model, 'model')
        elif hasattr(model.language_model, 'base_model'):
            causal_lm_model = getattr(model.language_model, 'base_model')
        else:
            causal_lm_model = model.language_model
        
        if hasattr(causal_lm_model, 'model'):
            layers_model = getattr(causal_lm_model, 'model')
            if hasattr(layers_model, 'layers'):
                layers = layers_model.layers
            else:
                raise AttributeError(f"layers_model ({type(layers_model)}) does not have 'layers' attribute")
        else:
            if hasattr(causal_lm_model, 'layers'):
                layers = causal_lm_model.layers
            else:
                raise AttributeError(f"causal_lm_model ({type(causal_lm_model)}) does not have 'model' or 'layers' attribute")
    except AttributeError as e:
        print(f"   ❌ Error accessing layers: {e}")
        return
    
    loaded_wqkv_count = 0
    loaded_q_proj_count = 0
    loaded_k_proj_count = 0
    loaded_v_proj_count = 0
    loaded_wo_count = 0
    
    for i, layer in enumerate(layers):
        attn = layer.attention
        if not isinstance(attn, (InternLM2Attention_Class, InternLM2FlashAttention2_Class)):
            continue
        
        # Check for direct weight format (non-LoRA)
        q_proj_key = f"language_model.base_model.model.model.layers.{i}.attention.q_proj.weight"
        k_proj_key = f"language_model.base_model.model.model.layers.{i}.attention.k_proj.weight"
        v_proj_key = f"language_model.base_model.model.model.layers.{i}.attention.v_proj.weight"
        wqkv_key = f"language_model.base_model.model.model.layers.{i}.attention.wqkv.weight"
        wo_key = f"language_model.base_model.model.model.layers.{i}.attention.wo.weight"
        
        # Also check for bias (if qkv_bias is enabled)
        q_proj_bias_key = f"language_model.base_model.model.model.layers.{i}.attention.q_proj.bias"
        k_proj_bias_key = f"language_model.base_model.model.model.layers.{i}.attention.k_proj.bias"
        v_proj_bias_key = f"language_model.base_model.model.model.layers.{i}.attention.v_proj.bias"
        wqkv_bias_key = f"language_model.base_model.model.model.layers.{i}.attention.wqkv.bias"
        
        # Process wqkv weights (if checkpoint has wqkv and model uses wqkv)
        if USE_WQKV_VERSION and hasattr(attn, 'wqkv') and wqkv_key in checkpoint_weights:
            try:
                wqkv_weight = checkpoint_weights[wqkv_key].to(dtype=torch.bfloat16)
                if not (torch.isnan(wqkv_weight).any() or torch.isinf(wqkv_weight).any()):
                    if wqkv_weight.shape == attn.wqkv.weight.shape:
                        with gather_context([attn.wqkv.weight]):
                            with torch.no_grad():
                                attn.wqkv.weight.copy_(wqkv_weight)
                                
                                # Verify and fix gate part if needed
                                headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                                elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                                
                                if headwise_gate or elementwise_gate:
                                    base_qkv_dim = (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim
                                    gate_dim = attn.num_heads if headwise_gate else attn.num_heads * attn.head_dim
                                    gate_part = attn.wqkv.weight[base_qkv_dim:, :]
                                    
                                    gate_mean = gate_part.mean().item()
                                    gate_std = gate_part.std().item()
                                    has_nan = torch.isnan(gate_part).any().item()
                                    has_inf = torch.isinf(gate_part).any().item()
                                    
                                    if has_nan or has_inf or (abs(gate_mean) < 1e-6 and gate_std < 1e-6) or abs(gate_mean) > 0.1 or gate_std > 0.1:
                                        initializer_range = model.config.llm_config.initializer_range
                                        gate_part.normal_(mean=0.0, std=initializer_range)
                                
                                loaded_wqkv_count += 1
            except Exception as e:
                print(f"   ⚠️  Layer {i}: Failed to load wqkv weight: {e}")
        
        # Process q_proj/k_proj/v_proj weights (if checkpoint has them)
        elif q_proj_key in checkpoint_weights and k_proj_key in checkpoint_weights and v_proj_key in checkpoint_weights:
            try:
                q_proj_weight = checkpoint_weights[q_proj_key].to(dtype=torch.bfloat16)
                k_proj_weight = checkpoint_weights[k_proj_key].to(dtype=torch.bfloat16)
                v_proj_weight = checkpoint_weights[v_proj_key].to(dtype=torch.bfloat16)
                
                # Load bias if exists
                q_proj_bias = checkpoint_weights.get(q_proj_bias_key, None)
                k_proj_bias = checkpoint_weights.get(k_proj_bias_key, None)
                v_proj_bias = checkpoint_weights.get(v_proj_bias_key, None)
                if q_proj_bias is not None:
                    q_proj_bias = q_proj_bias.to(dtype=torch.bfloat16)
                if k_proj_bias is not None:
                    k_proj_bias = k_proj_bias.to(dtype=torch.bfloat16)
                if v_proj_bias is not None:
                    v_proj_bias = v_proj_bias.to(dtype=torch.bfloat16)
                
                # If model uses wqkv, merge q_proj/k_proj/v_proj into wqkv
                if USE_WQKV_VERSION and hasattr(attn, 'wqkv'):
                    if i == 0:  # Only print for first layer to avoid spam
                        print(f"   🔄 Layer {i}: Merging q_proj/k_proj/v_proj into wqkv...")
                    # Extract gate part from q_proj if gating is enabled
                    headwise_gate = getattr(attn, 'headwise_attn_output_gate', False)
                    elementwise_gate = getattr(attn, 'elementwise_attn_output_gate', False)
                    base_qkv_dim = (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim
                    
                    if headwise_gate or elementwise_gate:
                        q_dim = attn.num_heads * attn.head_dim
                        gate_dim = attn.num_heads if headwise_gate else attn.num_heads * attn.head_dim
                        merged_q_proj = q_proj_weight[:q_dim, :]  # Only query part
                        gate_part = q_proj_weight[q_dim:, :]  # Gate part
                    else:
                        merged_q_proj = q_proj_weight
                        gate_part = None
                    
                    # Merge q_proj/k_proj/v_proj into wqkv format (InternLM2's interleaved organization)
                    # This is the REVERSE of the split logic in training script
                    # Training script splits wqkv into q_proj/k_proj/v_proj by:
                    # 1. For each kv_head: extract Q_groups (num_key_value_groups groups), then K, then V
                    # 2. Concatenate all Q weights, all K weights, all V weights
                    # So q_proj is organized as: [KV_head0_Q_groups, KV_head1_Q_groups, ...]
                    # To merge back, we need to:
                    # 1. Split q_proj by kv_head (each kv_head has num_key_value_groups * head_dim rows)
                    # 2. For each kv_head, interleave: [Q_group1, Q_group2, ..., Q_groupN, K, V]
                    
                    num_key_value_groups = attn.num_key_value_groups
                    gs = 2 + num_key_value_groups
                    
                    # Verify dimensions
                    q_dim_expected = attn.num_heads * attn.head_dim
                    k_dim_expected = attn.num_key_value_heads * attn.head_dim
                    v_dim_expected = attn.num_key_value_heads * attn.head_dim
                    
                    if merged_q_proj.shape[0] != q_dim_expected:
                        print(f"   ⚠️  Layer {i}: q_proj dimension mismatch: expected {q_dim_expected}, got {merged_q_proj.shape[0]}")
                        continue
                    if k_proj_weight.shape[0] != k_dim_expected:
                        print(f"   ⚠️  Layer {i}: k_proj dimension mismatch: expected {k_dim_expected}, got {k_proj_weight.shape[0]}")
                        continue
                    if v_proj_weight.shape[0] != v_dim_expected:
                        print(f"   ⚠️  Layer {i}: v_proj dimension mismatch: expected {v_dim_expected}, got {v_proj_weight.shape[0]}")
                        continue
                    
                    q_weights_list = []
                    k_weights_list = []
                    v_weights_list = []
                    
                    # Split q_proj by key_value_head groups
                    # q_proj is organized as: [KV_head0_Q_groups, KV_head1_Q_groups, ...]
                    # Each kv_head has num_key_value_groups * head_dim rows
                    q_per_kv_head = attn.num_key_value_groups * attn.head_dim
                    for kv_head in range(attn.num_key_value_heads):
                        q_start = kv_head * q_per_kv_head
                        q_end = q_start + q_per_kv_head
                        q_weights_list.append(merged_q_proj[q_start:q_end, :])
                        k_weights_list.append(k_proj_weight[kv_head * attn.head_dim:(kv_head + 1) * attn.head_dim, :])
                        v_weights_list.append(v_proj_weight[kv_head * attn.head_dim:(kv_head + 1) * attn.head_dim, :])
                    
                    # Interleave Q, K, V for each key_value_head (REVERSE of training script's extraction)
                    # Training script extracts: [Q_group1, Q_group2, ..., Q_groupN, K, V] for each kv_head
                    # So we need to reconstruct: [Q_group1, Q_group2, ..., Q_groupN, K, V] for each kv_head
                    wqkv_parts = []
                    for kv_head in range(attn.num_key_value_heads):
                        # For each Q group (in order)
                        for q_group in range(num_key_value_groups):
                            q_group_start = q_group * attn.head_dim
                            q_group_end = q_group_start + attn.head_dim
                            wqkv_parts.append(q_weights_list[kv_head][q_group_start:q_group_end, :])
                        # Then K (second-to-last in wqkv, which is gs - 2)
                        wqkv_parts.append(k_weights_list[kv_head])
                        # Then V (last in wqkv, which is gs - 1)
                        wqkv_parts.append(v_weights_list[kv_head])
                    
                    merged_wqkv_base = torch.cat(wqkv_parts, dim=0)
                    
                    # Add gate part if exists
                    if gate_part is not None:
                        merged_wqkv = torch.cat([merged_wqkv_base, gate_part], dim=0)
                    else:
                        merged_wqkv = merged_wqkv_base
                    
                    # Verify merged_wqkv_base shape
                    expected_base_shape = (attn.num_heads + 2 * attn.num_key_value_heads) * attn.head_dim
                    if merged_wqkv_base.shape[0] != expected_base_shape:
                        print(f"   ⚠️  Layer {i}: Merged wqkv_base shape mismatch: expected {expected_base_shape} rows, got {merged_wqkv_base.shape[0]}")
                        print(f"      num_heads={attn.num_heads}, num_key_value_heads={attn.num_key_value_heads}, head_dim={attn.head_dim}")
                        print(f"      num_key_value_groups={num_key_value_groups}, gs={gs}")
                        print(f"      merged_q_proj shape: {merged_q_proj.shape}")
                        print(f"      k_proj_weight shape: {k_proj_weight.shape}")
                        print(f"      v_proj_weight shape: {v_proj_weight.shape}")
                        continue
                    
                    # Check for NaN/Inf
                    if torch.isnan(merged_wqkv_base).any() or torch.isinf(merged_wqkv_base).any():
                        print(f"   ⚠️  Layer {i}: Merged wqkv_base (from q/k/v) contains NaN/Inf! Skipping...")
                        if i == 0:
                            print(f"      merged_q_proj has NaN: {torch.isnan(merged_q_proj).any()}, has Inf: {torch.isinf(merged_q_proj).any()}")
                            print(f"      k_proj_weight has NaN: {torch.isnan(k_proj_weight).any()}, has Inf: {torch.isinf(k_proj_weight).any()}")
                            print(f"      v_proj_weight has NaN: {torch.isnan(v_proj_weight).any()}, has Inf: {torch.isinf(v_proj_weight).any()}")
                        continue
                    
                    if gate_part is not None:
                        if torch.isnan(gate_part).any() or torch.isinf(gate_part).any():
                            print(f"   ⚠️  Layer {i}: Gate part contains NaN/Inf! Reinitializing...")
                            initializer_range = model.config.llm_config.initializer_range
                            gate_part.normal_(mean=0.0, std=initializer_range)
                    
                    expected_shape = attn.wqkv.weight.shape
                    if merged_wqkv.shape == expected_shape:
                        # Verify the merged weight before copying
                        if torch.isnan(merged_wqkv).any() or torch.isinf(merged_wqkv).any():
                            print(f"   ⚠️  Layer {i}: Merged wqkv (from q/k/v) contains NaN/Inf! Skipping...")
                        else:
                            with gather_context([attn.wqkv.weight]):
                                with torch.no_grad():
                                    attn.wqkv.weight.copy_(merged_wqkv)
                                    
                                    # Merge bias if exists (same interleaving as weights)
                                    if (q_proj_bias is not None or k_proj_bias is not None or v_proj_bias is not None) and hasattr(attn.wqkv, 'bias') and attn.wqkv.bias is not None:
                                        # Get device from one of the biases
                                        bias_device = None
                                        if q_proj_bias is not None:
                                            bias_device = q_proj_bias.device
                                        elif k_proj_bias is not None:
                                            bias_device = k_proj_bias.device
                                        elif v_proj_bias is not None:
                                            bias_device = v_proj_bias.device
                                        
                                        if bias_device is not None:
                                            # Extract gate bias from q_proj if gating is enabled
                                            if headwise_gate or elementwise_gate:
                                                q_dim = attn.num_heads * attn.head_dim
                                                merged_q_bias = q_proj_bias[:q_dim] if q_proj_bias is not None else torch.zeros(q_dim, dtype=torch.bfloat16, device=bias_device)
                                                gate_bias = q_proj_bias[q_dim:] if q_proj_bias is not None else None
                                            else:
                                                merged_q_bias = q_proj_bias if q_proj_bias is not None else torch.zeros(attn.num_heads * attn.head_dim, dtype=torch.bfloat16, device=bias_device)
                                                gate_bias = None
                                            
                                            # Merge q/k/v bias into wqkv format (same interleaving as weights)
                                            bias_parts = []
                                            for kv_head in range(attn.num_key_value_heads):
                                                # For each Q group
                                                for q_group in range(num_key_value_groups):
                                                    q_bias_start = kv_head * q_per_kv_head + q_group * attn.head_dim
                                                    q_bias_end = q_bias_start + attn.head_dim
                                                    bias_parts.append(merged_q_bias[q_bias_start:q_bias_end])
                                                # Then K and V
                                                if k_proj_bias is not None:
                                                    bias_parts.append(k_proj_bias[kv_head * attn.head_dim:(kv_head + 1) * attn.head_dim])
                                                else:
                                                    bias_parts.append(torch.zeros(attn.head_dim, dtype=torch.bfloat16, device=bias_device))
                                                if v_proj_bias is not None:
                                                    bias_parts.append(v_proj_bias[kv_head * attn.head_dim:(kv_head + 1) * attn.head_dim])
                                                else:
                                                    bias_parts.append(torch.zeros(attn.head_dim, dtype=torch.bfloat16, device=bias_device))
                                            
                                            merged_wqkv_bias_base = torch.cat(bias_parts, dim=0)
                                            
                                            # Add gate bias if exists
                                            if gate_bias is not None:
                                                merged_wqkv_bias = torch.cat([merged_wqkv_bias_base, gate_bias], dim=0)
                                            else:
                                                merged_wqkv_bias = merged_wqkv_bias_base
                                            
                                            # Copy bias
                                            if not (torch.isnan(merged_wqkv_bias).any() or torch.isinf(merged_wqkv_bias).any()):
                                                if merged_wqkv_bias.shape == attn.wqkv.bias.shape:
                                                    with gather_context([attn.wqkv.bias]):
                                                        with torch.no_grad():
                                                            attn.wqkv.bias.copy_(merged_wqkv_bias)
                                                else:
                                                    if i == 0:
                                                        print(f"   ⚠️  Layer {i}: Merged wqkv bias shape mismatch: expected {attn.wqkv.bias.shape}, got {merged_wqkv_bias.shape}")
                                            else:
                                                if i == 0:
                                                    print(f"   ⚠️  Layer {i}: Merged wqkv bias contains NaN/Inf!")
                                    
                                    # Verify after copying
                                    if torch.isnan(attn.wqkv.weight).any() or torch.isinf(attn.wqkv.weight).any():
                                        print(f"   ⚠️  Layer {i}: wqkv.weight contains NaN/Inf after copying! This is unexpected.")
                                    else:
                                        loaded_wqkv_count += 1
                                        if i == 0:
                                            print(f"   ✅ Successfully merged q_proj/k_proj/v_proj into wqkv")
                                            print(f"      wqkv.weight shape: {attn.wqkv.weight.shape}")
                                            print(f"      wqkv.weight stats: mean={attn.wqkv.weight.mean().item():.6f}, std={attn.wqkv.weight.std().item():.6f}")
                                            print(f"      wqkv.weight range: [{attn.wqkv.weight.min().item():.6f}, {attn.wqkv.weight.max().item():.6f}]")
                                            # Verify a sample of weights to ensure merge is correct (using torch operations to avoid BFloat16->numpy issue)
                                            # Check first few rows of wqkv_base (should match first Q group of first kv_head)
                                            sample_q_row = merged_q_proj[0, :5]  # First row, first 5 cols
                                            sample_wqkv_row = merged_wqkv_base[0, :5]  # First row of merged
                                            if not torch.allclose(sample_q_row, sample_wqkv_row, atol=1e-5):
                                                print(f"   ⚠️  Layer {i}: Weight verification failed! First row mismatch.")
                                                print(f"      q_proj[0, :5] = {sample_q_row.tolist()}")
                                                print(f"      wqkv_base[0, :5] = {sample_wqkv_row.tolist()}")
                                            else:
                                                print(f"      ✅ Weight verification passed (first row matches)")
                    else:
                        print(f"   ⚠️  Layer {i}: Merged wqkv shape mismatch: expected {expected_shape}, got {merged_wqkv.shape}")
                        if i == 0:
                            print(f"      base_qkv_dim={base_qkv_dim}, gate_dim={gate_dim if gate_part is not None else 0}")
                            print(f"      merged_wqkv_base shape: {merged_wqkv_base.shape}")
                            print(f"      gate_part shape: {gate_part.shape if gate_part is not None else 'None'}")
                
                # If model uses q_proj/k_proj/v_proj, load directly
                elif hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
                    if not (torch.isnan(q_proj_weight).any() or torch.isinf(q_proj_weight).any()):
                        if q_proj_weight.shape == attn.q_proj.weight.shape:
                            with gather_context([attn.q_proj.weight]):
                                with torch.no_grad():
                                    attn.q_proj.weight.copy_(q_proj_weight)
                                    loaded_q_proj_count += 1
                    # Load q_proj bias if exists
                    if q_proj_bias is not None and hasattr(attn.q_proj, 'bias') and attn.q_proj.bias is not None:
                        if not (torch.isnan(q_proj_bias).any() or torch.isinf(q_proj_bias).any()):
                            if q_proj_bias.shape == attn.q_proj.bias.shape:
                                with gather_context([attn.q_proj.bias]):
                                    with torch.no_grad():
                                        attn.q_proj.bias.copy_(q_proj_bias)
                    
                    if not (torch.isnan(k_proj_weight).any() or torch.isinf(k_proj_weight).any()):
                        if k_proj_weight.shape == attn.k_proj.weight.shape:
                            with gather_context([attn.k_proj.weight]):
                                with torch.no_grad():
                                    attn.k_proj.weight.copy_(k_proj_weight)
                                    loaded_k_proj_count += 1
                    # Load k_proj bias if exists
                    if k_proj_bias is not None and hasattr(attn.k_proj, 'bias') and attn.k_proj.bias is not None:
                        if not (torch.isnan(k_proj_bias).any() or torch.isinf(k_proj_bias).any()):
                            if k_proj_bias.shape == attn.k_proj.bias.shape:
                                with gather_context([attn.k_proj.bias]):
                                    with torch.no_grad():
                                        attn.k_proj.bias.copy_(k_proj_bias)
                    
                    if not (torch.isnan(v_proj_weight).any() or torch.isinf(v_proj_weight).any()):
                        if v_proj_weight.shape == attn.v_proj.weight.shape:
                            with gather_context([attn.v_proj.weight]):
                                with torch.no_grad():
                                    attn.v_proj.weight.copy_(v_proj_weight)
                                    loaded_v_proj_count += 1
                    # Load v_proj bias if exists
                    if v_proj_bias is not None and hasattr(attn.v_proj, 'bias') and attn.v_proj.bias is not None:
                        if not (torch.isnan(v_proj_bias).any() or torch.isinf(v_proj_bias).any()):
                            if v_proj_bias.shape == attn.v_proj.bias.shape:
                                with gather_context([attn.v_proj.bias]):
                                    with torch.no_grad():
                                        attn.v_proj.bias.copy_(v_proj_bias)
            except Exception as e:
                print(f"   ⚠️  Layer {i}: Failed to load q_proj/k_proj/v_proj weights: {e}")
                import traceback
                traceback.print_exc()
        
        # Process wo weights
        if wo_key in checkpoint_weights and hasattr(attn, 'wo'):
            try:
                wo_weight = checkpoint_weights[wo_key].to(dtype=torch.bfloat16)
                if not (torch.isnan(wo_weight).any() or torch.isinf(wo_weight).any()):
                    if wo_weight.shape == attn.wo.weight.shape:
                        with gather_context([attn.wo.weight]):
                            with torch.no_grad():
                                attn.wo.weight.copy_(wo_weight)
                                loaded_wo_count += 1
            except Exception as e:
                print(f"   ⚠️  Layer {i}: Failed to load wo weight: {e}")
    
    # Report loading results
    if USE_WQKV_VERSION:
        if loaded_wqkv_count > 0:
            print(f"   ✅ Successfully loaded wqkv weights for {loaded_wqkv_count} layers")
    else:
        if loaded_q_proj_count > 0:
            print(f"   ✅ Successfully loaded q_proj weights for {loaded_q_proj_count} layers")
        if loaded_k_proj_count > 0:
            print(f"   ✅ Successfully loaded k_proj weights for {loaded_k_proj_count} layers")
        if loaded_v_proj_count > 0:
            print(f"   ✅ Successfully loaded v_proj weights for {loaded_v_proj_count} layers")
    
    if loaded_wo_count > 0:
        print(f"   ✅ Successfully loaded wo weights for {loaded_wo_count} layers")
    
    total_loaded = (loaded_wqkv_count if USE_WQKV_VERSION else (loaded_q_proj_count + loaded_k_proj_count + loaded_v_proj_count)) + loaded_wo_count
    if total_loaded == 0:
        print(f"   ⚠️  No weights were loaded from checkpoint!")


def load_trained_model(checkpoint_path, base_model_name="OpenGVLab/InternVL2_5-2B"):
    """
    Load InternVL trained model from checkpoint
    Supports both standard and gated attention checkpoints
    """
    print("🚀 Loading InternVL model from checkpoint...")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Check if checkpoint was trained with gated attention
    # Method 1: Check checkpoint path name for gate indicators
    checkpoint_name = os.path.basename(checkpoint_path)
    parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
    full_path_lower = (checkpoint_path + " " + parent_dir).lower()
    
    has_gate_config = False
    elementwise_gate = False
    headwise_gate = False
    
    # Detect from path name
    if 'elementgate' in full_path_lower or 'elementwise' in full_path_lower:
        elementwise_gate = True
        has_gate_config = True
        print(f"   🔍 Detected elementwise gated attention from checkpoint path")
    elif 'headgate' in full_path_lower or 'headwise' in full_path_lower:
        headwise_gate = True
        has_gate_config = True
        print(f"   🔍 Detected headwise gated attention from checkpoint path")
    
    # Method 2: Check config.json file
    checkpoint_config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(checkpoint_config_path):
        try:
            import json
            with open(checkpoint_config_path, 'r') as f:
                checkpoint_config = json.load(f)
            # Check if config indicates gated attention
            if isinstance(checkpoint_config, dict):
                llm_config = checkpoint_config.get('llm_config', {})
                if isinstance(llm_config, dict):
                    config_elementwise = llm_config.get('elementwise_attn_output_gate', False)
                    config_headwise = llm_config.get('headwise_attn_output_gate', False)
                    if config_elementwise or config_headwise:
                        elementwise_gate = config_elementwise
                        headwise_gate = config_headwise
                        has_gate_config = True
                        gate_type = "elementwise" if elementwise_gate else "headwise"
                        print(f"   🔍 Detected {gate_type} gated attention in checkpoint config.json")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint config: {e}")
    
    # Step 1: Load base model config
    print("\n1️⃣ Loading base model config...")
    config = InternVLChatConfig.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Apply gate config if detected
    if has_gate_config and config.llm_config.model_type == 'internlm2':
        config.llm_config.elementwise_attn_output_gate = elementwise_gate
        config.llm_config.headwise_attn_output_gate = headwise_gate
        gate_type = "elementwise" if elementwise_gate else "headwise"
        print(f"   ✅ Config loaded with {gate_type} gated attention")
        
        # Check for qkv_bias in checkpoint config (from internvl_chat_finetune_gate.py)
        # If checkpoint has qkv_bias, use it; otherwise, qkv_bias will default to bias value
        # (handled by configuration_internlm2.py: if qkv_bias is None, defaults to bias)
        checkpoint_config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(checkpoint_config_path):
            try:
                import json
                with open(checkpoint_config_path, 'r') as f:
                    checkpoint_config = json.load(f)
                if isinstance(checkpoint_config, dict):
                    llm_config = checkpoint_config.get('llm_config', {})
                    if isinstance(llm_config, dict):
                        qkv_bias = llm_config.get('qkv_bias', None)
                        if qkv_bias is not None:
                            config.llm_config.qkv_bias = qkv_bias
                            print(f"   ✅ Set qkv_bias = {qkv_bias} from checkpoint config")
                        else:
                            # If qkv_bias is not in checkpoint, it will default to bias value
                            # (handled by configuration_internlm2.py)
                            print(f"   ℹ️  qkv_bias not found in checkpoint config, will use default (bias={config.llm_config.bias})")
            except Exception as e:
                print(f"   ⚠️  Could not read qkv_bias from checkpoint config: {e}")
    else:
        print("   ✅ Config loaded")
    
    # Step 2: Load model
    print(f"\n2️⃣ Loading model from checkpoint...")
    try:
        # For q_proj/k_proj/v_proj gate version, we need ignore_mismatched_sizes=True
        # because gate version splits wqkv into q_proj/k_proj/v_proj, which doesn't match pretrained wqkv
        # For wqkv gate version, we don't need ignore_mismatched_sizes
        ignore_mismatched = (has_gate_config and not USE_WQKV_VERSION)  # q_proj/k_proj/v_proj version needs this
        
        model = InternVLChatModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            ignore_mismatched_sizes=ignore_mismatched
        )
        print("   ✅ Model loaded from checkpoint")
        
        # CRITICAL: Load weights from checkpoint
        # Supports both LoRA format and direct weight format
        # Supports both q_proj/k_proj/v_proj format and wqkv format
        if has_gate_config and USE_GATE_VERSION:
            print(f"\n   🔧 Loading weights from checkpoint...")
            try:
                from safetensors.torch import load_file
                checkpoint_weights = load_file(os.path.join(checkpoint_path, "model.safetensors"))
                
                # Check if checkpoint uses LoRA format
                # Try q_proj LoRA format first (internvl_chat_finetune_gate.py with LoRA)
                sample_key_q_lora = "language_model.base_model.model.model.layers.0.attention.q_proj.base_layer.weight"
                # Try wqkv LoRA format (internvl_chat_finetune_wqkv.py with LoRA)
                sample_key_wqkv_lora = "language_model.base_model.model.model.layers.0.attention.wqkv.base_layer.weight"
                # Try q_proj direct weight format (internvl_chat_finetune_gate.py without LoRA)
                sample_key_q_direct = "language_model.base_model.model.model.layers.0.attention.q_proj.weight"
                # Try wqkv direct weight format (internvl_chat_finetune_wqkv.py without LoRA)
                sample_key_wqkv_direct = "language_model.base_model.model.model.layers.0.attention.wqkv.weight"
                
                if sample_key_q_lora in checkpoint_weights:
                    print(f"   📦 Detected LoRA format (q_proj/k_proj/v_proj), merging and loading weights...")
                    _load_and_convert_lora_weights(model, checkpoint_weights, checkpoint_path)
                elif sample_key_wqkv_lora in checkpoint_weights:
                    print(f"   📦 Detected LoRA format (wqkv), merging and loading weights...")
                    _load_and_convert_lora_weights(model, checkpoint_weights, checkpoint_path)
                elif sample_key_q_direct in checkpoint_weights:
                    print(f"   📦 Detected direct weight format (q_proj/k_proj/v_proj), loading weights...")
                    _load_direct_weights(model, checkpoint_weights, checkpoint_path)
                elif sample_key_wqkv_direct in checkpoint_weights:
                    print(f"   📦 Detected direct weight format (wqkv), loading weights...")
                    _load_direct_weights(model, checkpoint_weights, checkpoint_path)
                else:
                    print(f"   ⚠️  Could not detect checkpoint weight format!")
                    print(f"   🔍 Checked for: {sample_key_q_lora}, {sample_key_wqkv_lora}, {sample_key_q_direct}, {sample_key_wqkv_direct}")
            except Exception as e:
                print(f"   ⚠️  Failed to load weights: {e}")
                import traceback
                traceback.print_exc()
        # Check if num_image_token matches training (should be 64 for 224x224 with ratio 0.5)
        expected_num_image_token = int((224 // 14) ** 2 * (0.5 ** 2))  # 64
        if model.num_image_token != expected_num_image_token:
            print(f"   ⚠️  Warning: num_image_token={model.num_image_token} doesn't match expected {expected_num_image_token}")
            print(f"   🔧 Updating to match training config...")
            model.config.force_image_size = 224
            model.config.downsample_ratio = 0.5
            model.downsample_ratio = 0.5
            model.num_image_token = expected_num_image_token
            print(f"   ✅ Updated num_image_token to {model.num_image_token}")
    except Exception as e:
        print(f"   ⚠️  Failed to load from checkpoint, trying base model: {e}")
        # Fallback to base model if checkpoint loading fails
        # Apply gate config if detected
        if has_gate_config and config.llm_config.model_type == 'internlm2':
            print(f"   🔧 Applying gate config to base model...")
            config.llm_config.elementwise_attn_output_gate = elementwise_gate
            config.llm_config.headwise_attn_output_gate = headwise_gate
        
        model = InternVLChatModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            config=config,
            ignore_mismatched_sizes=has_gate_config and USE_GATE_VERSION
        )
        print("   ✅ Model loaded from base model")
        # Apply training config to base model
        force_image_size = 224
        down_sample_ratio = 0.5
        patch_size = model.patch_size
        if hasattr(model.config, 'force_image_size') and model.config.force_image_size != force_image_size:
            model.vision_model.resize_pos_embeddings(
                old_size=model.config.vision_config.image_size,
                new_size=force_image_size,
                patch_size=patch_size
            )
            model.config.vision_config.image_size = force_image_size
        model.config.force_image_size = force_image_size
        model.config.downsample_ratio = down_sample_ratio
        model.downsample_ratio = down_sample_ratio
        model.num_image_token = int((force_image_size // patch_size) ** 2 * (down_sample_ratio ** 2))
        print(f"   🔧 Updated config: num_image_token={model.num_image_token}")
    
    # Step 3: Load tokenizer
    print(f"\n3️⃣ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")) 
        else base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    # Set model_max_length to match training (12288 for InternVL2.5-2B)
    # This ensures evaluation uses the same sequence length limit as training
    tokenizer.model_max_length = 12288
    print(f"   ✅ Tokenizer loaded (model_max_length: {tokenizer.model_max_length})")
    
    model.eval()
    
    # Disable gradients for all parameters to save memory
    for param in model.parameters():
        param.requires_grad = False
    
    # CRITICAL FIX: Explicitly move model to GPU to ensure all components are on CUDA
    # device_map="auto" may leave some components on CPU, causing "Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)" error
    print(f"\n4️⃣ Moving model to GPU...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
        # Explicitly move vision_model to GPU (this is often the problematic component)
        if hasattr(model, 'vision_model'):
            model.vision_model = model.vision_model.to(device)
        print(f"   ✅ Model moved to {device}")
    else:
        print(f"   ⚠️  CUDA not available, model remains on CPU")
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE MODEL LOADED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    
    return model, tokenizer

def eval_model(args):
    device = "cuda"
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
    
    # Disable gradient computation globally to save memory
    torch.set_grad_enabled(False)
    print("🔒 Gradient computation disabled globally for memory optimization")
    
    # Load model
    try:
        if args.checkpoint_path and os.path.exists(str(args.checkpoint_path)):
            # Load from checkpoint
            model, tokenizer = load_trained_model(
                args.checkpoint_path, 
                args.model_base
            )
        else:
            # Load base model only
            if args.checkpoint_path:
                print(f"ℹ️  Checkpoint path '{args.checkpoint_path}' doesn't exist, loading base model...")
            else:
                print("ℹ️  No checkpoint path provided, loading base model...")
            model, tokenizer = load_base_model(args.model_base)
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Ensure model is in eval mode and disable gradients for all parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("🔒 All model parameters set to requires_grad=False")
    
    # Load test data (support both JSON and JSONL formats)
    print(f"📂 Loading test data from: {args.question_file}")
    data_dict = []
    
    # Check if file is JSONL (each line is a JSON object) or JSON (array or meta config)
    with open(args.question_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)  # Reset to beginning
        
        if first_line.startswith('['):
            # JSON array format (qwenvl style)
            data_dict = json.load(f)
        elif first_line.startswith('{'):
            # Could be JSON object (meta config) or JSONL format
            # Try to parse as JSON object first
            try:
                meta_data = json.load(f)
                # Check if it's a meta config file (has 'how2sign' key with 'annotation' field)
                if isinstance(meta_data, dict) and 'how2sign' in meta_data:
                    annotation_file = meta_data['how2sign'].get('annotation')
                    if annotation_file and os.path.exists(annotation_file):
                        print(f"   Detected meta config file, loading annotation from: {annotation_file}")
                        # Load the actual JSONL file
                        with open(annotation_file, 'r', encoding='utf-8') as ann_f:
                            for line in ann_f:
                                line = line.strip()
                                if line:
                                    try:
                                        data_dict.append(json.loads(line))
                                    except json.JSONDecodeError as e:
                                        print(f"⚠️  Warning: Failed to parse line: {line[:100]}... Error: {e}")
                                        continue
                    else:
                        print(f"⚠️  Warning: Annotation file not found or not specified in meta config")
                        print(f"   Meta data: {meta_data}")
                else:
                    # Not a meta config, treat as single JSON object (unlikely for evaluation)
                    print(f"⚠️  Warning: File is a JSON object but not a meta config. Expected JSONL or JSON array.")
            except json.JSONDecodeError:
                # Not valid JSON, treat as JSONL format
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data_dict.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Failed to parse line: {line[:100]}... Error: {e}")
                            continue
        else:
            # JSONL format (InternVL style) - each line is a JSON object
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data_dict.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Failed to parse line: {line[:100]}... Error: {e}")
                        continue
    
    if args.max_samples:
        data_dict = data_dict[:args.max_samples]
        print(f"   Limited to {args.max_samples} samples")
    
    print(f"   Total samples: {len(data_dict)}\n")
    
    # Validate that we loaded some data
    if len(data_dict) == 0:
        print(f"\n❌ ERROR: No data loaded from {args.question_file}")
        print(f"   Please check that the file exists and is in the correct format (JSONL or JSON array).")
        print(f"   Expected format: JSONL (one JSON object per line) or JSON array.")
        return
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create frames export folder if enabled
    frames_export_dir = None
    if args.export_frames:
        frames_export_dir = os.path.join(args.out_dir, "extracted_frames")
        os.makedirs(frames_export_dir, exist_ok=True)
        print(f"📁 Frames will be exported to: {frames_export_dir}\n")
    
    results = []
    references = []
    predictions = []
    
    print(f"🎬 Starting evaluation...")
    print(f"{'='*70}\n")
    
    # Build transform (same as training code)
    transform = build_transform(is_train=False, input_size=args.image_size, pad2square=False, normalize_type='imagenet')
    
    # Process samples
    for idx, source in enumerate(tqdm(data_dict, desc="Evaluating"), 1):
        try:
            # Support both InternVL format (conversations with "from"/"gpt") and qwenvl format (conversations with "role")
            video_file = source["video"]
            video_path = os.path.join(args.video_folder, video_file)
            
            # Use default prompt template (will be formatted with actual frame count after loading)
            fq = "Translate the American Sign Language in this video to English."
            # default_prompt_template = "Please describe each action change in this video based on the frame transitions. This video contains {num_frames} frames. For each action change, describe ONLY the physical movements: (1) what changes (hand movements, gestures, body positions), (2) how it changes (the transition and motion), and (3) when it changes (the sequence of changes). IMPORTANT: Only describe observable physical actions and movements. Do NOT include any interpretations, speculations, meanings, or explanations (such as 'possibly indicating', 'suggesting', 'rhythmic manner', etc.). Focus strictly on describing the physical actions and movements that you can see."
            # default_prompt = "Observe this ASL video and describe in detail how the person's hand gestures and facial expressions change. For each change you observe, describe: (1) which hand is moving, (2) the specific finger positions (which fingers are extended, curled, or touching), (3) the hand shape and orientation, (4) the hand location relative to the body, (5) the movement direction and how it transitions. Describe all visible changes frame by frame. Only describe the physical movements you can see - do not include background information, interpretations, or speculations."
#             fq = """
# You are an ASL motion-description annotator.
# Describe each video frame accurately, objectively, and with full linguistic detail.
# Do NOT translate the signs into English words. Do NOT infer meaning.
# Only describe observable physical movement.

# For each frame, report the following fields exactly:

# Right hand:
# - Handshape: (e.g., 1-hand, 5-hand, claw, fist, V-hand, bent-V, open-B, flat-O)
# - Palm orientation: (up, down, left, right, inward, outward)
# - Location in signing space: (upper/lower/center-left/center-right, near-face, near-chest, mid-space, side, neutral space)
# - Movement: 
#   - direction (up, down, toward body, away, left, right, circular, arc)
#   - speed (slow, medium, fast)
#   - path type (straight, curved, arc, repeated)
#   - start → end positions

# Left hand:
# - (Same structure as right hand; if static, write "no movement")

# Hand interaction:
# - Contact: (touch, brush, tap, cross, stack, approach-without-touch)
# - Relative position: (above/below, in front/behind, left/right, near/far)
# - Synchrony: (simultaneous movement / alternating movement)
# - Repetition count (1×, 2×, 3× if visible)

# Face / Non-manual markers (NMM):
# - Eyebrows: (raised, furrowed, neutral)
# - Eyes: (wide, squinting, blinking)
# - Mouth morphemes: (open, "oo", "mm", "ah", pursed, puffed)
# - Head movement: (tilt left/right, nod, shake, forward/backward)
# - Body posture: (lean forward/backward, shoulder shift)

# Formatting Requirements:
# Use this exact format:

# Frame X:
# Right hand: [handshape, palm orientation, location, movement]
# Left hand: [handshape, palm orientation, location, movement]
# Interaction: [contact / relative positioning / repetition]
# Face/NMM: [eyebrows, eyes, mouth shape, head movement, body posture]

# Rules:
# - Be extremely specific about movement path and spatial location.
# - Mention start and end positions for any movement.
# - Mention when a hand remains still.
# - Do not guess the sign or English meaning.
# - Keep descriptions factual, not interpretive.
# - If unsure, use "approximately".
# """
# Will be set after loading frames
            
            # Extract ground truth from conversations or source
            conversations = source.get('conversations', [])
            if len(conversations) >= 2:
                # Get ground truth from the second conversation (assistant's response)
                if "from" in conversations[1]:
                    ground_truth = conversations[1].get('value', '')
                else:
                    ground_truth = conversations[1].get('value', '')
            else:
                ground_truth = source.get('answer', source.get('ground_truth', ''))
            
            if not os.path.exists(video_path):
                print(f"\n⚠️  [{idx}/{len(data_dict)}] Video not found: {video_file}")
                results.append({
                    "video": video_file,
                    "model_output": "ERROR: Video not found",
                    "ground_truth": ground_truth
                })
                continue
            
            # Load video frames using the same method as training code
            try:
                print(f"\n📹 [{idx}/{len(data_dict)}] Loading video: {video_file}")
                image_list = _load_video_locally(
                    video_path,
                    max_num_frames=args.max_num_frames,
                    min_num_frames=args.min_num_frames,
                    sample=args.sampling_method,
                    clip=None
                )
                print(f"   ✅ Loaded {len(image_list)} frames")
                
                # Format prompt with actual number of frames
                num_frames = len(image_list)
                # fq = default_prompt_template.format(num_frames=num_frames)
                # print(f"   📝 Prompt configured for {num_frames} frames (describe each action change)")
            except Exception as e:
                print(f"\n⚠️  [{idx}/{len(data_dict)}] Failed to load video {video_file}: {e}")
                results.append({
                    "video": video_file,
                    "model_output": f"ERROR: Failed to load video - {str(e)}",
                    "ground_truth": ground_truth
                })
                continue
            
            # Export frames to folder if enabled
            if args.export_frames and frames_export_dir:
                try:
                    frame_folder = extract_and_export_frames(
                        video_path=video_path,
                        output_folder=frames_export_dir,
                        max_num_frames=args.max_num_frames,
                        min_num_frames=args.min_num_frames,
                        sample=args.sampling_method,
                        clip=None,
                        video_name=os.path.splitext(video_file)[0]
                    )
                    if idx <= 10:  # Print for first 10 samples
                        print(f"   📁 Frames exported to: {frame_folder} ({len(image_list)} frames)")
                except Exception as e:
                    print(f"\n⚠️  [{idx}/{len(data_dict)}] Failed to export frames for {video_file}: {e}")
            
            # Check if we have too many frames (would exceed sequence length)
            # Use the same max_seq_length as training (12288) to match training behavior
            num_image_token = model.num_image_token
            max_seq_length = tokenizer.model_max_length  # Use tokenizer's model_max_length (12288)
            # Estimate: each frame needs num_image_token tokens, plus text tokens
            # Use conservative estimate: allow max 80% of sequence length for safety
            max_frames_allowed = int((max_seq_length * 0.8) / num_image_token)
            
            if len(image_list) > max_frames_allowed:
                # Reduce number of frames to fit within sequence length
                if max_frames_allowed < args.min_num_frames:
                    print(f"\n⚠️  [{idx}/{len(data_dict)}] Video {video_file} has too many frames ({len(image_list)}), skipping")
                    results.append({
                        "video": video_file,
                        "model_output": f"ERROR: Too many frames ({len(image_list)}) for sequence length (max: {max_frames_allowed})",
                        "ground_truth": ground_truth
                    })
                    continue
                # Uniformly sample frames to reduce count
                step = len(image_list) / max_frames_allowed
                indices = [int(i * step) for i in range(max_frames_allowed)]
                image_list = [image_list[i] for i in indices if i < len(image_list)]
                print(f"\n⚠️  [{idx}/{len(data_dict)}] Reduced frames to {len(image_list)} (from original) to fit sequence length")
            
            # Transform each frame image and stack them (same as training code)
            # Disable gradient tracking during transformation to save memory
            with torch.no_grad():
                pixel_values = [transform(image) for image in image_list]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
            # Shape should be [num_frames, channels, height, width] (not [batch, num_frames, ...])
            # InternVL expects pixel_values without batch dimension for video frames
            
            # Clear cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate prompt with <image> placeholders (same format as training)
            # Training code uses: Frame-1: <image>\nFrame-2: <image>\n...
            # Each <image> represents 1 frame, and will be replaced with num_image_token IMG_CONTEXT_TOKENs
            special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
            question = f"{special_tokens}\n{fq}"
            
            # num_patches_list tells model how many frames each <image> placeholder represents
            # For training format, each <image> represents 1 frame
            # So num_patches_list = [1] * num_frames
            num_patches_list = [1] * len(image_list)
            
            # Generate using model.chat() (same parameters as Qwen2VL evaluation)
            generation_config = dict(
                num_beams=5,                    # Beam search for better quality
                do_sample=True,                 # Enable sampling for better diversity
                temperature=0.7,                # Temperature for generation (0.7 is a good balance)
                top_p=0.9,                      # Nucleus sampling
                top_k=50,                       # Top-k sampling
                length_penalty=1.0,             # Length penalty (1.0 = neutral)
                no_repeat_ngram_size=4,        # Prevent 4-gram repetition
                repetition_penalty=1.1,         # Slight penalty for token repetition
                min_length=1,                   # Minimum output length
                max_new_tokens=args.max_new_tokens  # Maximum tokens to generate
            )
            
            # Use no_grad context with explicit gradient disabling
            with torch.no_grad():
                # Ensure gradient is disabled (redundant but explicit for safety)
                torch.set_grad_enabled(False)
                output = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,  # [num_frames, C, H, W]
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,  # Each <image> represents 1 frame
                    verbose=False
                )
            
            # Explicitly delete pixel_values and clear cache
            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Force synchronization to ensure memory is freed
                torch.cuda.synchronize()
            
            # Store results
            references.append(ground_truth)
            predictions.append(output)
            
            results.append({
                "video": video_file,
                "model_output": output,
                "ground_truth": ground_truth
            })
            
            # Print first 10 examples
            if idx <= 10:
                print(f"\n{'─'*70}")
                print(f"[{idx}/{len(data_dict)}] {video_file}")
                print(f"Ground truth: {ground_truth}")
                print(f"Prediction:   {output}")
                
                # Check if it's learning
                if output.lower() == ground_truth.lower():
                    print(f"✅ EXACT MATCH!")
                elif any(word in output.lower() for word in ground_truth.lower().split()):
                    print(f"✅ Partial match (has some words)")
                else:
                    print(f"⚠️  No obvious match")
        
        except Exception as e:
            print(f"\n❌ Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            # Extract ground truth based on format
            conversations = source.get('conversations', [])
            if len(conversations) >= 2:
                if "from" in conversations[1]:
                    gt = conversations[1].get('value', 'unknown')
                else:
                    gt = conversations[1].get('value', 'unknown')
            else:
                gt = source.get('answer', source.get('ground_truth', 'unknown'))
            
            results.append({
                "video": source.get("video", "unknown"),
                "model_output": f"ERROR: {str(e)}",
                "ground_truth": gt
            })
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"internvl_results_{timestamp}.json"
    output_path = os.path.join(args.out_dir, output_file)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved: {output_path}")
    print(f"{'='*70}\n")
    
    # Evaluation metrics
    if references and predictions:
        try:
            print(f"📊 Calculating evaluation metrics...\n")
            # Auto-detect evaluation path
            script_dir_eval = os.path.dirname(os.path.abspath(__file__))
            project_root_eval = os.path.abspath(os.path.join(script_dir_eval, '../../..'))
            eval_path = os.path.join(project_root_eval, 'evaluation')
            if os.path.exists(eval_path):
                sys.path.append(eval_path)
            else:
                sys.path.append('/code/doc_sign_search/evaluation')
            from ssvp_evaluation import comprehensive_evaluation, print_evaluation_results
            
            eval_results = comprehensive_evaluation(references, predictions)
            print_evaluation_results(eval_results, "InternVL")
            
            # Save metrics
            eval_file = os.path.join(args.out_dir, f"metrics_{timestamp}.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"\n📊 Metrics saved: {eval_file}")
            
        except Exception as e:
            print(f"\n⚠️  Evaluation metrics error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    successful = len([r for r in results if not r['model_output'].startswith('ERROR')])
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"   Total samples: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(results) - successful}")
    if len(results) > 0:
        print(f"   Success rate: {successful/len(results)*100:.1f}%")
    else:
        print(f"   Success rate: N/A (no samples processed)")
    if args.export_frames and frames_export_dir:
        print(f"   📁 Extracted frames saved to: {frames_export_dir}")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate InternVL trained checkpoint on test set")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                       help="Path to checkpoint directory (optional, if not provided, will use base model)")
    parser.add_argument("--model-base", type=str, 
                       default="OpenGVLab/InternVL2_5-2B",
                       help="Base model name")
    parser.add_argument("--video-folder", type=str, required=True,
                       help="Folder containing test videos")
    parser.add_argument("--question-file", type=str, required=True,
                       help="JSON file with test questions")
    parser.add_argument("--out-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                       help="Max tokens to generate")
    parser.add_argument("--min-num-frames", type=int, default=8,
                       help="Minimum number of frames for video data (same as training)")
    parser.add_argument("--max-num-frames", type=int, default=32,
                       help="Maximum number of frames for video data (same as training)")
    parser.add_argument("--sampling-method", type=str, default='fps12.0',
                       help="Video frame sampling method: 'fpsX.X' (e.g., 'fps12.0'), 'rand' (default), 'random_start_every2'")
    parser.add_argument("--image-size", type=int, default=448,
                       help="Image size for processing")
    parser.add_argument("--export-frames", action="store_true",
                       help="Export extracted video frames to folder (saved in out_dir/extracted_frames)")
    
    args = parser.parse_args()
    
    # Validate paths
    if args.checkpoint_path is not None and not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        return
    if not os.path.exists(args.video_folder):
        print(f"❌ Video folder not found: {args.video_folder}")
        return
    if not os.path.exists(args.question_file):
        print(f"❌ Question file not found: {args.question_file}")
        return
    
    eval_model(args)

if __name__ == "__main__":
    main()

