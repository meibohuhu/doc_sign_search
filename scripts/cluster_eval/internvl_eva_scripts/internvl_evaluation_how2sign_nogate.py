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
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
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


######GRPO checkpoint保存的是DeepSpeed格式（PEFT结构，973个key：ViT + LLM base + LoRA A/B + MLP1）。
######直接 from_pretrained 加载merged的 model.safetensors 在60+帧时会产生乱码，原因不明（权重值完全一致，但inference行为不同）。不merge、保持LoRA动态结构可以正常工作。
def _find_ds_checkpoint(checkpoint_path):
    """Return path to mp_rank_00_model_states.pt if this is a DeepSpeed GRPO checkpoint."""
    import glob
    # Look for global_stepXXX/mp_rank_00_model_states.pt
    pattern = os.path.join(checkpoint_path, "global_step*", "mp_rank_00_model_states.pt")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def load_trained_model(checkpoint_path, base_model_name="OpenGVLab/InternVL2_5-2B", sft_checkpoint=None):
    """
    Load InternVL trained model from checkpoint.

    For GRPO/DeepSpeed checkpoints (contain global_stepXXX/mp_rank_00_model_states.pt),
    loads the SFT base (PEFT) model first then overlays the DS weights without merging,
    which avoids the safetensors-reload bug that breaks long-context inference.
    sft_checkpoint: path to the SFT checkpoint (PEFT format) used as base for GRPO.
    """
    print("🚀 Loading InternVL model from checkpoint...")
    print(f"   Checkpoint: {checkpoint_path}")

    ds_ckpt = _find_ds_checkpoint(checkpoint_path)

    if ds_ckpt:
        # ── GRPO checkpoint: load SFT PEFT base, then overlay DS weights ──
        print(f"\n   Detected GRPO DeepSpeed checkpoint: {ds_ckpt}")
        peft_base = sft_checkpoint if sft_checkpoint else base_model_name
        print(f"   Loading SFT base model from: {peft_base}")

        print("\n1️⃣ Loading SFT base (PEFT) model...")
        model = InternVLChatModel.from_pretrained(
            peft_base,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("   ✅ SFT base loaded")

        print("\n2️⃣ Overlaying DeepSpeed weights (no merge)...")
        ds_state = torch.load(ds_ckpt, map_location="cpu")
        sd = ds_state.get("module", ds_state)
        missing, unexpected = model.load_state_dict(sd, strict=False)###### 把GRPO的DS权重覆盖到已经加载好的SFT模型结构里。
### missing=0 → GRPO的每一个key在SFT模型里都找到了对应位置，全部写入
###### unexpected=0 → GRPO没有SFT模型结构里不存在的key
        print(f" GRPO missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"   missing[:5]: {missing[:5]}")
        print("   ✅ GRPO DS weights loaded (LoRA kept unmerged)")

    else:
        # ── Regular checkpoint (SFT safetensors) ──
        print(f"\n1️⃣ Loading model from checkpoint directory...")
        try:
            model = InternVLChatModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            )
            print("   ✅ Model loaded from checkpoint")
        except Exception as e:
            print(f"   ⚠️  Failed to load from checkpoint, trying base model: {e}")
            model = InternVLChatModel.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            )
            print("   ✅ Model loaded from base model")

        force_image_size = 224
        down_sample_ratio = 0.5
        patch_size = model.patch_size
        model.config.force_image_size = force_image_size
        model.config.downsample_ratio = down_sample_ratio
        model.downsample_ratio = down_sample_ratio
        model.num_image_token = int((force_image_size // patch_size) ** 2 * (down_sample_ratio ** 2))
        print(f"   num_image_token={model.num_image_token}")

    # Step 3: Load tokenizer
    print(f"\n3️⃣ Loading tokenizer...")
    # 优先用sft_checkpoint（vocab文件完整），其次checkpoint_path，最后base_model_name
    if sft_checkpoint and os.path.exists(os.path.join(sft_checkpoint, "vocab.json")):
        tok_path = sft_checkpoint
    elif os.path.exists(os.path.join(checkpoint_path, "vocab.json")):
        tok_path = checkpoint_path
    else:
        tok_path = base_model_name
    print(f"   Tokenizer path: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=False)
    tokenizer.model_max_length = 32768
    print(f"   ✅ Tokenizer loaded (model_max_length: {tokenizer.model_max_length})")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"\n4️⃣ Moving model to GPU...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
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
                args.model_base,
                sft_checkpoint=getattr(args, 'sft_checkpoint', None),
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
            # fq = "How many people are in the video"
            # fq = "Translate the American Sign Language in this video to English."
            
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
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                       help="Path to SFT (PEFT) checkpoint used as base for GRPO checkpoints. "
                            "Required when --checkpoint-path is a DeepSpeed GRPO checkpoint.")

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

