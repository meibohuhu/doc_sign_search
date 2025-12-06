#!/usr/bin/env python3
"""
InternVL2 Processing Script with Attention Visualization
Uses hook to intercept projector output and feed to InternLM2Model.forward() for attention extraction.

Usage:
    python internlm2/internvl2_evaluation_attention.py \
        --model-path OpenGVLab/InternVL2-2B \
        --video-path /path/to/video.mp4 \
        --out-dir ./output \
        --save-attention
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
from internlm2.modeling_internlm2_old import InternLM2Model
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
    
    total_visual_patches = sum(num_patches_list)
    visual_token_end = visual_token_start + total_visual_patches - 1
    visual_token_index = torch.tensor([visual_token_start, visual_token_end], dtype=torch.long, device=model.device)
    
    # Find language model
    language_model = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model
    
    # Detect model type
    model_type_name = type(language_model).__name__
    print(f"   🔍 Language model type: {model_type_name}")
    
    # Monkey patch InternLM2Model.forward to use local version (only for InternLM2Model)
    if 'InternLM2Model' in model_type_name or 'InternLM2' in model_type_name:
        import importlib.util
        modeling_path = os.path.join(current_dir, 'modeling_internlm2_old.py')
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
        except RuntimeError as e:
            # Chat failed, but hooks should have captured embeddings
            print(f"   ⚠️  Chat method failed (expected): {str(e)[:100]}")
            # Continue - we'll generate text manually if needed
    
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
    
    # Reshape visual_embeds if needed
    batch_size = text_embeds.shape[0]
    
    if len(visual_embeds.shape) == 3:
        # visual_embeds shape: [num_frames, patches_per_frame, hidden_dim] or [batch, seq_len, hidden_dim]
        if visual_embeds.shape[0] != batch_size:
            # Reshape from [num_frames, patches_per_frame, hidden_dim] to [batch, total_visual_tokens, hidden_dim]
            num_frames, patches_per_frame, hidden_dim = visual_embeds.shape
            total_visual_tokens = num_frames * patches_per_frame
            visual_embeds = visual_embeds.view(1, total_visual_tokens, hidden_dim)
            # Update visual_token_index based on actual visual tokens
            visual_token_start = visual_token_index[0].item()
            visual_token_end = visual_token_start + total_visual_tokens - 1
            visual_token_index = torch.tensor([visual_token_start, visual_token_end], dtype=torch.long, device=model.device)
            print(f"   📐 Reshaped visual_embeds: {visual_embeds.shape}")
            print(f"   📐 Updated visual_token_index: {visual_token_index.tolist()}")
    elif len(visual_embeds.shape) == 2:
        # Flattened: [batch*seq_len, hidden_dim] -> [batch, seq_len, hidden_dim]
        visual_seq_len = visual_embeds.shape[0] // batch_size
        visual_embeds = visual_embeds.view(batch_size, visual_seq_len, -1)
    
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
    with torch.no_grad():
        if is_internlm2:
            # InternLM2Model supports visual_token_index
            outputs = language_model.forward(
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                visual_token_index=visual_token_index,
                return_dict=True
            )
            # Get aggregated attention from model output
            aggregated_attention = outputs.aggregated_viusal_token_attention
        else:
            # For other models (Phi3), extract attention manually
            print(f"   🔍 Using manual attention extraction for {model_type_name}")
            outputs = language_model.forward(
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                return_dict=True
            )
            
            # Extract attention from all layers and aggregate
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # attentions is a tuple of tensors, one per layer
                # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
                all_attentions = outputs.attentions
                print(f"   📊 Found {len(all_attentions)} attention layers")
                
                # Use last layer attention (most relevant)
                last_layer_attn = all_attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
                
                # Average over heads: [batch_size, num_heads, seq_len, seq_len] -> [batch_size, seq_len, seq_len]
                if len(last_layer_attn.shape) == 4:
                    last_layer_attn = last_layer_attn.mean(dim=1)  # Average over heads
                
                # Extract attention from text tokens to visual tokens
                # Get visual token indices
                visual_start = visual_token_index[0].item()
                visual_end = visual_token_index[1].item() + 1
                seq_len = last_layer_attn.shape[-1]
                
                # Attention matrix: attention[i, j] = attention from token i to token j
                # We want attention from all tokens (especially text tokens) to visual tokens
                # Shape: [batch_size, seq_len, visual_token_count]
                visual_attn = last_layer_attn[:, :, visual_start:visual_end]
                
                # Aggregate: sum attention from all tokens to each visual token
                # This gives us how much attention each visual token receives
                # Shape: [batch_size, visual_token_count]
                aggregated_attention = visual_attn.sum(dim=1).squeeze(0)  # Sum over source tokens, remove batch dim
                
                # Normalize to get relative attention (optional, but helps with visualization)
                if aggregated_attention.sum() > 0:
                    aggregated_attention = aggregated_attention / aggregated_attention.sum() * visual_attn.shape[1]
                
                print(f"   📊 Last layer attention shape: {last_layer_attn.shape}")
                print(f"   📊 Visual token attention shape: {aggregated_attention.shape}")
            else:
                print(f"   ⚠️  No attentions found in outputs")
                aggregated_attention = None
    
    if aggregated_attention is not None:
        print(f"   📊 Final aggregated attention shape: {aggregated_attention.shape}")
        print(f"   📊 Attention numel: {aggregated_attention.numel()}")
    
    # Set default text if generation failed
    if generated_text is None:
        generated_text = "[Generation failed, but attention extracted successfully]"
    
    return aggregated_attention, visual_token_index, generated_text


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
    aggregated_attention, visual_token_idx, generated_text = extract_attention_with_hook(
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
                
                output_path = os.path.join(output_dir, f"{video_file}_frame_{frame_idx:04d}_attention.png")
                visualize(
                    frame_rgb, frame_attn, output_path,
                    use_overlay=args.use_overlay,
                    overlay_threshold=args.overlay_threshold,
                    num_patches_h=num_patches_h,
                    num_patches_w=num_patches_w
                )
                print(f"   ✅ Saved frame {frame_idx} attention: {output_path}")
            
            cap.release()
    
    return generated_text, aggregated_attention


def process_model(args):
    print(f"🤖 Loading model: {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    
    # Force eager attention to enable attention extraction
    # Flash Attention doesn't support output_attentions=True
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        lang_model = model.language_model.model
        if hasattr(lang_model, 'config'):
            original_attn = lang_model.config.attn_implementation
            lang_model.config.attn_implementation = 'eager'
            print(f"   ✅ Changed attention from '{original_attn}' to 'eager' for attention extraction")
            # Rebuild layers with eager attention
            from internlm2.modeling_internlm2_old import INTERNLM2_ATTENTION_CLASSES
            device = next(lang_model.layers[0].attention.parameters()).device
            for idx, layer in enumerate(lang_model.layers):
                old_attn = layer.attention
                new_attn = INTERNLM2_ATTENTION_CLASSES['eager'](lang_model.config)
                new_attn.to(device)
                
                # Copy weights from old attention to new attention
                new_attn.wqkv.weight.data = old_attn.wqkv.weight.data.clone()
                if hasattr(old_attn.wqkv, 'bias') and old_attn.wqkv.bias is not None:
                    new_attn.wqkv.bias.data = old_attn.wqkv.bias.data.clone()
                new_attn.wo.weight.data = old_attn.wo.weight.data.clone()
                if hasattr(old_attn.wo, 'bias') and old_attn.wo.bias is not None:
                    new_attn.wo.bias.data = old_attn.wo.bias.data.clone()
                
                # Copy rotary embedding (it's a module, so we need to copy its state)
                if hasattr(old_attn, 'rotary_emb') and hasattr(new_attn, 'rotary_emb'):
                    # Copy rotary embedding buffers
                    if hasattr(old_attn.rotary_emb, 'inv_freq'):
                        new_attn.rotary_emb.inv_freq = old_attn.rotary_emb.inv_freq
                    if hasattr(old_attn.rotary_emb, 'cos_cached'):
                        new_attn.rotary_emb.cos_cached = old_attn.rotary_emb.cos_cached
                    if hasattr(old_attn.rotary_emb, 'sin_cached'):
                        new_attn.rotary_emb.sin_cached = old_attn.rotary_emb.sin_cached
                    if hasattr(old_attn.rotary_emb, 'max_seq_len_cached'):
                        new_attn.rotary_emb.max_seq_len_cached = old_attn.rotary_emb.max_seq_len_cached
                
                layer.attention = new_attn
            print(f"   ✅ Rebuilt {len(lang_model.layers)} layers with eager attention")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Resolve image size
    image_size = args.image_size
    if image_size is None:
            image_size = getattr(model.config, 'vision_tower_image_size', 448)
    args.image_size = image_size
    
    # Fixed question
    # question = "Translate the American Sign Language in this video to English. Pay close attention to the person's facial expressions and hand movement."
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
        generated_text, aggregated_attention = process_video(
                model, tokenizer, video_path, question, args,
            attention_dir, video_name
            )
            
        results.append({
        "video": video_name,
        "video_path": video_path,
            "model_output": generated_text,
            "has_attention": aggregated_attention is not None
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
    parser.add_argument("--model-path", type=str, required=True)
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
