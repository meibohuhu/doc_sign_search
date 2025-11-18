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
from scripts.visualize_attention_simple_correct import (
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
    
    # Prepare prompt - use the same format as model.chat() internally uses
    # InternVL2 chat() method will add video_prefix internally, so we pass question directly
    # But for manual tokenization, we need to build the full prompt with chat template
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    full_question = video_prefix + question
    
    # Use build_inputs to get the correct chat template format (same as model.chat() uses)
    # This ensures prompt format consistency
    if hasattr(model.language_model, 'build_inputs'):
        # Use the same build_inputs method that chat() uses
        inputs = model.language_model.build_inputs(tokenizer, full_question, history=[], meta_instruction='')
        input_ids = inputs['input_ids'].to(model.device)
    else:
        # Fallback: manual tokenization (may not match chat() format exactly)
    inputs = tokenizer(full_question, return_tensors='pt').to(model.device)
    input_ids = inputs.input_ids
    
    # Find <image> token positions in the tokenized input
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
    
    # Find InternLM2Model (but don't monkey-patch yet!)
    language_model = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model
    
    # Step 1: Use hook to capture embeddings from model.chat() call
    # CRITICAL: Do NOT monkey-patch before model.chat() - use original forward for generation
    intercepted_visual_embeds = []
    intercepted_text_embeds = [None]
    
    # Find projector
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
    
    # Hook text embeddings
    def text_emb_hook(module, input, output):
        intercepted_text_embeds[0] = output.detach().clone()
        print(f"   ✅ Hooked text embeddings: {output.shape}")
    
    handle_text_emb = language_model.tok_embeddings.register_forward_hook(text_emb_hook)
    handles = [handle_projector, handle_text_emb]
    
    # Step 2: Call model.chat() FIRST (using original forward, not monkey-patched)
    # This ensures correct generation with proper position encoding, KV cache, and final norm
    generation_config = {
        'max_new_tokens': 128,
        'do_sample': False,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }
    
    generated_text = None
    with torch.no_grad():
        try:
            # Pass question directly (model.chat() will add video_prefix internally)
            # This uses the ORIGINAL forward method (not monkey-patched)
            response, _ = model.chat(
                        tokenizer,
                        pixel_values,
                        question,  # Pass question directly, let model.chat() handle video_prefix
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                        return_history=False
                    )
                    generated_text = response.strip()
            print(f"   ✅ Chat method succeeded, generated: {generated_text[:100]}")
        except RuntimeError as e:
            print(f"   ⚠️  Chat method failed (expected): {str(e)[:100]}")
    
    # Remove hooks after model.chat() call
    for handle in handles:
        handle.remove()
    
    # Step 3: NOW monkey-patch InternLM2Model.forward to use local version for attention extraction
    # This is done AFTER generation, so generation uses the original forward
    import importlib.util
    modeling_path = os.path.join(current_dir, 'modeling_internlm2_old.py')
    spec = importlib.util.spec_from_file_location("internlm2.modeling_internlm2_local", modeling_path)
    local_modeling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_modeling)
    LocalInternLM2Model = local_modeling.InternLM2Model
    language_model.forward = lambda *args, **kwargs: LocalInternLM2Model.forward(language_model, *args, **kwargs)
    print(f"   ✅ Monkey-patched language_model.forward AFTER generation")
    
    # Get captured embeddings
    if intercepted_text_embeds[0] is None or len(intercepted_visual_embeds) == 0:
        raise ValueError("Failed to capture text or visual embeddings from hooks")
    
    text_embeds = intercepted_text_embeds[0]
    visual_embeds = intercepted_visual_embeds[-1]  # Use the last one
    
    print(f"   📐 Hooked text embeds shape: {text_embeds.shape}")
    print(f"   📐 Hooked visual embeds shape: {visual_embeds.shape}")
    
    # Reshape visual_embeds if needed
    batch_size = text_embeds.shape[0]
    text_seq_len = text_embeds.shape[1]
    
    # Handle different shapes of visual_embeds from hook
    if len(visual_embeds.shape) == 3:
        # Shape: [num_frames, patches_per_frame, hidden_dim] or [batch, seq_len, hidden_dim]
        if visual_embeds.shape[0] != batch_size:
            # Reshape from [num_frames, patches_per_frame, hidden_dim] to [batch, total_visual_tokens, hidden_dim]
            num_frames, patches_per_frame, hidden_dim = visual_embeds.shape
            total_visual_tokens = num_frames * patches_per_frame
            visual_embeds = visual_embeds.view(1, total_visual_tokens, hidden_dim)
            print(f"   📐 Reshaped visual_embeds from [num_frames, patches, hidden] to [1, {total_visual_tokens}, {hidden_dim}]")
        else:
            total_visual_tokens = visual_embeds.shape[1]
    elif len(visual_embeds.shape) == 2:
        # Flattened: [batch*seq_len, hidden_dim] -> [batch, seq_len, hidden_dim]
        visual_seq_len = visual_embeds.shape[0] // batch_size
        visual_embeds = visual_embeds.view(batch_size, visual_seq_len, -1)
        total_visual_tokens = visual_seq_len
    else:
        total_visual_tokens = sum(num_patches_list)
    
    # Ensure visual_embeds has correct batch size
    if visual_embeds.shape[0] != batch_size:
        visual_embeds = visual_embeds[:batch_size]
    
    # CRITICAL: Recalculate visual_token_index based on hook-captured text_embeds
    # The hook-captured text_embeds correspond to model.chat()'s internal prompt format
    # We need to rebuild the prompt the same way model.chat() does to find <image> token positions
    # InternVLChatModel.chat() adds video_prefix internally, so we need to match that format
    
    # Rebuild prompt the same way model.chat() does (with video_prefix)
    video_prefix_for_chat = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    full_question_for_chat = video_prefix_for_chat + question
    
    # Tokenize using build_inputs to match model.chat()'s internal format
    if hasattr(model.language_model, 'build_inputs'):
        chat_inputs = model.language_model.build_inputs(tokenizer, full_question_for_chat, history=[], meta_instruction='')
        chat_input_ids = chat_inputs['input_ids'].to(model.device)
    else:
        chat_inputs = tokenizer(full_question_for_chat, return_tensors='pt').to(model.device)
        chat_input_ids = chat_inputs.input_ids
    
    print(f"   📐 Hook-captured text_seq_len: {text_seq_len}, chat_input_ids length: {chat_input_ids.shape[1]}")
    
    # Use chat_input_ids to find <image> token positions (this matches model.chat()'s internal format)
    if image_token_id is not None:
        image_positions = (chat_input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(image_positions) > 0:
            visual_token_start = image_positions[0].item()
            print(f"   📐 Found <image> token at position {visual_token_start} in chat_input_ids")
        else:
            # No <image> tokens found, append visual tokens at the end
            visual_token_start = chat_input_ids.shape[1]
            print(f"   ⚠️  No <image> tokens found, appending visual tokens at position {visual_token_start}")
    else:
        visual_token_start = chat_input_ids.shape[1]
        print(f"   ⚠️  image_token_id is None, appending visual tokens at position {visual_token_start}")
    
    # Adjust visual_token_start if text_seq_len doesn't match chat_input_ids length
    if text_seq_len != chat_input_ids.shape[1]:
        print(f"   ⚠️  Text embeds length ({text_seq_len}) != chat_input_ids length ({chat_input_ids.shape[1]})")
        print(f"   💡 Scaling visual_token_start proportionally")
        scale_factor = text_seq_len / chat_input_ids.shape[1]
        visual_token_start = int(visual_token_start * scale_factor)
        print(f"   📐 Scaled visual_token_start to {visual_token_start}")
    
    visual_token_end = visual_token_start + total_visual_tokens - 1
    visual_token_index = torch.tensor([visual_token_start, visual_token_end], dtype=torch.long, device=model.device)
    print(f"   📐 Final visual_token_index: {visual_token_index.tolist()} (text_seq_len={text_seq_len}, total_visual_tokens={total_visual_tokens})")
    
    # Insert visual tokens into text embeddings at visual_token_index positions
    inputs_embeds = _insert_visual_tokens(text_embeds, visual_embeds, visual_token_index)
    print(f"   ✅ Combined inputs_embeds shape: {inputs_embeds.shape}")
    
    # Debug: check inputs_embeds shape and visual_token_index
    print(f"   📐 inputs_embeds shape: {inputs_embeds.shape}")
    print(f"   📐 visual_token_index: {visual_token_index.tolist()}")
    print(f"   📐 text_seq_len (from hook): {text_seq_len}, input_ids length: {input_ids.shape[1]}")
    
    # Initialize aggregated_attention
    aggregated_attention = None
    
    # If generation failed, manually implement generation loop to accumulate attention
    if generated_text is None:
        print(f"   🔄 Attempting manual generation with attention accumulation...")
        try:
            # Get the language model wrapper (InternLM2ForCausalLM)
            causal_lm = model.language_model if hasattr(model, 'language_model') else None
            if causal_lm is None:
                raise ValueError("Could not find causal_lm model")
            
            # Manual generation loop
            with torch.no_grad():
                max_new_tokens = 128
                eos_token_id = tokenizer.eos_token_id
                pad_token_id = tokenizer.eos_token_id
                
                # Get im_end token ID
                try:
                    im_end_token_id = tokenizer.convert_tokens_to_ids(['<|im_end|>'])[0]
                except:
                    im_end_token_id = None
                
                # Initialize generation state
                # CRITICAL: Use chat_input_ids which matches hook-captured text_embeds
                # inputs_embeds has shape [batch, text_seq_len + total_visual_tokens, hidden_dim]
                # We need input_ids with length = text_seq_len (before visual tokens are inserted)
                # chat_input_ids corresponds to the prompt format used by model.chat()
                # which matches hook-captured text_embeds
                
                if text_seq_len == chat_input_ids.shape[1]:
                    # Lengths match, use chat_input_ids
                    generated_ids = chat_input_ids.clone()
                    print(f"   📊 Using chat_input_ids for generated_ids (length: {text_seq_len})")
                else:
                    # Lengths don't match, use chat_input_ids but adjust
                    print(f"   ⚠️  Text seq len ({text_seq_len}) != chat_input_ids length ({chat_input_ids.shape[1]})")
                    # Use chat_input_ids as base, but we'll adjust during decoding
                    generated_ids = chat_input_ids.clone()
                    print(f"   📊 Using chat_input_ids as base (will adjust during decoding)")
                
                past_key_values = None
                aggregated_attention = None
                
                # Step 1: First forward pass (non-generation case)
                # Extract attention from text tokens to visual tokens
                print(f"   📊 Step 1: First forward pass (text tokens → visual tokens)...")
                print(f"   📊 Using inputs_embeds with shape {inputs_embeds.shape}, visual_token_index: {visual_token_index.tolist()}")
                
                # Use language_model.forward() directly (it's monkey-patched to support visual_token_index)
                # CRITICAL: Only pass visual_token_index in the first forward pass
                first_outputs = language_model.forward(
                    inputs_embeds=inputs_embeds,
                    output_attentions=True,
                    visual_token_index=visual_token_index,  # ✅ Only in first forward pass
                    return_dict=True,
                    use_cache=True,
                )
                
                # Get logits from causal_lm's output layer (lm_head)
                # Use the correct logits calculation method
                if hasattr(causal_lm, 'lm_head'):
                    first_logits = causal_lm.lm_head(first_outputs.last_hidden_state)
                elif hasattr(causal_lm, 'output'):
                    first_logits = causal_lm.output(first_outputs.last_hidden_state)
                else:
                    raise ValueError("Could not find lm_head or output method in causal_lm")
                
                # Get attention from first forward pass
                if hasattr(first_outputs, 'aggregated_viusal_token_attention'):
                    aggregated_attention = first_outputs.aggregated_viusal_token_attention
                    if aggregated_attention is not None:
                        print(f"   ✅ First forward attention shape: {aggregated_attention.shape}")
                
                # Get logits and past_key_values for generation
                # The last_hidden_state includes both text and visual tokens
                # We want the logits for the last position (after processing all tokens)
                next_token_logits = first_logits[:, -1, :]  # [batch, vocab_size]
                past_key_values = first_outputs.past_key_values
                
                # Get position_ids from first_outputs if available, or create them
                # Position IDs should be continuous and increment for each new token
                if hasattr(first_outputs, 'position_ids') and first_outputs.position_ids is not None:
                    current_position_ids = first_outputs.position_ids
                else:
                    # Create position_ids based on sequence length
                    seq_len = inputs_embeds.shape[1]
                    current_position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
                
                print(f"   📊 Initial position_ids shape: {current_position_ids.shape}, last position: {current_position_ids[0, -1].item()}")
                
                # Generation loop
                print(f"   🔄 Starting generation loop (max {max_new_tokens} tokens)...")
                print(f"   📊 EOS token ID: {eos_token_id}, Pad token ID: {pad_token_id}, im_end token ID: {im_end_token_id}")
                for step in range(max_new_tokens):
                    # Get next token
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch, 1]
                    next_token_id = next_token.item()
                    
                    # Debug: print first few tokens
                    if step < 5:
                        print(f"   📊 Step {step}: next_token_id={next_token_id}, token='{tokenizer.decode([next_token_id], skip_special_tokens=False)}'")
                    
                    # Append to generated sequence FIRST (before checking stop conditions)
                    # This ensures we accumulate attention for the current token
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Step 2: Generation step (only new token, no visual_token_index)
                    # Prepare inputs for next step (only the new token)
                    # Get embeddings for the new token
                    next_token_embeds = language_model.tok_embeddings(next_token)  # [batch, 1, hidden_dim]
                    
                    # Update position_ids: increment by 1 for the new token
                    # Position IDs must be continuous and increment for each new token
                    next_position_id = current_position_ids[0, -1].item() + 1
                    next_position_ids = torch.tensor([[next_position_id]], dtype=torch.long, device=model.device)
                    
                    # Forward pass with past_key_values
                    # Use language_model.forward() directly (it's monkey-patched)
                    # CRITICAL: We still need to pass visual_token_index in generation steps
                    # to extract attention from generated tokens to visual tokens
                    # Visual tokens are in past_key_values, but their relative positions remain the same
                    # The visual_token_index refers to positions in the original sequence (before generation)
                    # In generation, the attention matrix shape is [batch, num_heads, 1, key_len]
                    # where key_len includes all tokens in past_key_values (including visual tokens)
                    # So we can still use the original visual_token_index to extract attention
                    step_outputs = language_model.forward(
                        inputs_embeds=next_token_embeds,
                        position_ids=next_position_ids,  # ✅ Pass correct position_ids (incremented)
                        output_attentions=True,
                        visual_token_index=visual_token_index,  # ✅ Pass visual_token_index to extract attention
                        past_key_values=past_key_values,  # ✅ Use KV cache from previous steps
                        return_dict=True,
                        use_cache=True,
                    )
                    
                    # Get logits from causal_lm's output layer (lm_head)
                    # Use the correct logits calculation method
                    if hasattr(causal_lm, 'lm_head'):
                        step_logits = causal_lm.lm_head(step_outputs.last_hidden_state)
                    elif hasattr(causal_lm, 'output'):
                        step_logits = causal_lm.output(step_outputs.last_hidden_state)
                    else:
                        raise ValueError("Could not find lm_head or output method in causal_lm")
                    
                    # Accumulate attention from generation steps (generated tokens → visual tokens)
                    # This is important to capture how each generated token attends to visual tokens
                    if hasattr(step_outputs, 'aggregated_viusal_token_attention'):
                        step_attention = step_outputs.aggregated_viusal_token_attention
                        if step_attention is not None:
                            if aggregated_attention is None:
                                aggregated_attention = step_attention.clone()
                            else:
                                aggregated_attention = aggregated_attention + step_attention
                            if step < 3:
                                print(f"   ✅ Step {step} attention accumulated, shape: {step_attention.shape}")
                    
                    # Update for next iteration
                    next_token_logits = step_logits[:, -1, :]  # [batch, vocab_size]
                    past_key_values = step_outputs.past_key_values  # ✅ Update KV cache
                    current_position_ids = next_position_ids  # ✅ Update position_ids for next iteration
                    
                    if step < 3:
                        print(f"   📊 Step {step}: position_id={next_position_id}, logits shape={step_logits.shape}")
                    
                    # NOW check for stop conditions (after accumulating attention)
                    should_stop = False
                    if next_token_id == eos_token_id:
                        if step == 0:
                            print(f"   ⚠️  EOS token reached at first step (step {step}), continuing anyway...")
                            # Don't break on first EOS, might be a false positive
                        else:
                            print(f"   ✅ EOS token reached at step {step}")
                            should_stop = True
                    
                    # Check for im_end token
                    if im_end_token_id is not None and next_token_id == im_end_token_id:
                        print(f"   ✅ <|im_end|> token reached at step {step}")
                        should_stop = True
                    
                    if should_stop:
                        break
                    
                    if (step + 1) % 10 == 0:
                        print(f"   📊 Generated {step + 1} tokens, attention accumulated")
                
                # Decode generated text
                # Extract only the newly generated tokens (after the original text sequence)
                # Use chat_input_ids.shape[1] since generated_ids is based on chat_input_ids
                # This ensures we extract tokens after the prompt (including video_prefix)
                prompt_length = chat_input_ids.shape[1]
                generated_tokens = generated_ids[0, prompt_length:]
                print(f"   📊 Generated {len(generated_tokens)} tokens (prompt length: {prompt_length})")
                print(f"   📊 Generated token IDs: {generated_tokens.tolist()[:20]}")  # Show first 20 tokens
                
                # Check if EOS or im_end appeared early
                eos_positions = (generated_tokens == eos_token_id).nonzero(as_tuple=True)[0]
                im_end_positions = []
                if im_end_token_id is not None:
                    im_end_positions = (generated_tokens == im_end_token_id).nonzero(as_tuple=True)[0]
                
                # Find the first stopping token position
                stop_positions = []
                if len(eos_positions) > 0:
                    stop_positions.append(eos_positions[0].item())
                if len(im_end_positions) > 0:
                    stop_positions.append(im_end_positions[0].item())
                
                if len(stop_positions) > 0:
                    first_stop = min(stop_positions)
                    print(f"   ⚠️  Stopping token found at position: {first_stop}")
                    # Remove tokens after the first stopping token
                    generated_tokens = generated_tokens[:first_stop]
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                if aggregated_attention is not None:
                    print(f"   ✅ Manual generation successful!")
                    print(f"   📊 Final aggregated attention shape: {aggregated_attention.shape}")
                    print(f"   📝 Generated text ({len(generated_text)} chars): {generated_text[:100]}")
                else:
                    print(f"   ⚠️  Manual generation successful but no attention accumulated")
        
        except Exception as e:
            print(f"   ⚠️  Manual generation with inputs_embeds failed: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            
            # Fallback: try simple generation without visual tokens
            try:
                print(f"   🔄 Trying text-only generation as fallback...")
                from transformers import GenerationConfig
                gen_config = GenerationConfig(
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Just use text embeddings for generation
                with torch.no_grad():
                    text_only_ids = causal_lm.generate(
                        input_ids=input_ids,
                        generation_config=gen_config,
                    )
                    generated_text = tokenizer.decode(text_only_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                    print(f"   ✅ Text-only generation successful: {generated_text[:100]}")
            except Exception as e2:
                print(f"   ⚠️  Text-only generation also failed: {str(e2)[:200]}")
                generated_text = "[Generation failed, but attention extracted successfully]"
                # Extract attention from the initial forward pass (non-generation case)
                if aggregated_attention is None:
                    print(f"   🔄 Extracting attention from initial forward pass...")
                    with torch.no_grad():
                        outputs = language_model.forward(
                            inputs_embeds=inputs_embeds,
                            output_attentions=True,
                            visual_token_index=visual_token_index,
                            return_dict=True
                        )
                        aggregated_attention = outputs.aggregated_viusal_token_attention
                        if aggregated_attention is not None:
                            print(f"   ✅ Extracted attention: {aggregated_attention.shape}")
    
    # If generation succeeded but we don't have attention yet, extract it
    if aggregated_attention is None and generated_text is not None and "[Generation failed" not in generated_text:
        print(f"   🔄 Generation succeeded but no attention, extracting from forward pass...")
        with torch.no_grad():
            outputs = language_model.forward(
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                visual_token_index=visual_token_index,
                return_dict=True
            )
            aggregated_attention = outputs.aggregated_viusal_token_attention
            if aggregated_attention is not None:
                print(f"   ✅ Extracted attention: {aggregated_attention.shape}")
    
    if aggregated_attention is not None:
        print(f"   📊 Final attention shape: {aggregated_attention.shape}")
        print(f"   📊 Final attention numel: {aggregated_attention.numel()}")
    
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
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Resolve image size
    image_size = args.image_size
    if image_size is None:
            image_size = getattr(model.config, 'vision_tower_image_size', 448)
    args.image_size = image_size
    
    # Fixed question
    # question = "Translate the American Sign Language in this video to English. Pay close attention to the person's facial expressions and hand movement."
    # question = "Translate the American Sign Language in this video to English."
    # question = "How many people are in the video?"
    # question = "What does this person do in the video?"
    # question = "How many items are in the video?"
    question = "What's the color of the background?"
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
