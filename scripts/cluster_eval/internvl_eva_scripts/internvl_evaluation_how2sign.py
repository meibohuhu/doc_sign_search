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

# Add InternVL paths
sys.path.insert(0, '/home/mh2803/projects/sign_language_llm/InternVL/internvl_chat')
sys.path.insert(0, '/home/mh2803/projects/sign_language_llm/InternVL')

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

def load_trained_model(checkpoint_path, base_model_name="OpenGVLab/InternVL2-26B"):
    """
    Load InternVL trained model from checkpoint
    """
    print("🚀 Loading InternVL model from checkpoint...")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Step 1: Load base model config
    print("\n1️⃣ Loading base model config...")
    config = InternVLChatConfig.from_pretrained(base_model_name, trust_remote_code=True)
    print("   ✅ Config loaded")
    
    # Step 2: Load model
    print(f"\n2️⃣ Loading model from checkpoint...")
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
        # Fallback to base model if checkpoint loading fails
        model = InternVLChatModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        print("   ✅ Model loaded from base model")
    
    # Step 3: Load tokenizer
    print(f"\n3️⃣ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")) 
        else base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    print("   ✅ Tokenizer loaded")
    
    model.eval()
    
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
    
    # Load model
    try:
        model, tokenizer = load_trained_model(
            args.checkpoint_path, 
            args.model_base
        )
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test data (support both JSON and JSONL formats)
    print(f"📂 Loading test data from: {args.question_file}")
    data_dict = []
    
    # Check if file is JSONL (each line is a JSON object) or JSON (array)
    with open(args.question_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)  # Reset to beginning
        
        if first_line.startswith('['):
            # JSON array format (qwenvl style)
            data_dict = json.load(f)
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
    
    os.makedirs(args.out_dir, exist_ok=True)
    
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
            
            # Extract question and ground truth based on format
            conversations = source.get('conversations', [])
            if len(conversations) >= 2:
                # InternVL format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
                if "from" in conversations[0]:
                    # InternVL format
                    question_text = conversations[0].get('value', '')
                    # Extract question without <video> tag if present
                    if '<video>' in question_text:
                        fq = question_text.split('<video>', 1)[-1].strip()
                        if not fq:
                            fq = "Translate the American Sign Language in this video to English."
                    else:
                        fq = question_text if question_text else "Translate the American Sign Language in this video to English."
                    ground_truth = conversations[1].get('value', '')
                else:
                    # qwenvl format: [{"role": "user", ...}, {"role": "assistant", ...}]
                    fq = "Translate the American Sign Language in this video to English."
                    ground_truth = conversations[1].get('value', '')
            else:
                fq = "Translate the American Sign Language in this video to English."
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
                image_list = _load_video_locally(
                    video_path,
                    max_num_frames=args.max_num_frames,
                    min_num_frames=args.min_num_frames,
                    sample=args.sampling_method,
                    clip=None
                )
            except Exception as e:
                print(f"\n⚠️  [{idx}/{len(data_dict)}] Failed to load video {video_file}: {e}")
                results.append({
                    "video": video_file,
                    "model_output": f"ERROR: Failed to load video - {str(e)}",
                    "ground_truth": ground_truth
                })
                continue
            
            # Generate special tokens for each video frame (same as training code)
            # Format: Frame-1: <image>\nFrame-2: <image>\n...
            special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
            
            # Prepare question with video frames (same format as training)
            question = f"{special_tokens}\n{fq}"
            
            # Transform each frame image and stack them (same as training code)
            pixel_values = [transform(image) for image in image_list]
            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
            
            # Generate using model.chat() (same as InternVL evaluation)
            generation_config = dict(
                num_beams=5,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                do_sample=True,
                temperature=0.7,
            )
            
            with torch.no_grad():
                output = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values.unsqueeze(0),
                    question=question,
                    generation_config=generation_config,
                    verbose=False
                )
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            sys.path.append('/home/mh2803/projects/sign_language_llm/evaluation')
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
    print(f"   Success rate: {successful/len(results)*100:.1f}%")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate InternVL trained checkpoint on test set")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--model-base", type=str, 
                       default="OpenGVLab/InternVL2-26B",
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
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint_path):
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

