#!/usr/bin/env python3
"""
Qwen2VL Evaluation Script for DailyMoth-70h Dataset
Modified to handle DailyMoth's subdirectory structure
"""

import os
import sys
import json
import torch
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import math
import cv2
import numpy as np

# Disable flash attention
os.environ["DISABLE_FLASH_ATTN"] = "1"
warnings.filterwarnings("ignore")

# Add the src directory to Python path
sys.path.append('/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def resolve_image_size(model, args):
    """
    Pick image size from (1) --image_size,
    (2) model config fallbacks,
    else default=1024.
    """
    # 1) explicit CLI
    if getattr(args, "image_size", None):
        try:
            return int(args.image_size)
        except Exception:
            pass

    # 2) model.config fallbacks
    if hasattr(model, "config"):
        for k in ("vision_tower_image_size", "image_size", "vision_config"):
            v = getattr(model.config, k, None)
            if v:
                try:
                    if isinstance(v, dict) and "image_size" in v:
                        return int(v["image_size"])
                    elif isinstance(v, int):
                        return int(v)
                except Exception:
                    pass

    # 3) default for Qwen2.5-VL
    return 1024

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_dailymoth_video_path(video_folder, video_filename):
    """
    Construct full video path for DailyMoth dataset.
    DailyMoth videos are organized in subdirectories: 00000/, 00001/, etc.
    
    Example: 
        video_filename: "0000004-2022.6.14.mp4"
        returns: video_folder/00000/0000004-2022.6.14.mp4
    """
    # Extract the subdirectory from the filename (first 5 characters)
    # "0000004-2022.6.14.mp4" -> "00000"
    subdir = video_filename[:5]
    return os.path.join(video_folder, subdir, video_filename)

def extract_and_save_frames(video_path, output_dir, video_filename, fps=12):
    """
    Extract frames from video at specified FPS and save them.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        video_filename: Name of the video file (for creating subdirectory)
        fps: Frames per second to extract (default: 12)
    
    Returns:
        List of saved frame paths, or None if extraction failed
    """
    try:
        # Create subdirectory for this video's frames
        video_name = os.path.splitext(video_filename)[0]
        frames_dir = os.path.join(output_dir, video_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return None
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(video_fps / fps))
        
        saved_frames = []
        frame_count = 0
        extracted_count = 0
        
        print(f"📹 Extracting frames from {video_filename}")
        print(f"   Video FPS: {video_fps:.2f}, Target FPS: {fps}, Interval: {frame_interval}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at specified interval
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{extracted_count:05d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Extracted {extracted_count} frames to {frames_dir}")
        return saved_frames
        
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        import traceback
        traceback.print_exc()
        return None

def eval_model(args):
    device = "cuda"
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please run on a GPU node.")
        return
    
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"🔍 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"🎬 Qwen2VL Evaluation on DailyMoth-70h Dataset")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Model Base: {args.model_base}")
    print(f"Video Folder: {args.video_folder}")
    print(f"Question File: {args.question_file}")
    print(f"Output Dir: {args.out_dir}")
    print()
    
    # Load the model using LoRA checkpoint
    print("🚀 Loading model...")
    try:
        # Load base model first
        from transformers import AutoConfig
        lora_cfg_pretrained = AutoConfig.from_pretrained(args.checkpoint_path)
        # Remove quantization config if present (not needed for inference)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            delattr(lora_cfg_pretrained, 'quantization_config')
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_base,
            dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            config=lora_cfg_pretrained
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_base, trust_remote_code=True)
        
        # Load non-LoRA trainable weights (vision tower + merger)
        non_lora_path = os.path.join(args.checkpoint_path, 'non_lora_state_dict.bin')
        if os.path.exists(non_lora_path):
            print("🔄 Loading non-LoRA trainable weights (vision tower + merger)...")
            non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
            # Clean up state dict keys
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
            print("✅ Loaded non-LoRA weights successfully!")
        else:
            print("⚠️  Warning: non_lora_state_dict.bin not found - vision tower and merger may not be trained")
        
        # Load LoRA weights from checkpoint
        print("🔄 Loading LoRA weights...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_path)
        
        model.eval()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    image_size = resolve_image_size(model, args)
    print(f"[qwen2vl_evaluation] Resolved vision image size: {image_size}")

    # Load test data
    if not os.path.exists(args.question_file):
        print(f"❌ Question file not found: {args.question_file}")
        return
    
    data_dict = json.load(open(args.question_file, 'r'))
    
    # Apply chunking if specified
    if args.num_chunks > 1:
        data_dict = get_chunk(data_dict, args.num_chunks, args.chunk_idx)
    
    # Limit the number of samples if specified
    if args.max_samples is not None:
        original_size = len(data_dict)
        data_dict = data_dict[:args.max_samples]
        print(f"Limited to {len(data_dict)} samples (from {original_size} total)")
    
    print(f"📋 Processing {len(data_dict)} samples")
    
    # Create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Create frames output directory if frame saving is enabled
    frames_output_dir = None
    if args.save_frames:
        frames_output_dir = os.path.join(args.out_dir, "extracted_frames")
        os.makedirs(frames_output_dir, exist_ok=True)
        print(f"🖼️  Frame extraction enabled - saving to: {frames_output_dir}")

    results = []
    references = []
    predictions = []
    
    print(f"🎬 Starting evaluation on {len(data_dict)} videos...")
    print("-" * 60)
    
    # Process each sample
    for source in tqdm(data_dict, desc="Processing videos"):
        try:
            # Prepare the prompt for ASL translation (same as reference)
            fq = "Translate the American Sign Language in this video to English."
            if 'video' in source:
                video_file = source["video"]
                
                # Use DailyMoth-specific path construction
                video = get_dailymoth_video_path(args.video_folder, video_file)
                
                # Debug: print video path (only for first few samples)
                if len(results) < 3:
                    print(f"🔍 Video path: {video}")
                    print(f"🔍 Video exists: {os.path.exists(video)}")
                
                # Get ground truth early for error handling
                first_answer = source['conversations'][1]['value']
                
                # Check if video exists
                if not os.path.exists(video):
                    print(f"⚠️  Video not found: {video}")
                    results.append({
                        "video": video_file,
                        "prompt": fq,
                        "model_output": f"ERROR: Video file not found",
                        "ground_truth": first_answer
                    })
                    continue
                
                # Extract and save frames if enabled
                if args.save_frames and frames_output_dir:
                    extracted_frames = extract_and_save_frames(
                        video, 
                        frames_output_dir, 
                        video_file, 
                        fps=args.video_fps
                    )
                    if extracted_frames:
                        print(f"📸 Saved {len(extracted_frames)} frames")
                
                # Use official Qwen2.5-VL video processing pattern - correct format
                # Add system message to match training setup
                conversation = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video, "fps": args.video_fps},  # Include fps in video content
                            {"type": "text", "text": fq},
                        ],
                    }
                ]
                
                # Process video using the official pattern
                try:
                    # Import process_vision_info helper from qwen_vl_utils
                    from qwen_vl_utils import process_vision_info
                    
                    # Clear GPU cache before processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Step 1: Apply chat template to get text
                    text = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    
                    # Step 2: Extract vision inputs
                    image_inputs, video_inputs, video_kwargs = process_vision_info(conversation, return_video_kwargs=True)
                    
                    # Step 3: Process with processor
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                        **video_kwargs
                    ).to(device)
                    
                except RuntimeError as e:
                    # Handle CUDA errors gracefully
                    if "CUDA" in str(e) or "device-side assert" in str(e):
                        print(f"❌ CUDA error processing video {video_file}: {e}")
                        print(f"⚠️  Skipping this video and clearing GPU cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        results.append({
                            "video": video_file,
                            "prompt": fq,
                            "model_output": f"ERROR: CUDA error - {str(e)[:100]}",
                            "ground_truth": first_answer
                        })
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"❌ Error in video processing: {e}")
                    print(f"Conversation structure: {conversation}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "video": video_file,
                        "prompt": fq,
                        "model_output": f"ERROR: {str(e)}",
                        "ground_truth": first_answer
                    })
                    continue
                
                # Generate response using official pattern with optimizations
                with torch.no_grad():
                    # Use mixed precision for faster inference
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                        output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                        
                        # Clear GPU cache for memory efficiency
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                print(f"Video: {video_file}")
                print(f"Ground truth: {first_answer}")
                print(f"Prediction: {output}")
                print("-" * 60)
                
                # Store references and predictions for evaluation
                references.append(first_answer)
                predictions.append(output)
                
                results.append({
                    "video": video_file,
                    "prompt": fq,
                    "model_output": output,
                    "ground_truth": first_answer
                })
                
            else:
                print(f"⚠️  No video found in sample: {source}")
                continue
                
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "video": source.get("video", "unknown"),
                "prompt": fq,
                "model_output": f"ERROR: {str(e)}",
                "ground_truth": source.get('conversations', [{}])[1].get('value', 'unknown') if len(source.get('conversations', [])) > 1 else 'unknown'
            })
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"qwen2vl_dailymoth_results_{timestamp}.json"
    output_path = os.path.join(args.out_dir, output_file)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ All outputs saved successfully!")
    print(f"📄 Results saved to: {output_path}")
    
    # Calculate evaluation metrics
    if args.enable_evaluation and references and predictions:
        try:
            print("\n📊 Calculating evaluation metrics...")
            
            # Import evaluation metrics
            sys.path.append('/home/mh2803/projects/sign_language_llm/evaluation')
            from ssvp_evaluation import comprehensive_evaluation, print_evaluation_results, save_evaluation_results
            
            # Use comprehensive evaluation
            eval_results = comprehensive_evaluation(references, predictions)
            
            # Print results
            print_evaluation_results(eval_results, "Qwen2VL-DailyMoth")
            
            # Save evaluation metrics
            eval_file = os.path.join(args.out_dir, f"evaluation_metrics_{timestamp}.json")
            save_evaluation_results(eval_results, eval_file, "Qwen2VL-DailyMoth")
            
            print(f"📊 Evaluation metrics saved to: {eval_file}")
            
        except Exception as e:
            print(f"⚠️  Error calculating evaluation metrics: {e}")
            print("Continuing without evaluation metrics...")
    else:
        print("⚠️  No valid references and predictions found for evaluation.")
    
    # Print summary
    if len(results) > 0:
        successful = len([r for r in results if not r['model_output'].startswith('ERROR')])
        success_rate = (successful / len(results)) * 100
        print(f"\n🎯 Summary:")
        print(f"   Total samples: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Success rate: {success_rate:.1f}%")
    else:
        print("⚠️  No results processed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, 
                       default="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/qwen2vl_ssvp_2xa100_12fps_diverse/checkpoint-4000",
                       help="Path to LoRA checkpoint")
    parser.add_argument("--model-base", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Base model to load")
    parser.add_argument("--video-folder", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/videos/",
                       help="Path to video folder")
    parser.add_argument("--question-file", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/vanshika/asl_test/test_ssvp.json",
                       help="Path to question file")
    parser.add_argument("--out-dir", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/outputs/dailymoth/",
                       help="Output directory")
    parser.add_argument("--image-size", type=int, default=None,
                       help="Force vision image size (overrides configs)")
    parser.add_argument("--num-chunks", type=int, default=1,
                       help="Number of chunks to split data into")
    parser.add_argument("--chunk-idx", type=int, default=0,
                       help="Which chunk to process (0-indexed)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--enable-evaluation", action="store_true", default=True,
                       help="Enable evaluation metrics calculation")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--save-frames", action="store_true", default=False,
                       help="Extract and save video frames before processing")
    parser.add_argument("--video-fps", type=int, default=12,
                       help="FPS for frame extraction (default: 12)")
    
    args = parser.parse_args()
    eval_model(args)

if __name__ == "__main__":
    main()

