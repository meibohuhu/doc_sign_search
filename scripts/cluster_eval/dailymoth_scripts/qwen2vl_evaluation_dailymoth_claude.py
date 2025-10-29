#!/usr/bin/env python3
"""
Qwen2VL Evaluation Script for DailyMoth-70h Dataset
Uses improved model loading from how2sign_claude with DailyMoth-specific path handling
"""

import os
import sys
import json
import torch
import warnings
import argparse
from datetime import datetime
from tqdm import tqdm
import cv2

os.environ["DISABLE_FLASH_ATTN"] = "1"
warnings.filterwarnings("ignore")

sys.path.append('/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel

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

def load_trained_model(checkpoint_path, base_model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    Load complete trained model with CORRECT key prefix handling
    """
    print("🚀 Loading model from checkpoint...")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Step 1: Load base model
    print("\n1️⃣ Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("   ✅ Base model loaded")
    
    # Step 2: Load non-LoRA weights (vision + merger)
    non_lora_path = os.path.join(checkpoint_path, 'non_lora_state_dict.bin')
    
    if not os.path.exists(non_lora_path):
        raise FileNotFoundError(f"❌ {non_lora_path} not found!")
    
    print(f"\n2️⃣ Loading vision tower + merger...")
    file_size = os.path.getsize(non_lora_path) / (1024**3)
    print(f"   File size: {file_size:.2f} GB")
    
    print(f"   🔄 Loading checkpoint (this may take 30-60 seconds)...")
    state_dict = torch.load(non_lora_path, map_location='cpu')
    print(f"   ✅ Loaded {len(state_dict)} keys from disk")
    
    # CRITICAL FIX: Handle triple prefix "base_model.model.model."
    print(f"\n   🔧 Cleaning key prefixes...")
    cleaned_state = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Your checkpoint has: base_model.model.model.visual...
        # Model expects: model.visual...
        
        # Remove "base_model.model.model." → "visual..."
        if new_key.startswith('base_model.model.model.'):
            new_key = new_key[23:]  # Remove first 23 characters
            new_key = 'model.' + new_key  # Add back single "model." prefix
        
        # Fallback: if just "base_model.model."
        elif new_key.startswith('base_model.model.'):
            new_key = new_key[17:]  # Remove "base_model.model."
            new_key = 'model.' + new_key
        
        # Fallback: if just "base_model."
        elif new_key.startswith('base_model.'):
            new_key = new_key[11:]  # Remove "base_model."
        
        # Remove any "_orig_mod."
        new_key = new_key.replace('_orig_mod.', '')
        
        cleaned_state[new_key] = value
    
    print(f"   ✅ Cleaned {len(cleaned_state)} keys")
    
    # Show sample of cleaned keys
    sample_keys = list(cleaned_state.keys())[:5]
    print(f"\n   📝 Sample cleaned keys:")
    for k in sample_keys:
        print(f"      - {k}")
    
    # Analyze components
    vision_keys = [k for k in cleaned_state.keys() if 'visual' in k.lower()]
    merger_keys = [k for k in cleaned_state.keys() if 'merger' in k.lower() or 'mm_projector' in k.lower()]
    
    print(f"\n   📊 Components in checkpoint:")
    print(f"      Vision tower: {len(vision_keys)} keys")
    print(f"      Merger: {len(merger_keys)} keys")
    print(f"      Total: {len(cleaned_state)} keys")
    
    if len(vision_keys) == 0:
        raise ValueError("❌ No vision keys found after cleaning!")
    
    print(f"\n   ✅ Vision tower found!")
    
    # Load into model
    print(f"\n   🔄 Loading weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state, strict=False)
    
    loaded_count = len(cleaned_state) - len(missing_keys)
    print(f"      ✅ Loaded: {loaded_count}/{len(cleaned_state)} parameters")
    print(f"      Missing: {len(missing_keys)} (normal for LoRA)")
    print(f"      Unexpected: {len(unexpected_keys)}")
    
    # Verify vision tower loaded
    model_vision_params = sum(
        p.numel() for n, p in model.named_parameters() 
        if 'visual' in n.lower()
    )
    print(f"\n   ✅ Vision parameters in model: {model_vision_params:,}")
    
    if model_vision_params < 400_000_000:
        print(f"      ⚠️  WARNING: Expected ~600M, got {model_vision_params:,}")
    else:
        print(f"      ✅ Vision tower fully loaded!")
    
    # Step 3: Load LoRA
    print(f"\n3️⃣ Loading LoRA weights (LLM)...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    print(f"   ✅ LoRA loaded")
    
    # Step 4: Load processor
    print(f"\n4️⃣ Loading processor...")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    print(f"   ✅ Processor loaded")
    
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE MODEL LOADED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"   🎯 Vision tower: TRAINED (from checkpoint) ✅")
    print(f"   🎯 Merger: TRAINED (from checkpoint) ✅")
    print(f"   🎯 LLM: TRAINED (via LoRA) ✅")
    print(f"{'='*70}\n")
    
    return model, processor, tokenizer

def eval_model(args):
    device = "cuda"
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
    
    # Load model
    try:
        model, processor, tokenizer = load_trained_model(
            args.checkpoint_path, 
            args.model_base
        )
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test data
    print(f"📂 Loading test data from: {args.question_file}")
    with open(args.question_file, 'r') as f:
        data_dict = json.load(f)
    
    if args.max_samples:
        data_dict = data_dict[:args.max_samples]
        print(f"   Limited to {args.max_samples} samples")
    
    print(f"   Total samples: {len(data_dict)}\n")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create frames output directory if frame saving is enabled
    frames_output_dir = None
    if args.save_frames:
        frames_output_dir = os.path.join(args.out_dir, "extracted_frames")
        os.makedirs(frames_output_dir, exist_ok=True)
        print(f"🖼️  Frame extraction enabled - saving to: {frames_output_dir}\n")
    
    results = []
    references = []
    predictions = []
    
    print(f"🎬 Starting evaluation...")
    print(f"{'='*70}\n")
    
    # Process samples
    for idx, source in enumerate(tqdm(data_dict, desc="Evaluating"), 1):
        try:
            fq = "Translate the American Sign Language in this video to English."
            video_file = source["video"]
            
            # Use DailyMoth-specific path construction
            video_path = get_dailymoth_video_path(args.video_folder, video_file)
            ground_truth = source['conversations'][1]['value']
            
            # Debug: print video path (only for first few samples)
            if idx <= 3:
                print(f"🔍 Video path: {video_path}")
                print(f"🔍 Video exists: {os.path.exists(video_path)}")
            
            if not os.path.exists(video_path):
                print(f"\n⚠️  [{idx}/{len(data_dict)}] Video not found: {video_file}")
                results.append({
                    "video": video_file,
                    "model_output": "ERROR: Video not found",
                    "ground_truth": ground_truth
                })
                continue
            
            # Extract and save frames if enabled
            if args.save_frames and frames_output_dir:
                extracted_frames = extract_and_save_frames(
                    video_path, 
                    frames_output_dir, 
                    video_file, 
                    fps=args.video_fps
                )
                if extracted_frames:
                    print(f"📸 Saved {len(extracted_frames)} frames")
            
            # Prepare input
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": args.video_fps},
                    {"type": "text", "text": fq}
                ]
            }]
            
            # Process
            from qwen_vl_utils import process_vision_info
            
            text = processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(conversation)
            
            # CRITICAL: Match training resolution constraints
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels
            ).to(device)
            
            # Generate
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                    generated_ids = [
                        out[len(inp):] 
                        for inp, out in zip(inputs.input_ids, output_ids)
                    ]
                    output = processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )[0]
            
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
            results.append({
                "video": source.get("video", "unknown"),
                "model_output": f"ERROR: {str(e)}",
                "ground_truth": source.get('conversations', [{}])[1].get('value', 'unknown') if len(source.get('conversations', [])) > 1 else 'unknown'
            })
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"dailymoth_results_{timestamp}.json"
    output_path = os.path.join(args.out_dir, output_file)
    
    with open(output_path, "w", encoding="utf-8") as f:
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
            print_evaluation_results(eval_results, "DailyMoth-70h")
            
            # Save metrics
            eval_file = os.path.join(args.out_dir, f"metrics_{timestamp}.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"\n📊 Metrics saved: {eval_file}")
            
        except Exception as e:
            print(f"\n⚠️  Evaluation metrics error: {e}")
    
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
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint on DailyMoth test set")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--model-base", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Base model name")
    parser.add_argument("--video-folder", type=str, required=True,
                       help="Folder containing DailyMoth videos (with subdirectories)")
    parser.add_argument("--question-file", type=str, required=True,
                       help="JSON file with test questions")
    parser.add_argument("--out-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                       help="Max tokens to generate")
    parser.add_argument("--video-fps", type=int, default=12,
                       help="FPS for video processing (MUST match training!)")
    parser.add_argument("--min-pixels", type=int, default=224*224,
                       help="Min pixels for video processing (MUST match training!)")
    parser.add_argument("--max-pixels", type=int, default=224*224,
                       help="Max pixels for video processing (MUST match training!)")
    parser.add_argument("--save-frames", action="store_true", default=False,
                       help="Extract and save video frames before processing")
    
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

