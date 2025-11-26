#!/usr/bin/env python3
"""
Qwen2.5-VL-3B-Instruct Base Model Evaluation Script for How2Sign Dataset
Evaluates pretrained model WITHOUT fine-tuning as baseline comparison
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

def eval_model(args):
    device = "cuda"
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please run on a GPU node.")
        return
    
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"🔍 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"🎬 Qwen2.5-VL-3B-Instruct BASE Model Evaluation on How2Sign")
    print("=" * 70)
    print(f"Model: {args.model_base}")
    print(f"Video Folder: {args.video_folder}")
    print(f"Question File: {args.question_file}")
    print(f"Output Dir: {args.out_dir}")
    print()
    
    # Load the BASE model (no fine-tuning)
    print("🚀 Loading BASE model (pretrained, no fine-tuning)...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_base,
            dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_base, trust_remote_code=True)
        
        model.eval()
        print("✅ Base model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    image_size = resolve_image_size(model, args)
    print(f"[qwen2vl_base_evaluation] Resolved vision image size: {image_size}")

    # Load test data
    if not os.path.exists(args.question_file):
        print(f"❌ Question file not found: {args.question_file}")
        return
    
    with open(args.question_file, 'r') as f:
        data_dict = json.load(f)
    
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

    results = []
    references = []
    predictions = []
    
    print(f"🎬 Starting evaluation on {len(data_dict)} videos...")
    print("-" * 60)
    
    # Process each sample
    for source in tqdm(data_dict, desc="Processing videos"):
        try:
            # Prepare the prompt for ASL translation (same as fine-tuned model)
            fq = "Translate the American Sign Language in this video to English."
            # fq = "Translate the ASL signs in this video to English text. Provide only the English translation without describing the person, gestures, or video content. Answer in one sentence only. If you cannot determine the meaning, RESPOND with nothing."
            if 'video' in source:
                video_file = source["video"]
                video = os.path.join(args.video_folder, video_file)
                
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
                
                # Use official Qwen2.5-VL video processing pattern
                # Add system message to match training setup
                conversation = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video, "fps": args.video_fps},
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
                        output_ids = model.generate(
                            **inputs,
                            num_beams=5,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=50,
                            length_penalty=1.0,
                            no_repeat_ngram_size=4,
                            repetition_penalty=1.1,
                            min_length=1,
                            max_new_tokens=args.max_new_tokens,
                        )
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
    output_file = f"qwen2vl_base_how2sign_results_{timestamp}.json"
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
            print_evaluation_results(eval_results, "Qwen2VL-BASE-How2Sign")
            
            # Save evaluation metrics
            eval_file = os.path.join(args.out_dir, f"base_evaluation_metrics_{timestamp}.json")
            save_evaluation_results(eval_results, eval_file, "Qwen2VL-BASE-How2Sign")
            
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
    parser.add_argument("--model-base", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Base model to load")
    parser.add_argument("--video-folder", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/",
                       help="Path to video folder")
    parser.add_argument("--question-file", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/vanshika/asl_test/segmented_videos.json",
                       help="Path to question file")
    parser.add_argument("--out-dir", type=str, 
                       default="/home/mh2803/projects/sign_language_llm/outputs/how2sign_base/",
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
    parser.add_argument("--video-fps", type=int, default=12,
                       help="FPS for video processing (default: 12)")
    
    args = parser.parse_args()
    eval_model(args)

if __name__ == "__main__":
    main()

