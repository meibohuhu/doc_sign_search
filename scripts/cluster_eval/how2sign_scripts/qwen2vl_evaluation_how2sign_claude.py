#!/usr/bin/env python3
"""
CORRECTED Evaluation Script for checkpoint-4000
Handles triple prefix: base_model.model.model.visual...
"""

import os
import sys
import json
import torch
import warnings
import argparse
from datetime import datetime
from tqdm import tqdm

os.environ["DISABLE_FLASH_ATTN"] = "1"
warnings.filterwarnings("ignore")

sys.path.append('/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel

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
    
    # Handle frozen vision tower case
    if len(vision_keys) == 0:
        print(f"\n   ⚠️  No vision tower weights in checkpoint")
        print(f"      This indicates vision tower was FULLY FROZEN during training")
        print(f"      ✅ Using pretrained vision tower from base model")
        vision_tower_status = "FROZEN (using pretrained)"
    elif len(vision_keys) < 50:  # Partial unfreezing (e.g., only top layers)
        print(f"\n   ⚠️  Only {len(vision_keys)} vision tower keys found")
        print(f"      This indicates partial unfreezing (e.g., unfreeze_topk_vision)")
        print(f"      ✅ Loading partially trained vision tower weights")
        vision_tower_status = "PARTIALLY TRAINED"
    else:
        print(f"\n   ✅ Full vision tower found in checkpoint!")
        vision_tower_status = "FULLY TRAINED"
    
    # Load into model
    if len(cleaned_state) > 0:
        print(f"\n   🔄 Loading weights into model...")
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state, strict=False)
        
        loaded_count = len(cleaned_state) - len(missing_keys)
        print(f"      ✅ Loaded: {loaded_count}/{len(cleaned_state)} parameters")
        print(f"      Missing: {len(missing_keys)} (normal for LoRA)")
        print(f"      Unexpected: {len(unexpected_keys)}")
    else:
        print(f"\n   ⚠️  No trainable weights to load (vision was frozen)")
        vision_tower_status = "FROZEN (using pretrained)"
    
    # Verify vision tower status
    model_vision_params = sum(
        p.numel() for n, p in model.named_parameters() 
        if 'visual' in n.lower()
    )
    print(f"\n   📊 Vision parameters in model: {model_vision_params:,}")
    
    if model_vision_params < 400_000_000:
        print(f"      ⚠️  Note: Expected ~600M for full vision tower")
    else:
        print(f"      ✅ Full vision tower present ({model_vision_params:,} params)")
    
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
    print(f"   🎯 Vision tower: {vision_tower_status}")
    if len(merger_keys) > 0:
        print(f"   🎯 Merger: TRAINED (from checkpoint) ✅")
    else:
        print(f"   🎯 Merger: FROZEN (using pretrained) ⚠️")
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
            video_path = os.path.join(args.video_folder, video_file)
            ground_truth = source['conversations'][1]['value']
            
            if not os.path.exists(video_path):
                print(f"\n⚠️  [{idx}/{len(data_dict)}] Video not found: {video_file}")
                results.append({
                    "video": video_file,
                    "model_output": "ERROR: Video not found",
                    "ground_truth": ground_truth
                })
                continue
            
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
                    output_ids = model.generate(
                            **inputs,
                            num_beams=5,                    # Beam search for better quality
                            do_sample=True,                 # Enable sampling for better diversity
                            temperature=0.7,                # Temperature for generation (0.7 is a good balance)
                            top_p=0.9,                      # Nucleus sampling
                            top_k=50,                      # Top-k sampling
                            length_penalty=1.0,            # Length penalty (1.0 = neutral)
                            no_repeat_ngram_size=4,        # Prevent 4-gram repetition
                            repetition_penalty=1.1,        # Slight penalty for token repetition
                            min_length=1,                   # Minimum output length
                            max_new_tokens=args.max_new_tokens  # Maximum tokens to generate
                        )
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
            # Extract ground truth based on format
            conversations = source.get('conversations', [])
            if len(conversations) >= 2:
                gt = conversations[1].get('value', 'unknown')
            else:
                gt = source.get('answer', source.get('ground_truth', 'unknown'))
            results.append({
                "video": source.get("video", "unknown"),
                "model_output": f"ERROR: {str(e)}",
                "ground_truth": source.get('conversations', [{}])[1].get('value', 'unknown') if len(source.get('conversations', [])) > 1 else 'unknown'
            })
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"checkpoint4000_results_{timestamp}.json"
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
            print_evaluation_results(eval_results, "Checkpoint-4000")
            
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
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint on test set")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--model-base", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
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
    parser.add_argument("--video-fps", type=int, default=18,
                       help="FPS for video processing (MUST match training!)")
    parser.add_argument("--min-pixels", type=int, default=320*320,
                       help="Min pixels for video processing (MUST match training!)")
    parser.add_argument("--max-pixels", type=int, default=320*320,
                       help="Max pixels for video processing (MUST match training!)")
    
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