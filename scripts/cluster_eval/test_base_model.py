#!/usr/bin/env python3
"""
Test base Qwen2VL model without LoRA with 10 videos and evaluation metrics
"""

import os
import sys
import json
import torch
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Disable flash attention
os.environ["DISABLE_FLASH_ATTN"] = "1"
warnings.filterwarnings("ignore")

# Add the src directory to Python path
sys.path.append('/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def test_base_model():
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"🔍 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Configuration
    model_base = "Qwen/Qwen2.5-VL-3B-Instruct"
    test_data_path = "/home/mh2803/projects/sign_language_llm/vanshika/asl_test/segmented_videos.json"
    video_folder = "/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/"
    out_dir = "/home/mh2803/projects/sign_language_llm/outputs/"
    
    print("🎬 Testing Base Qwen2VL Model (No LoRA)")
    print("=" * 50)
    print(f"Model: {model_base}")
    print(f"Test Data: {test_data_path}")
    print(f"Video Folder: {video_folder}")
    print(f"Output Dir: {out_dir}")
    print()
    
    # Load base model (no LoRA)
    print("🚀 Loading base model...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_base,
            dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_base, trust_remote_code=True)
        
        model.eval()
        print("✅ Base model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test data
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found: {test_data_path}")
        return
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Limit to 10 samples for testing
    test_data = test_data[:10]
    print(f"📋 Processing {len(test_data)} samples")
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    results = []
    references = []
    predictions = []
    
    print(f"🎬 Starting evaluation on {len(test_data)} videos...")
    print("-" * 60)
    
    # Process each sample
    for i, source in enumerate(tqdm(test_data, desc="Processing videos")):
        try:
            fq = "Translate the ASL signs in this video to English. Provide only the English translation in one sentence."
            
            if 'video' in source:
                video_file = source["video"]
                video = os.path.join(video_folder, video_file)
                
                # Debug: print video path
                print(f"🔍 Video path: {video}")
                print(f"🔍 Video exists: {os.path.exists(video)}")
                
                # Prepare conversation
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": video},
                            {"type": "text", "text": fq},
                        ],
                    }
                ]
                
                # Process video using the official pattern
                inputs = processor.apply_chat_template(
                    conversation,
                    video_fps=10,  # Start with 1 fps for base model
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)
                
                # Generate response
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output_ids = model.generate(**inputs, max_new_tokens=128)
                        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                        
                        # Clear GPU cache for memory efficiency
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Get ground truth from conversations
                first_answer = source['conversations'][1]['value']
                
                print(f"Video: {video_file}")
                print(f"Ground truth: {first_answer}")
                print(f"Prediction: {response}")
                print("-" * 60)
                
                # Store references and predictions for evaluation
                references.append(first_answer)
                predictions.append(response)
                
                results.append({
                    "video": video_file,
                    "prompt": fq,
                    "model_output": response,
                    "ground_truth": first_answer
                })
                
            else:
                print(f"⚠️  No video found in sample: {source}")
                continue
                
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            results.append({
                "video": source.get("video", "unknown"),
                "prompt": fq,
                "model_output": f"ERROR: {str(e)}",
                "ground_truth": source.get('conversations', [{}])[1].get('value', 'unknown') if len(source.get('conversations', [])) > 1 else 'unknown'
            })
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"base_qwen2vl_results_{timestamp}.json"
    output_path = os.path.join(out_dir, output_file)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ All outputs saved successfully!")
    print(f"📄 Results saved to: {output_path}")
    
    # Calculate evaluation metrics
    if references and predictions:
        try:
            print("\n📊 Calculating evaluation metrics...")
            
            # Import evaluation metrics
            sys.path.append('/home/mh2803/projects/sign_language_llm/evaluation')
            from ssvp_evaluation import comprehensive_evaluation, print_evaluation_results, save_evaluation_results
            
            # Use comprehensive evaluation
            eval_results = comprehensive_evaluation(references, predictions)
            
            # Print results
            print_evaluation_results(eval_results, "Base-Qwen2VL-3B")
            
            # Save evaluation metrics
            eval_file = os.path.join(out_dir, f"base_model_evaluation_metrics_{timestamp}.json")
            save_evaluation_results(eval_results, eval_file, "Base-Qwen2VL-3B")
            
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

if __name__ == "__main__":
    test_base_model()
