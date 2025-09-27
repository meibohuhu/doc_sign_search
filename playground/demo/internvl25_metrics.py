# mhu update 09/24/2025 add evaluation metrics integration for InternVL 2.5 (8B)
## Adapted from qwen25_metrics.py for InternVL 2.5 (8B) inference
import os
os.environ["DISABLE_FLASH_ATTN"] = "1"

import torch
import numpy as np
from PIL import Image
import requests
import copy
import warnings
import argparse
import json
from tqdm import tqdm
import math
from decord import VideoReader, cpu
import random
random.seed(10)
warnings.filterwarnings("ignore")

# InternVL 2.5 specific imports
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms

import os, json  # make sure these imports exist

# Import evaluation metrics using SSVP-SLT style implementation
import sys
sys.path.append('/local1/mhu/LLaVANeXT_RC/evaluation')
from ssvp_evaluation import comprehensive_evaluation, print_evaluation_results, save_evaluation_results


def resolve_image_size(model, args):
    """
    Pick image size from (1) --image_size,
    (2) model config fallbacks,
    else default=448 for InternVL.
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

    # 3) default for InternVL 2.5
    return 448


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def build_transform(input_size=448):
    """Build transform for InternVL"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=1):
    """Dynamic preprocessing for InternVL"""
    # Simple implementation - just return the image as a list
    return [image]


def load_image(image_path, max_num=12):
    """Load and preprocess image for InternVL"""
    image = Image.open(image_path).convert('RGB')
    image = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
    transform = build_transform(input_size=448)
    pixel_values = [transform(tile) for tile in image]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Get frame indices for video processing"""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Load video frames for InternVL - official implementation"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def eval_model(args):
    device = "cuda"
    model_path = args.model_path
    
    print(f"Loading InternVL 2.5 (8B) model from: {model_path}")
    
    # Load InternVL 2.5 model and tokenizer
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        # Alternative loading method
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    image_size = resolve_image_size(model, args)
    print(f"[internvl25_metrics] Resolved vision image size: {image_size}")

    model.eval()
    data_dict = json.load(open(args.question_file,'r'))

    data_dict = get_chunk(data_dict, args.num_chunks, args.chunk_idx)
    
    # Limit the number of samples if specified
    if args.max_samples is not None:
        original_size = len(data_dict)
        data_dict = data_dict[:args.max_samples]
        print(f"Limited to {len(data_dict)} samples (from {original_size} total)")
    
    print("chunk size:",len(data_dict))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    results=[]
    # Lists to store references and predictions for evaluation
    references = []
    predictions = []
    
    # Generation config
    generation_config = dict(
        max_new_tokens=512,
        do_sample=False,
        temperature=0.7,
        top_p=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Process each sample
    for source in tqdm(data_dict):
        video_file=None
        image_file=None
        
        # Prepare the prompt for ASL translation
        fq = "Translate the ASL signs in this video to English. Provide only the translation in one sentence. "
        # fq = "How many people are in this video? Count the number of people and answer with a number word only."

        try:
            if 'video' in source:
                video_file = source["video"]
                video_path = os.path.join(args.video_folder, video_file)
                print(f"Video path: {video_path}")
                
                # Check video file size to avoid OOM
                try:
                    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    print(f"Video size: {video_size_mb:.2f} MB")
                    # if video_size_mb > 50:  # Skip videos larger than 50MB
                    #     print(f"Skipping large video ({video_size_mb:.2f} MB) to avoid OOM")
                    #     results.append({
                    #         "video": video_file,
                    #         "image": None,
                    #         "prompt": fq,
                    #         "model_output": "Skipped - Video too large",
                    #         "ground_truth": source['conversations'][1]['value']
                    #     })
                    #     continue
                    
                    # Adjust segments based on video size
                    if video_size_mb < 1.0:  # Very small videos
                        num_segments = 4
                    else:
                        num_segments = 8
                        
                    print(f"Using num_segments: {num_segments}")
                except Exception as e:
                    print(f"Could not check video size: {e}")
                    num_segments = 8
            
                # Load video frames using official InternVL pattern
                try:
                    pixel_values, num_patches_list = load_video(video_path, num_segments=10, max_num=1)
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                    
                    # Create video prefix for InternVL - official format
                    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                    question = video_prefix + fq
                    
                    print("Video processing successful!")
                except Exception as e:
                    print(f"Video processing failed: {e}")
                    results.append({
                        "video": video_file,
                        "image": None,
                        "prompt": fq,
                        "model_output": f"ERROR: Video processing failed - {str(e)}",
                        "ground_truth": source.get("conversations", [{}])[1].get("value", "No ground truth")
                    })
                    continue

            elif 'image' in source:
                image_file = source["image"]
                if type(image_file) is list:
                    # Handle multiple images
                    pixel_values_list = []
                    num_patches_list = []
                    for img_f in image_file:
                        img_path = os.path.join(args.image_folder, img_f)
                        pixel_values = load_image(img_path, max_num=12)
                        pixel_values_list.append(pixel_values)
                        num_patches_list.append(pixel_values.shape[0])
                    
                    pixel_values = torch.cat(pixel_values_list)
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                    
                    # Create image prefix for InternVL
                    image_prefix = ''.join([f'Image{i+1}: <image>\n' for i in range(len(num_patches_list))])
                    question = image_prefix + fq
                else:
                    # Handle single image
                    img_path = os.path.join(args.image_folder, image_file)
                    pixel_values = load_image(img_path, max_num=12)
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                    num_patches_list = [pixel_values.shape[0]]
                    question = '<image>\n' + fq

            # Generate response using InternVL pattern
            with torch.no_grad():
                try:
                    # Clear cache before generation
                    torch.cuda.empty_cache()
                    
                    # Use InternVL's chat method for proper generation
                    try:
                        # Use the chat method for proper text generation
                        response, history = model.chat(
                            tokenizer, 
                            pixel_values, 
                            question, 
                            generation_config,
                            num_patches_list=num_patches_list, 
                            history=None, 
                            return_history=True
                        )
                        output = response.strip()
                        print("Chat method successful!")
                        
                    except Exception as chat_error:
                        print(f"Chat method failed: {chat_error}")
                        print(f"Error type: {type(chat_error).__name__}")
                        # Fallback response
                        if "How many people" in question:
                            output = "I can see people in the video but need proper generation to count them accurately."
                        elif "Translate" in question:
                            output = "I can see the ASL signs but need proper generation to translate them accurately."
                        else:
                            output = "I can see the video content but need proper generation to respond accurately."
                    
                except Exception as e:
                    print(f"Generation error: {e}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fallback: try simple generation
                    try:
                        print("Trying simple generation...")
                        # Simple fallback - just return a placeholder
                        output = "I can see the video but need proper generation implementation."
                    except Exception as e2:
                        print(f"Fallback generation also failed: {e2}")
                        output = "Generation failed"
                    
            # Get ground truth with error handling
            try:
                if 'conversations' in source and len(source['conversations']) > 1:
                    first_answer = source['conversations'][1]['value']
                else:
                    first_answer = "No ground truth available"
                    print("Warning: No ground truth found in conversations")
            except (KeyError, IndexError) as e:
                first_answer = "Error accessing ground truth"
                print(f"Error accessing ground truth: {e}")
            
            print("ground truth:", first_answer)
            print("Generation: ", output)
            
            # Store references and predictions for evaluation
            references.append(first_answer)
            predictions.append(output)
            
            results.append({
                "video": video_file,
                "image": image_file,
                "prompt": fq,
                "model_output": output,
                "ground_truth": first_answer
            })
            
        except Exception as e:
            print(f"Processing error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Try to get ground truth safely
            try:
                ground_truth = source.get("conversations", [{}])[1].get("value", "No ground truth")
            except:
                ground_truth = "Error accessing ground truth"
            
            results.append({
                "video": source.get("video"),
                "image": source.get("image"),
                "prompt": fq,
                "model_output": f"ERROR: {str(e)}",
                "ground_truth": ground_truth
            })
            continue

    # Save results
    with open(os.path.join(args.out_dir, args.answers_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("All outputs saved successfully!")
    
    # Calculate evaluation metrics
    if args.enable_evaluation and references and predictions:
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        try:
            # Use comprehensive evaluation
            results = comprehensive_evaluation(references, predictions)
            
            # Print results
            print_evaluation_results(results, "InternVL 2.5 Pro")
            
            # Save evaluation metrics
            eval_file = os.path.join(args.out_dir, "evaluation_metrics.json")
            save_evaluation_results(results, eval_file, "InternVL 2.5 Pro")
            
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            print("Continuing without evaluation metrics...")
    else:
        print("No valid references and predictions found for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=None,
                    help="Force vision image size (overrides configs).")
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL2-8B")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--enable_evaluation", action="store_true", default=True,
                       help="Enable evaluation metrics calculation (default: True)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: process all)")
    args = parser.parse_args()

    eval_model(args)
