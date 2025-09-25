# mhu update 09/22/2025 add evaluation metrics integration for GPT-4o
## Adapted from internvl25_metrics.py for GPT-4o API inference
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
import base64
import cv2
random.seed(10)
warnings.filterwarnings("ignore")

# OpenAI API imports
import openai
from openai import OpenAI

import os, json  # make sure these imports exist

# Import evaluation metrics
import sys
sys.path.append('/local1/mhu/LLaVANeXT_RC/evaluation/uni_sign')
from SLRT_metrics import translation_performance


def resolve_image_size(model, args):
    """
    Pick image size from (1) --image_size,
    (2) model config fallbacks,
    else default=1024 for GPT-4o.
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

    # 3) default for GPT-4o
    return 1024


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def encode_image(image_path):
    """Encode image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_video_frames(video_path, num_frames=8):
    """Extract frames from video for GPT-4o"""
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices
        frame_indices = []
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // num_frames
            frame_indices = [i * step for i in range(num_frames)]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []


def frames_to_base64(frames):
    """Convert frames to base64 encoded images"""
    encoded_frames = []
    for frame in frames:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Save to temporary bytes
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        # Encode to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        encoded_frames.append(encoded_image)
    
    return encoded_frames


def call_gpt4o_api(client, messages, max_tokens=512, temperature=0.7):
    """Call GPT-4o API with messages"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return f"Error: {str(e)}"


def eval_model(args):
    # Initialize OpenAI client
    client = OpenAI(api_key=args.openai_api_key)
    
    print(f"Using GPT-4o API for inference")
    
    image_size = resolve_image_size(None, args)
    print(f"[gpt4o_metrics] Using image size: {image_size}")

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
    
    # Process each sample
    for source in tqdm(data_dict):
        video_file=None
        image_file=None
        
        # Prepare the prompt for ASL translation
        fq = "Translate the ASL signs in this video to English. Provide only the translation in one sentence. If you cannot understand the signs, respond with 'I don't know'."

        # fq = "How many people are in this video? Count the number of people and answer with a number word only."

        try:
            if 'video' in source:
                video_file = source["video"]
                video_path = os.path.join(args.video_folder, video_file)
                print(f"Video path: {video_path}")
                
                # Check video file size to avoid issues
                try:
                    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    print(f"Video size: {video_size_mb:.2f} MB")
                    if video_size_mb > 100:  # Skip very large videos
                        print(f"Skipping large video ({video_size_mb:.2f} MB) to avoid API limits")
                        results.append({
                            "video": video_file,
                            "image": None,
                            "prompt": fq,
                            "model_output": "Skipped - Video too large",
                            "ground_truth": source['conversations'][1]['value']
                        })
                        continue
                    
                    # # Adjust frames based on video size
                    # if video_size_mb < 5.0:  # Small videos
                    #     num_frames = 8
                    # elif video_size_mb < 20.0:  # Medium videos
                    #     num_frames = 6
                    # else:  # Large videos
                    #     num_frames = 4
                        
                    print(f"Using num_frames: {num_frames}")
                except Exception as e:
                    print(f"Could not check video size: {e}")
                    num_frames = 6
            
                # Extract frames from video
                try:
                    frames = extract_video_frames(video_path, 10)
                    if not frames:
                        print("No frames extracted from video")
                        results.append({
                            "video": video_file,
                            "image": None,
                            "prompt": fq,
                            "model_output": "ERROR: No frames extracted",
                            "ground_truth": source.get("conversations", [{}])[1].get("value", "No ground truth")
                        })
                        continue
                    
                    # Convert frames to base64
                    encoded_frames = frames_to_base64(frames)
                    print(f"Extracted {len(frames)} frames successfully!")
                    
                except Exception as e:
                    print(f"Frame extraction failed: {e}")
                    results.append({
                        "video": video_file,
                        "image": None,
                        "prompt": fq,
                        "model_output": f"ERROR: Frame extraction failed - {str(e)}",
                        "ground_truth": source.get("conversations", [{}])[1].get("value", "No ground truth")
                    })
                    continue

            elif 'image' in source:
                image_file = source["image"]
                if type(image_file) is list:
                    # Handle multiple images
                    encoded_frames = []
                    for img_f in image_file:
                        img_path = os.path.join(args.image_folder, img_f)
                        encoded_image = encode_image(img_path)
                        encoded_frames.append(encoded_image)
                else:
                    # Handle single image
                    img_path = os.path.join(args.image_folder, image_file)
                    encoded_image = encode_image(img_path)
                    encoded_frames = [encoded_image]

            # Prepare messages for GPT-4o API
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in American Sign Language (ASL) translation. Analyze the provided video frames and translate the ASL signs to English. Provide only the translation in one sentence."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": fq
                        }
                    ]
                }
            ]
            
            # Add video frames to the message
            for i, encoded_frame in enumerate(encoded_frames):
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_frame}"
                    }
                })
            
            # Call GPT-4o API
            try:
                output = call_gpt4o_api(
                    client, 
                    messages, 
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
                print("API call successful!")
                
            except Exception as api_error:
                print(f"API call failed: {api_error}")
                output = f"Error: API call failed - {str(api_error)}"
                    
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
            bleu_dict, rouge_score = translation_performance(references, predictions)
            
            print(f"\nBLEU Scores:")
            for k, v in bleu_dict.items():
                print(f"  {k}: {v:.4f}")
            
            print(f"\nROUGE-L F1 Score: {rouge_score:.4f}")
            
            # Save evaluation metrics to a separate file
            eval_results = {
                "bleu_scores": bleu_dict,
                "rouge_l_f1": rouge_score,
                "total_samples": len(references),
                "evaluation_summary": {
                    "bleu_1": bleu_dict.get("bleu1", 0),
                    "bleu_2": bleu_dict.get("bleu2", 0),
                    "bleu_3": bleu_dict.get("bleu3", 0),
                    "bleu_4": bleu_dict.get("bleu4", 0),
                    "rouge_l_f1": rouge_score
                }
            }
            
            eval_file = os.path.join(args.out_dir, "evaluation_metrics.json")
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            
            print(f"\nEvaluation metrics saved to: {eval_file}")
            
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            print("Continuing without evaluation metrics...")
    else:
        print("No valid references and predictions found for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=None,
                    help="Force vision image size (overrides configs).")
    parser.add_argument("--openai_api_key", type=str, required=True,
                    help="OpenAI API key for GPT-4o")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--enable_evaluation", action="store_true", default=True,
                       help="Enable evaluation metrics calculation (default: True)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: process all)")
    args = parser.parse_args()

    eval_model(args)



