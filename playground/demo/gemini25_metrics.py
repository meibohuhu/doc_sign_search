# mhu update 09/24/2025 add evaluation metrics integration for Gemini 2.5 Pro
## Adapted from gpt4o_metrics.py for Gemini 2.5 Pro API inference
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
import io
random.seed(10)
warnings.filterwarnings("ignore")

# Google Gemini API imports
import google.generativeai as genai

import os, json  # make sure these imports exist

# Import evaluation metrics
import sys
sys.path.append('/local1/mhu/LLaVANeXT_RC/evaluation')
from common_evaluation import comprehensive_evaluation, print_evaluation_results, save_evaluation_results


def resolve_image_size(model, args):
    """
    Pick image size from (1) --image_size,
    (2) model config fallbacks,
    else default=1024 for Gemini.
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

    # 3) default for Gemini
    return 1024


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def extract_video_frames(video_path, num_frames=8):
    """Extract frames from video for Gemini"""
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


def frames_to_pil_images(frames, max_size=(512, 512)):
    """Convert frames to PIL Images for Gemini with aggressive size optimization"""
    pil_images = []
    for frame in frames:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Resize if too large to reduce token usage (more aggressive)
        if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            print(f"Resized frame from {frame.shape[:2][::-1]} to {pil_image.size}")
        
        pil_images.append(pil_image)
    
    return pil_images


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_gemini_api(model, content, images=None, max_tokens=4096, temperature=0.7):
    """Call Gemini 2.5 Pro API with content and images"""
    try:
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        if images:
            # For video frames, send all images with the prompt
            if isinstance(images, list):
                # Combine all images with the text prompt
                response = model.generate_content(
                    [content] + images,
                    generation_config=generation_config
                )
            else:
                # Single image
                response = model.generate_content(
                    [content, images],
                    generation_config=generation_config
                )
        else:
            # Text only
            response = model.generate_content(
                content,
                generation_config=generation_config
            )
        print("response:", response)
        # Check response status and handle different cases
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            
            if finish_reason == 1:  # STOP - normal completion
                if hasattr(candidate, 'content') and candidate.content.parts:
                    return candidate.content.parts[0].text.strip()
                else:
                    print(f"Warning: No content in response")
                    return "No content generated"
            elif finish_reason == 2:  # MAX_TOKENS
                print(f"Warning: Response truncated due to max tokens")
                if hasattr(candidate, 'content') and candidate.content.parts:
                    text = candidate.content.parts[0].text.strip()
                    if text and len(text) > 0:
                        # Try to extract a number from truncated response
                        import re
                        numbers = re.findall(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|\d+)\b', text.lower())
                        if numbers:
                            return numbers[0]
                        return text
                    else:
                        return "I don't know"
                else:
                    return "I don't know"
            elif finish_reason == 3:  # SAFETY - blocked by safety filters
                print(f"Warning: Response blocked by safety filters")
                return "Response blocked by safety filters"
            elif finish_reason == 4:  # RECITATION - blocked by recitation filters
                print(f"Warning: Response blocked by recitation filters")
                return "Response blocked by recitation filters"
            else:
                print(f"Warning: Unexpected finish reason: {finish_reason}")
                return f"Unexpected finish reason: {finish_reason}"
        else:
            print(f"Warning: No candidates in response")
            return "No candidates generated"
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"API Error: {str(e)}"


def eval_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    api_key = args.api_key
    
    print(f"Initializing Gemini 2.5 Pro model...")
    
    # Configure Gemini API
    try:
        genai.configure(api_key=api_key)
        # Use Gemini 2.5 Pro model
        model = genai.GenerativeModel('gemini-2.5-pro')
        print("Gemini 2.5 Pro model initialized successfully!")
    except Exception as e:
        print(f"Error initializing Gemini 2.5 Pro model: {e}")
    

    image_size = resolve_image_size(None, args)
    print(f"[gemini25_metrics] Resolved vision image size: {image_size}")

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
        
        # Prepare the prompt for person counting
        # fq = "How many people are in this video? Count the number of people and answer with a number word only."
        fq = "Translate the ASL signs in this video to English text. Provide only the English translation without describing the person, gestures, or video content. Answer in one sentence only. If you don't understand, DON'T respond."

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
                    
                    # # Adjust frames based on video size (reduced for Gemini API limits)
                    # if video_size_mb < 5.0:  # Small videos
                    #     num_frames = 2  # Further reduced for Gemini
                    # elif video_size_mb < 20.0:  # Medium videos
                    #     num_frames = 1  # Only 1 frame for medium videos
                    # else:  # Large videos (>20MB)
                    #     num_frames = 1  # Only 1 frame for large videos
                        
                    # print(f"Using num_frames: {2}")
                except Exception as e:
                    print(f"Could not check video size: {e}")
                    num_frames = 1  # Further reduced default
            
                # Extract frames from video
                try:
                    frames = extract_video_frames(video_path, 8)
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
                    
                    # Convert frames to PIL images for Gemini
                    pil_images = frames_to_pil_images(frames)
                    print(f"Extracted {len(frames)} frames successfully!")
                    print(f"Frame dimensions: {pil_images[0].size if pil_images else 'No images'}")
                    
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
                    pil_images = []
                    for img_f in image_file:
                        img_path = os.path.join(args.image_folder, img_f)
                        pil_image = Image.open(img_path).convert('RGB')
                        pil_images.append(pil_image)
                else:
                    # Handle single image
                    img_path = os.path.join(args.image_folder, image_file)
                    pil_image = Image.open(img_path).convert('RGB')
                    pil_images = [pil_image]

            # Call Gemini API
            try:
                output = call_gemini_api(
                    model, 
                    fq,
                    images=pil_images,
                    max_tokens=2048,  # Much lower for simple counting
                    temperature=args.temperature
                )
                print("API call successful!")
                
            except Exception as e:
                print(f"API call failed: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                
                # Fallback response
                output = "API call failed - unable to generate response"
            
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
            print_evaluation_results(results, "Gemini 2.5 Pro")
            
            # Save evaluation metrics
            eval_file = os.path.join(args.out_dir, "evaluation_metrics.json")
            save_evaluation_results(results, eval_file, "Gemini 2.5 Pro")
            
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            print("Continuing without evaluation metrics...")
    else:
        print("No valid references and predictions found for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=None,
                    help="Force vision image size (overrides configs).")
    parser.add_argument("--api-key", type=str, required=True,
                    help="Google Gemini API key")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--enable_evaluation", action="store_true", default=True,
                       help="Enable evaluation metrics calculation (default: True)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: process all)")
    args = parser.parse_args()

    eval_model(args)
