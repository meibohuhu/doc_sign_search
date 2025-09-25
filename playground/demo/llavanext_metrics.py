# mhu update 09/24/2025 add evaluation metrics integration for LLaVA-NeXT Video
## Adapted from simpleQA_metrics.py for LLaVA-NeXT Video model inference
import os
os.environ["DISABLE_FLASH_ATTN"] = "1"

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print
import torch
import cv2
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

import os, json  # make sure these imports exist

# Import evaluation metrics
import sys
sys.path.append('/local1/mhu/LLaVANeXT_RC/evaluation')
from common_evaluation import comprehensive_evaluation, print_evaluation_results, save_evaluation_results


def resolve_image_size(model, args):
    """
    Pick image size from (1) --image_size,
    (2) model config fallbacks,
    else default=336 for LLaVA-NeXT Video.
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

    # 3) default for LLaVA-NeXT Video
    return 336


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_video(video_path, max_frames_num, fps=1, force_sample=True):
    """
    Load video frames using LLaVA-NeXT official pattern
    Based on: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_Video_1003.md
    """
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    
    # Calculate frame sampling
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    
    # Uniform sampling if needed
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    
    # Convert frame_time to string format
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    
    # Extract frames
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    return spare_frames, frame_time_str, video_time


def eval_model(args):
    device = "cuda"
    device_map = "auto"
    model_path = os.path.expanduser(args.model_path)
    
    print(f"Loading LLaVA-NeXT Video model from: {model_path}")
    
    # Use official LLaVA-NeXT model loading pattern
    llava_model_args = {
        "multimodal": True,
    }
    
    # Auto-detect model name if not provided
    if not hasattr(args, 'model_name') or not args.model_name:
        args.model_name = get_model_name_from_path(model_path)
    
    try:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path, args.model_base, args.model_name, 
            device_map=device_map, attn_implementation="sdpa", **llava_model_args
        )
    except ValueError as e:
        if "image_newline" in str(e) or "shape" in str(e):
            print("⚠️  Model configuration mismatch. Trying without model_name specification...")
            # Try with auto-detected model name
            model_name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                model_path, args.model_base, model_name, 
                device_map=device_map, attn_implementation="eager"
            )
        else:
            raise e

    image_size = resolve_image_size(model, args)
    print(f"[llavanext_metrics] Resolved vision image size: {image_size}")

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
    
    # Process each sample
    for source in tqdm(data_dict):
        video_file=None
        image_file=None
        
        # Prepare the prompt for person counting
        # fq = "How many people are in this video? Count the number of people and answer with a number word only."
        #  fq = "Translate the ASL signs in this video to English. Provide only the translation in one sentence. If you cannot understand the signs, respond with 'I don't know'."

        fq = "Translate the ASL signs in this video to English. Provide only the translation in one sentence. "
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
                        print(f"Skipping large video ({video_size_mb:.2f} MB) to avoid memory issues")
                        results.append({
                            "video": video_file,
                            "image": None,
                            "prompt": fq,
                            "model_output": "Skipped - Video too large",
                            "ground_truth": source['conversations'][1]['value']
                        })
                        continue
                    
                    # Adjust frames based on video size
                    if video_size_mb < 5.0:  # Small videos
                        max_frames_num = 16
                    elif video_size_mb < 20.0:  # Medium videos
                        max_frames_num = 10
                    else:  # Large videos
                        max_frames_num = 8
                        
                    print(f"Using max_frames_num: {max_frames_num}")
                except Exception as e:
                    print(f"Could not check video size: {e}")
                    max_frames_num = 16
            
                # Load video frames using LLaVA-NeXT pattern
                try:
                    video_frames, frame_time, video_time = load_video(video_path, max_frames_num, fps=1, force_sample=True)
                    print(f"Extracted {len(video_frames)} frames successfully!")
                    print(f"Video time: {video_time:.2f}s, Frame times: {frame_time}")
                    
                except Exception as e:
                    print(f"Video loading failed: {e}")
                    results.append({
                        "video": video_file,
                        "image": None,
                        "prompt": fq,
                        "model_output": f"ERROR: Video loading failed - {str(e)}",
                        "ground_truth": source.get("conversations", [{}])[1].get("value", "No ground truth")
                    })
                    continue

            elif 'image' in source:
                image_file = source["image"]
                if type(image_file) is list:
                    # Handle multiple images
                    video_frames = []
                    for img_f in image_file:
                        img_path = os.path.join(args.image_folder, img_f)
                        img = Image.open(img_path).convert('RGB')
                        video_frames.append(np.array(img))
                    video_frames = np.stack(video_frames)
                    frame_time = "0.00s"
                    video_time = len(video_frames)
                else:
                    # Handle single image
                    img_path = os.path.join(args.image_folder, image_file)
                    img = Image.open(img_path).convert('RGB')
                    video_frames = np.array(img)
                    video_frames = np.expand_dims(video_frames, axis=0)  # Add frame dimension
                    frame_time = "0.00s"
                    video_time = 1.0

            # Process video frames
            video_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
            video_list = [video_tensor]  # LLaVA expects a list of tensors

            # Build prompt with video tokens
            if args.add_time_instruction:
                time_instruction = (
                    f"The video lasts for {video_time:.2f} seconds, and {len(video_frames)} frames are uniformly sampled. "
                    f"These frames are located at {frame_time}. Please answer the following question."
                )
                qs = f"{time_instruction}\n{fq}"
            else:
                qs = fq

            # Add image tokens
            if getattr(model.config, "mm_use_im_start_end", False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            # Conversation template fallback
            conv_mode = args.conv_mode
            if conv_mode is None:
                name = (tokenizer.name_or_path or "").lower()
                if "qwen" in name:
                    conv_mode = "qwen_1_5" if "1.5" in name or "qwen2" not in name else "qwen_2"
                elif "mistral" in name:
                    conv_mode = "mistral_instruct"
                else:
                    conv_mode = "vicuna_v1"

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Generate response
            with torch.no_grad():
                try:
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                    
                    cont = model.generate(
                        input_ids,
                        images=video_list,
                        modalities=["video"],
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True
                    )
                    
                    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                    
                    # Extract the response part (after the prompt)
                    if conv.roles[1] in text_outputs:
                        output = text_outputs.split(conv.roles[1])[-1].strip()
                    else:
                        output = text_outputs
                    
                    print("Generation successful!")
                    
                except Exception as e:
                    print(f"Generation error: {e}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fallback response
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
            print_evaluation_results(results, "LLaVA-NeXT Video")
            
            # Save evaluation metrics
            eval_file = os.path.join(args.out_dir, "evaluation_metrics.json")
            save_evaluation_results(results, eval_file, "LLaVA-NeXT Video")
            
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            print("Continuing without evaluation metrics...")
    else:
        print("No valid references and predictions found for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--add_time_instruction", action="store_true", default=True,
                       help="Add time instruction to prompt (default: True)")
    parser.add_argument("--enable_evaluation", action="store_true", default=True,
                       help="Enable evaluation metrics calculation (default: True)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (default: process all)")
    args = parser.parse_args()

    eval_model(args)
