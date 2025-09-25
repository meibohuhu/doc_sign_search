# mhu update 09/24/2025 add evaluation metrics integration for Qwen2.5-VL-7B
## Adapted from simpleQA_metrics.py for Qwen2.5-VL-7B inference
## https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl?usage=Pipeline
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

# Qwen2.5-VL specific imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import os, json  # make sure these imports exist

# Import evaluation metrics
import sys
sys.path.append('/local1/mhu/LLaVANeXT_RC/evaluation')
from common_evaluation import comprehensive_evaluation, print_evaluation_results, save_evaluation_results


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


# # Function to extract frames from video using decord (official pattern)
# def load_video(video_path, max_frames_num):
#     if type(video_path) == str:
#         vr = VideoReader(video_path, ctx=cpu(0))
#     else:
#         vr = VideoReader(video_path[0], ctx=cpu(0))
#     total_frame_num = len(vr)
#     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
#     frame_idx = uniform_sampled_frames.tolist()
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     return spare_frames  # (frames, height, width, channels)


def eval_model(args):
    device = "cuda"
    model_path = os.path.expanduser(args.model_path)
    
    print(f"Loading Qwen2.5-VL model from: {model_path}")
    
    # Load Qwen2.5-VL model and tokenizer
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,          # ← Optimized: faster than bfloat16
            torch_dtype=torch.float16,    # ← Ensure consistent dtype
            device_map="cuda:0",          # ← Explicit device mapping
            trust_remote_code=True,
            low_cpu_mem_usage=True        # ← Memory efficient loading
        )
        model.eval()  # ← Put model in evaluation mode for faster inference
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        # Alternative loading method
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    image_size = resolve_image_size(model, args)
    print(f"[qwen25_metrics] Resolved vision image size: {image_size}")

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
        
        # Prepare the prompt for ASL translation
        fq = "Translate the ASL signs in this video to English. Provide only the English translation in one sentence."   ## If you cannot understand the signs, respond with 'I don't know'.
        # fq = "How many people are in this video? Count the number of people and answer with a number word only."

        try:
            if 'video' in source:
                video_file = source["video"]
                video = os.path.join(args.video_folder, video_file)
                # print(f"Video path: {video}")  # ← Optimized: commented out for speed
            
                # Use official Qwen2.5-VL video processing pattern - correct format
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": video},
                            {"type": "text", "text": fq},
                        ],
                    }
                ]

            elif 'image' in source:
                image_file = source["image"]
                if type(image_file) is list:
                    # Handle multiple images
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": fq},
                            ],
                        }
                    ]
                    for img_f in image_file:
                        img_path = os.path.join(args.image_folder, img_f)
                        messages[0]["content"].insert(-1, {
                            "type": "image",
                            "image": f"file://{img_path}"
                        })
                else:
                    # Handle single image
                    img_path = os.path.join(args.image_folder, image_file)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f"file://{img_path}"
                                },
                                {"type": "text", "text": fq},
                            ],
                        }
                    ]

            # Use the correct Qwen2.5-VL pattern
            if 'video' in source:
                # Process video using the official pattern
                ## Uses video_fps=1 to extract 1 frame per second, Processes these frames as a sequence of images
                inputs = processor.apply_chat_template(
                    conversation,
                    video_fps=1,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(device)
                # print("Video processing successful!")  # ← Optimized: commented out for speed
            else:
                # Handle image case (fallback)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                if image_inputs is not None:
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        **video_kwargs,
                    )
                else:
                    # print("Skipping this sample due to invalid image inputs")  # ← Optimized: commented out for speed
                    continue

            # Generate response using official pattern with optimizations
            with torch.no_grad():
                # Use mixed precision for faster inference
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    try:
                        output_ids = model.generate(**inputs, max_new_tokens=512)
                        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
                        output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                        # print("Generation successful!")  # ← Optimized: commented out for speed
                        
                        # Clear GPU cache for memory efficiency
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Generation error: {e}")
                        output = "Generation failed"
            first_answer = source['conversations'][1]['value']
            
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
            print(f"Generation error: {e}")
            results.append({
                "video": source.get("video"),
                "image": source.get("image"),
                "prompt": source.get("conversations", [{}])[0].get("value"),
                "model_output": "ERROR"
            })
            continue

    # Save results
    with open(os.path.join(args.out_dir, args.answers_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("All outputs saved successfully!")
    
    # Calculate comprehensive evaluation metrics (same as gemini25_metrics.py)
    if args.enable_evaluation and references and predictions:
        try:
            # Use comprehensive evaluation
            results = comprehensive_evaluation(references, predictions)
            
            # Print results
            print_evaluation_results(results, "Qwen2.5-VL-7B")
            
            # Save evaluation metrics
            eval_file = os.path.join(args.out_dir, "evaluation_metrics.json")
            save_evaluation_results(results, eval_file, "Qwen2.5-VL-7B")
            
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")
            print("Continuing without evaluation metrics...")
    else:
        print("No valid references and predictions found for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=None,
                    help="Force vision image size (overrides configs).")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
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
