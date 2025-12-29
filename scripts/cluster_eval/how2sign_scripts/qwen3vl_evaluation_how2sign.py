#!/usr/bin/env python3
"""
Qwen3-VL Evaluation Script for How2Sign
Based on Qwen3-VL official API
"""

import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
import torch
import warnings
import argparse
from datetime import datetime
from tqdm import tqdm

os.environ["DISABLE_FLASH_ATTN"] = "1"
warnings.filterwarnings("ignore")

# Add Qwen2VL paths - auto-detect project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
qwen2vl_path = os.path.join(project_root, 'qwenvl/Qwen2-VL-Finetune/src')
if os.path.exists(qwen2vl_path):
    sys.path.insert(0, qwen2vl_path)
# Fallback to hardcoded paths if auto-detection fails
if '/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src' not in sys.path:
    sys.path.append('/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src')

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def extract_frames_from_video(video_path, num_frames=None, fps=None, save_frames_dir=None):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (if None, use fps-based sampling)
        fps: Target FPS for frame extraction (if num_frames is None)
        save_frames_dir: Directory to save extracted frames (optional)
    
    Returns:
        List of PIL Image objects, and list of saved frame paths (if save_frames_dir is provided)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame indices to extract
    if num_frames is not None:
        # Extract evenly spaced frames
        if num_frames > total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    elif fps is not None:
        # Extract frames at target FPS
        frame_interval = max(1, int(video_fps / fps))
        frame_indices = list(range(0, total_frames, frame_interval))
    else:
        # Default: extract 6 frames evenly spaced
        num_frames = min(6, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frames = []
    saved_paths = []
    frame_count = 0
    saved_count = 0
    
    # Create output directory if saving frames
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in frame_indices:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            
            # Save frame if directory is provided
            if save_frames_dir:
                frame_filename = f"{video_name}_frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(save_frames_dir, frame_filename)
                pil_image.save(frame_path, quality=95)
                saved_paths.append(frame_path)
                saved_count += 1
        
        frame_count += 1
        if len(frames) >= len(frame_indices):
            break
    
    cap.release()
    
    if save_frames_dir:
        return frames, saved_paths
    return frames

def load_trained_model(checkpoint_path, base_model_name="Qwen/Qwen3-VL-2B-Instruct"):
    """
    Load trained model from checkpoint (if provided) or base model
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("🚀 Loading model from checkpoint...")
        print(f"   Checkpoint: {checkpoint_path}")
        
        # Load from checkpoint
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        print("✅ Model loaded from checkpoint")
    else:
        print("ℹ️  Loading base model...")
        # Load base model
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        print("✅ Base model loaded")
    
    model.eval()
    return model, processor

def eval_model(args):
    device = "cuda"
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
    
    # Load model
    try:
        model, processor = load_trained_model(
            args.checkpoint_path if args.checkpoint_path else None,
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
            fq = "Translate the ASL signs in this video to English text. Provide only the English translation without describing the person, gestures, or video content. Answer in one sentence only. If you cannot determine the meaning, RESPOND with nothing."
            # fq = "Observe this ASL video and describe in detail how the person's hand gestures change. For each change you observe, describe: (1) which hand is moving, (2) the specific finger positions (which fingers are extended, curled, or touching), (3) the hand shape and orientation, (4) the hand location relative to the body, (5) the movement direction and how it transitions. Describe all visible changes frame by frame. CRITICAL RULES: (1) Only describe the physical movements you can directly observe. (2) Use factual, objective language only. (3) FORBIDDEN WORDS AND PHRASES - DO NOT USE: 'indicate', 'suggests', 'seems', 'appears', 'may', 'might', 'could', 'possibly', 'probably', 'likely', 'represents', 'signifies', 'means', 'communicates', 'expresses', 'implies', 'conveys', 'shows that', 'demonstrates that', or any other interpretive or speculative language. (4) Describe ONLY what you see: hand positions, finger states, movements, locations, and transitions. Do NOT describe what the gestures might mean, what they could represent, or what they seem to indicate."
            # fq = "Describe concisely. Format:\n\nRight hand: [position, finger states, palm orientation]\nLeft hand: [position, finger states, palm orientation]\nFace: [expression]\n\nBe brief and factual. No interpretations."
            
            # fq = """
            # You are an ASL motion-description annotator.

            # Describe each distinct action/movement in the video. For EACH action, provide:

            # Action X:
            # - Right hand: handshape, palm orientation, location, movement path, finger positions
            # - Left hand: handshape, palm orientation, location, movement path, finger positions
            # - Hand interaction: contact, relative position, synchrony

            # Rules:
            # - Describe EVERY distinct action/movement you observe.
            # - DO NOT infer meaning; only describe observable motion.
            # - Capture all movement transitions and changes.
            # - Be detailed and precise for each action.
            # - Use factual, objective language only.
            # - FORBIDDEN WORDS: 'indicate', 'suggests', 'seems', 'appears', 'may', 'might', 'could', 'possibly', 'probably', 'likely', 'represents', 'signifies', 'means', 'communicates', 'expresses', 'implies', 'conveys', 'shows that', 'demonstrates that'.
            # """

            # fq = """
            # You are an ASL motion-description annotator.

            # Describe the most important hand movements in the video using exactly TWO statements.

            # Format:
            # Statement 1: [Describe the most significant hand movement, including handshape, palm orientation, location, movement path, and finger positions for both hands if relevant]
            # Statement 2: [Describe the second most important hand movement or interaction, including handshape, palm orientation, location, movement path, and hand interaction if relevant]

            # Rules:
            # - Focus on the TWO most important/distinctive movements in the video.
            # - DO NOT infer meaning; only describe observable motion.
            # - Use factual, objective language only.
            # - Each statement should be a complete sentence describing one movement.
            # - FORBIDDEN WORDS: 'indicate', 'suggests', 'seems', 'appears', 'may', 'might', 'could', 'possibly', 'probably', 'likely', 'represents', 'signifies', 'means', 'communicates', 'expresses', 'implies', 'conveys', 'shows that', 'demonstrates that'.
            # """



            video_file = source["video"]
            video_path = os.path.join(args.video_folder, video_file)
            
            # Extract ground truth from conversations
            conversations = source.get('conversations', [])
            if len(conversations) >= 2:
                ground_truth = conversations[1].get('value', '')
            else:
                ground_truth = source.get('answer', source.get('ground_truth', ''))
            
            if not os.path.exists(video_path):
                print(f"\n⚠️  [{idx}/{len(data_dict)}] Video not found: {video_file}")
                results.append({
                    "video": video_file,
                    "model_output": "ERROR: Video not found",
                    "ground_truth": ground_truth
                })
                continue
            
            # Prepare input using Qwen3-VL format
            if args.use_frames:
                # Extract frames and process each frame individually
                try:
                    # Create frames output directory if saving frames
                    frames_output_dir = None
                    if args.save_frames:
                        frames_output_dir = os.path.join(args.out_dir, "extracted_frames")
                        os.makedirs(frames_output_dir, exist_ok=True)
                    
                    # Extract frames (and save if requested)
                    result = extract_frames_from_video(
                        video_path, 
                        num_frames=args.num_frames,
                        fps=args.video_fps,
                        save_frames_dir=frames_output_dir
                    )
                    
                    if isinstance(result, tuple):
                        frames, saved_paths = result
                    else:
                        frames = result
                        saved_paths = []
                    
                    if not frames:
                        raise ValueError("No frames extracted from video")
                    
                    print(f"\n📹 [{idx}/{len(data_dict)}] Processing {len(frames)} frames individually for {video_file}")
                    if saved_paths:
                        print(f"   💾 Saved {len(saved_paths)} frames to: {frames_output_dir}")
                    
                    # Process each frame separately
                    frame_descriptions = []
                    for frame_idx, frame in enumerate(frames):
                        # Format prompt with frame index
                        frame_prompt = fq.format(index=frame_idx + 1) if "{index}" in fq else fq
                        
                        # Create content with single image
                        content = [
                            {"type": "image", "image": frame},
                            {"type": "text", "text": frame_prompt}
                        ]
                        messages = [{"role": "user", "content": content}]
                        
                        # Preparation for inference
                        inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        inputs = inputs.to(model.device)
                        
                        # Generate for this frame
                        with torch.no_grad():
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                generated_ids = model.generate(
                                    **inputs,
                                    max_new_tokens=args.max_new_tokens,
                                    num_beams=1,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    top_k=50,
                                    repetition_penalty=1.1,
                                    use_cache=False
                                )
                                
                                generated_ids_trimmed = [
                                    out_ids[len(in_ids):] 
                                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                ]
                                
                                frame_output = processor.batch_decode(
                                    generated_ids_trimmed,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True
                                )[0]
                        
                        frame_descriptions.append(f"Frame {frame_idx + 1}: {frame_output}")
                        
                        # Clear cache after each frame
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Combine all frame descriptions
                    output = "\n\n".join(frame_descriptions)
                    
                except Exception as e:
                    print(f"\n⚠️  [{idx}/{len(data_dict)}] Failed to extract/process frames: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "video": video_file,
                        "model_output": f"ERROR: Frame processing failed: {str(e)}",
                        "ground_truth": ground_truth
                    })
                    continue
            else:
                # Use video directly
                video_content = {"type": "video", "video": video_path}
                # Add fps if provided
                if args.video_fps is not None:
                    video_content["fps"] = args.video_fps
                    if idx <= 3:  # Print for first few samples
                        print(f"\n📹 [{idx}/{len(data_dict)}] Using video input with fps={args.video_fps} for {video_file}")
                else:
                    if idx <= 3:  # Print for first few samples
                        print(f"\n📹 [{idx}/{len(data_dict)}] Using video input (no fps specified, auto-detect) for {video_file}")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            video_content,
                            {"type": "text", "text": fq}
                        ]
                    }
                ]
            
                # Preparation for inference using Qwen3-VL API
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(model.device)
                
                # Generate
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            num_beams=1,  # Use 1 to avoid video input expansion issues
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=50,
                            repetition_penalty=1.1,
                            use_cache=False  # Disable cache to avoid dimension issues
                        )
                        
                        # Trim input tokens from output
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] 
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        
                        output = processor.batch_decode(
                            generated_ids_trimmed,
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
                "ground_truth": gt
            })
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"qwen3vl_results_{timestamp}.json"
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
            # Auto-detect evaluation path
            script_dir_eval = os.path.dirname(os.path.abspath(__file__))
            project_root_eval = os.path.abspath(os.path.join(script_dir_eval, '../../..'))
            eval_path = os.path.join(project_root_eval, 'evaluation')
            if os.path.exists(eval_path):
                sys.path.append(eval_path)
            else:
                sys.path.append('/home/mh2803/projects/sign_language_llm/evaluation')
            from ssvp_evaluation import comprehensive_evaluation, print_evaluation_results
            
            eval_results = comprehensive_evaluation(references, predictions)
            print_evaluation_results(eval_results, "Qwen3-VL")
            
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
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on test set")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                       help="Path to checkpoint directory (optional, if not provided, will use base model)")
    parser.add_argument("--model-base", type=str, 
                       default="Qwen/Qwen3-VL-2B-Instruct",
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
    parser.add_argument("--video-fps", type=int, default=None,
                       help="FPS for video processing (optional, Qwen3-VL may auto-detect)")
    parser.add_argument("--use-frames", action="store_true",
                       help="Extract frames from video and use as images instead of video input")
    parser.add_argument("--num-frames", type=int, default=None,
                       help="Number of frames to extract when using --use-frames (if None, use fps-based sampling)")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save extracted frames to disk (only works with --use-frames)")
    
    args = parser.parse_args()
    
    # Validate paths
    if args.checkpoint_path is not None and not os.path.exists(args.checkpoint_path):
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

