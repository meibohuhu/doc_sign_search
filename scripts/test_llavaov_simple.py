#!/usr/bin/env python3
"""
Test script for simplified LLaVA-OV inference
"""
import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
#### mhu update 09/19/2025 Direct llavaov_simple approach, Object-oriented approach with error handling

# Add current directory to Python path
sys.path.insert(0, '/local1/mhu/LLaVANeXT_RC')

from playground.demo.llavaov_simple import LLaVA

def extract_frames(video_path, num_frames=8):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="LLaVA-OneVision 0.5B ASL Video Inference")
    parser.add_argument("--question-file", required=True, help="Path to JSON file with video data")
    parser.add_argument("--video-folder", required=True, help="Path to video folder")
    parser.add_argument("--output-file", default="llavaov_results.json", help="Output JSON file")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to extract")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    print("🚀 Starting simplified LLaVA-OV ASL inference test...")
    
    # Load the model
    print("📥 Loading LLaVA-OneVision 0.5B model...")
    model = LLaVA()
    
    # Load test data
    with open(args.question_file, 'r') as f:
        data = json.load(f)
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"📁 Processing {len(data)} samples from {args.question_file}")
    print(f"🎥 Video folder: {args.video_folder}")
    
    results = []
    
    for sample in tqdm(data, desc="Processing videos"):
        try:
            video_file = sample.get("video")
            if not video_file:
                continue
            
            video_path = os.path.join(args.video_folder, video_file)
            
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"⚠️  Video not found: {video_path}")
                results.append({
                    "video": video_file,
                    "prompt": sample.get("conversations", [{}])[0].get("value", ""),
                    "model_output": "ERROR: Video file not found"
                })
                continue
            
            # Extract frames
            frames = extract_frames(video_path, args.num_frames)
            if not frames:
                print(f"⚠️  No frames extracted from: {video_path}")
                results.append({
                    "video": video_file,
                    "prompt": sample.get("conversations", [{}])[0].get("value", ""),
                    "model_output": "ERROR: No frames extracted"
                })
                continue
            
            print(f"🎬 Processing {len(frames)} frames from {video_file}")
            
            # Get ASL translation using simplified LLaVA-OV
            translations = model(frames)
            
            # Combine translations from all frames (take the first valid one or combine)
            valid_translations = [t for t in translations if not t.startswith("ERROR")]
            if valid_translations:
                combined_translation = valid_translations[0]  # Take the first good translation
            else:
                combined_translation = "No valid translation generated"
            
            # Get ground truth for comparison
            ground_truth = sample.get("conversations", [{}])[1].get("value", "")
            
            print(f"📝 Ground truth: {ground_truth[:100]}...")
            print(f"🤖 Generated: {combined_translation[:100]}...")
            print("-" * 50)
            
            results.append({
                "video": video_file,
                "prompt": sample.get("conversations", [{}])[0].get("value", ""),
                "ground_truth": ground_truth,
                "model_output": combined_translation,
                "frame_translations": translations  # Individual frame translations
            })
            
        except Exception as e:
            print(f"❌ Error processing {sample.get('video', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "video": sample.get("video"),
                "prompt": sample.get("conversations", [{}])[0].get("value", ""),
                "model_output": f"ERROR: {str(e)}"
            })
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Simplified LLaVA-OV inference test completed!")
    print(f"📄 Results saved to: {args.output_file}")
    print(f"📊 Processed {len(results)} samples")

if __name__ == "__main__":
    main()
