#!/usr/bin/env python3
"""
Script to load Qwen2.5-VL model with LoRA adapter from checkpoint-400 for inference and evaluation.
"""

import os
import sys
import torch
from pathlib import Path

# Add the src directory to Python path
sys.path.append('/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src')

from utils import load_pretrained_model

def load_qwen25vl_lora_model(checkpoint_path, model_base=None):
    """
    Load Qwen2.5-VL model with LoRA adapter.
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint (e.g., checkpoint-400)
        model_base: Base model path (if None, will be inferred)
    
    Returns:
        processor, model
    """
    
    # Set up paths
    if model_base is None:
        model_base = "Qwen/Qwen2.5-VL-3B-Instruct"  # Base model
    
    print(f"Loading base model: {model_base}")
    print(f"Loading LoRA checkpoint: {checkpoint_path}")
    
    # Load the model with LoRA adapter
    processor, model = load_pretrained_model(
        model_path=checkpoint_path,
        model_base=model_base,
        model_name="qwen2.5-vl-3b-instruct",
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",
        use_flash_attn=True
    )
    
    print("Model loaded successfully!")
    return processor, model

def run_inference(processor, model, video_path, question):
    """
    Run inference on a video with a question.
    
    Args:
        processor: Model processor
        model: Loaded model
        video_path: Path to video file
        question: Question to ask about the video
    
    Returns:
        Generated response
    """
    
    # Prepare the conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = processor(text, [video_path], return_tensors="pt")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        image_inputs = {k: v.to(model.device) for k, v in image_inputs.items()}
        video_inputs = {k: v.to(model.device) for k, v in video_inputs.items()}
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **image_inputs,
            **video_inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(image_inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def batch_evaluate(processor, model, test_data_path, output_path=None):
    """
    Run batch evaluation on test data.
    
    Args:
        processor: Model processor
        model: Loaded model
        test_data_path: Path to test JSON file
        output_path: Path to save results
    
    Returns:
        Results dictionary
    """
    import json
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    results = []
    
    print(f"Evaluating {len(test_data)} samples...")
    
    for i, item in enumerate(test_data):
        try:
            video_path = item.get('video', '')
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            
            print(f"Processing {i+1}/{len(test_data)}: {video_path}")
            
            # Run inference
            response = run_inference(processor, model, video_path, question)
            
            result = {
                'video': video_path,
                'question': question,
                'ground_truth': answer,
                'prediction': response,
                'index': i
            }
            results.append(result)
            
            print(f"Q: {question}")
            print(f"A: {response}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return results

def main():
    # Configuration
    checkpoint_path = "/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/output/qwen2vl_ssvp_2xa100_10fps/checkpoint-400"
    model_base = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        base_dir = Path(checkpoint_path).parent
        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    print(f"  - {item}")
        return
    
    # Load model
    try:
        processor, model = load_qwen25vl_lora_model(checkpoint_path, model_base)
        
        # Example inference
        video_path = "/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/videos/00001.mp4"
        question = "What sign language is being performed in this video?"
        
        if os.path.exists(video_path):
            print(f"\nRunning inference on: {video_path}")
            print(f"Question: {question}")
            
            response = run_inference(processor, model, video_path, question)
            print(f"Response: {response}")
        else:
            print(f"Video not found: {video_path}")
            
        # Batch evaluation (optional)
        test_data_path = "/home/mh2803/projects/sign_language_llm/vanshika/asl_test/test_ssvp.json"
        if os.path.exists(test_data_path):
            print(f"\nRunning batch evaluation on: {test_data_path}")
            results = batch_evaluate(
                processor, 
                model, 
                test_data_path,
                output_path=f"qwen25vl_lora_checkpoint400_results.json"
            )
            print(f"Evaluation completed. Processed {len(results)} samples.")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




