#!/usr/bin/env python3
"""
Classify How2Sign video captions into 10 topics using LLM.
"""

import json
import os
import sys
import re
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Set environment variable for torch < 2.6 to allow unsafe loading
# This needs to be set before importing torch/transformers
os.environ['TORCH_ALLOW_UNSAFE_LOAD'] = '1'

# Add qwenvl source path to sys.path BEFORE importing transformers
# This is critical for Qwen2.5-VL model support
script_dir = os.path.dirname(os.path.abspath(__file__))
qwen2vl_path = os.path.join(script_dir, 'qwenvl', 'Qwen2-VL-Finetune', 'src')
if os.path.exists(qwen2vl_path) and qwen2vl_path not in sys.path:
    sys.path.insert(0, qwen2vl_path)
    print(f"📁 Added qwenvl source path: {qwen2vl_path}")

# Topic categories
TOPICS = [
    "Cars and Other Vehicle",
    "Games",
    "Arts and Entertainment",
    "Personal Care and Style",
    "Food and Drinks",
    "Education and Communication",
    "Home and Garden",
    "Pets and Animals",
    "Hobbies and Crafts",
    "Sports and Fitness"
]


def classify_with_qwen(
    video_text: str,
    model=None,
    processor=None,
    device="cuda"
) -> List[int]:
    """Classify video content using Qwen2.5-VL model."""
    import torch
    
    if model is None or processor is None:
        raise ValueError("Model and processor must be provided for Qwen classification")
    
    # Create classification prompt
    topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(TOPICS)])
    
    prompt = f"""You are a content classifier. Classify the following instructional video content into one or more of these 10 topics:

{topics_list}

Video content: "{video_text}"

Instructions:
1. Read the video content carefully
2. Identify which topic(s) from 1-10 best match the content
3. Output ONLY a JSON list of topic numbers (e.g., [3] or [5, 7] or [2, 6, 9])
4. Do NOT output any explanation or other text
5. If the content doesn't clearly match any topic, choose the closest one(s)

Output:"""
    
    # Prepare conversation format for Qwen2.5-VL
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant that classifies instructional video content into topics."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        # Apply chat template
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process text only (no images/videos for classification)
        inputs = processor(
            text=[text],
            return_tensors="pt"
        ).to(device)
        
        # Generate classification
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,  # Low temperature for deterministic classification
                do_sample=False,  # Use greedy decoding for consistency
            )
        
        # Decode
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Debug: print first few outputs to see what model is generating
        # (only print for first 5 samples to debug)
        if not hasattr(classify_with_qwen, '_debug_count'):
            classify_with_qwen._debug_count = 0
        
        if classify_with_qwen._debug_count < 5:
            print(f"\n🔍 Debug sample {classify_with_qwen._debug_count + 1}:")
            print(f"   Input text: {video_text[:150]}...")
            # Extract just the new generated part (remove the prompt)
            if prompt in generated_text:
                new_text = generated_text.split(prompt)[-1].strip()
            else:
                new_text = generated_text
            print(f"   Model raw output: {new_text[:300]}")
            print(f"   Full generated text (first 500 chars): {generated_text[:500]}")
            classify_with_qwen._debug_count += 1
        
        # Extract topic numbers from output
        # Look for list format like [1, 2, 3] or [1,2,3]
        topic_numbers = extract_topic_numbers(generated_text)
        
        if classify_with_qwen._debug_count <= 5:
            print(f"   Extracted topic numbers: {topic_numbers}")
            if not topic_numbers:
                print(f"   ⚠️  No topic numbers extracted! Trying alternative parsing...")
                # Try to extract from the new text part only
                if prompt in generated_text:
                    new_text = generated_text.split(prompt)[-1].strip()
                    topic_numbers = extract_topic_numbers(new_text)
                    print(f"   Retry with new text only: {topic_numbers}")
        
        return topic_numbers
        
    except Exception as e:
        print(f"   ⚠️  Error classifying with Qwen: {e}")
        return []


def classify_with_openai(
    video_text: str,
    api_key: str = None
) -> List[int]:
    """Classify video content using OpenAI API."""
    try:
        # Try new OpenAI SDK (v1.0+)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            
            topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(TOPICS)])
            
            prompt = f"""You are a content classifier. Classify the following instructional video content into one or more of these 10 topics:

{topics_list}

Video content: "{video_text}"

Instructions:
1. Read the video content carefully
2. Identify which topic(s) from 1-10 best match the content
3. Output ONLY a JSON list of topic numbers (e.g., [3] or [5, 7] or [2, 6, 9])
4. Do NOT output any explanation or other text
5. If the content doesn't clearly match any topic, choose the closest one(s)

Output:"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies instructional video content into topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            generated_text = response.choices[0].message.content.strip()
            topic_numbers = extract_topic_numbers(generated_text)
            
            return topic_numbers
        except ImportError:
            # OpenAI SDK not installed
            print(f"   ❌ OpenAI SDK not installed. Please install it with: pip install openai")
            return []
            
            topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(TOPICS)])
            
            prompt = f"""You are a content classifier. Classify the following instructional video content into one or more of these 10 topics:

{topics_list}

Video content: "{video_text}"

Instructions:
1. Read the video content carefully
2. Identify which topic(s) from 1-10 best match the content
3. Output ONLY a JSON list of topic numbers (e.g., [3] or [5, 7] or [2, 6, 9])
4. Do NOT output any explanation or other text
5. If the content doesn't clearly match any topic, choose the closest one(s)

Output:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies instructional video content into topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            generated_text = response.choices[0].message.content.strip()
            topic_numbers = extract_topic_numbers(generated_text)
            
            return topic_numbers
    except Exception as e:
        print(f"   ⚠️  OpenAI API error: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_topic_numbers(text: str) -> List[int]:
    """Extract topic numbers from LLM output."""
    # First, try to find JSON list format [1, 2, 3] or [1,2,3]
    # This is the most reliable format
    list_pattern = r'\[([\d,\s]+)\]'
    matches = re.findall(list_pattern, text)
    
    if matches:
        # Use the last match (most likely to be the answer)
        numbers_str = matches[-1]
        try:
            numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
            # Filter valid topic numbers (1-10)
            valid_numbers = [n for n in numbers if 1 <= n <= 10]
            if valid_numbers:
                return valid_numbers
        except ValueError:
            pass
    
    # Try to find individual numbers (fallback)
    # Look for numbers that appear after "Output:" or at the end
    number_pattern = r'\b([1-9]|10)\b'
    numbers = re.findall(number_pattern, text)
    if numbers:
        try:
            valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 10]
            # Remove duplicates while preserving order
            seen = set()
            unique_numbers = []
            for n in valid_numbers:
                if n not in seen:
                    seen.add(n)
                    unique_numbers.append(n)
            # If we found multiple numbers, return them
            # But if we only found one number and it appears multiple times, return it once
            if unique_numbers:
                return unique_numbers
        except ValueError:
            pass
    
    return []


def classify_caption(
    video_text: str,
    use_openai: bool = False,
    openai_key: str = None,
    use_qwen: bool = False,
    qwen_model=None,
    qwen_processor=None,
    device: str = "cuda"
) -> List[int]:
    """
    Classify video caption into topics.
    
    Args:
        video_text: Video caption text
        use_openai: Whether to use OpenAI API
        openai_key: OpenAI API key (if use_openai is True)
        use_qwen: Whether to use Qwen2.5-VL model
        qwen_model: Qwen model instance
        qwen_processor: Qwen processor instance
        device: Device to run model on
    
    Returns:
        List of topic numbers (1-10)
    """
    if use_qwen and qwen_model is not None and qwen_processor is not None:
        return classify_with_qwen(video_text, qwen_model, qwen_processor, device)
    elif use_openai and openai_key:
        return classify_with_openai(video_text, openai_key)
    else:
        print("   ⚠️  No classification method specified, returning empty list")
        return []


def process_json_file(
    input_file: str,
    output_file: str,
    max_samples: int = None,
    use_openai: bool = False,
    openai_key: str = None,
    use_qwen: bool = False,
    qwen_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cuda"
):
    """Process JSON file and classify captions."""
    
    print(f"📖 Loading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"   Limited to {max_samples} samples")
    
    print(f"   Total samples: {len(data)}")
    
    # Load Qwen2.5-VL model if requested
    qwen_model = None
    qwen_processor = None
    if use_qwen:
        print(f"\n🤖 Loading Qwen2.5-VL model: {qwen_model_path}")
        try:
            import torch
            
            # Add qwenvl source path to sys.path if it exists (for Qwen2.5-VL support)
            qwenvl_src_path = os.path.join(os.path.dirname(__file__), 'qwenvl', 'Qwen2-VL-Finetune', 'src')
            if os.path.exists(qwenvl_src_path) and qwenvl_src_path not in sys.path:
                sys.path.insert(0, qwenvl_src_path)
                print(f"   📁 Added qwenvl source path: {qwenvl_src_path}")
            
            # Try to import Qwen2.5-VL class
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                ModelClass = Qwen2_5_VLForConditionalGeneration
                print(f"   ✅ Using Qwen2_5_VLForConditionalGeneration")
            except ImportError:
                # Try using AutoModel with trust_remote_code
                print(f"   ⚠️  Qwen2_5_VLForConditionalGeneration not found, trying AutoModelForCausalLM with trust_remote_code")
                from transformers import AutoModelForCausalLM
                ModelClass = AutoModelForCausalLM
            
            # Import AutoProcessor
            from transformers import AutoProcessor
            qwen_processor = AutoProcessor.from_pretrained(qwen_model_path, trust_remote_code=True)
            
            qwen_model = ModelClass.from_pretrained(
                qwen_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
            qwen_model.eval()
            print(f"   ✅ Model loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading Qwen model: {e}")
            import traceback
            traceback.print_exc()
            print(f"   💡 Tip: You may need to:")
            print(f"      1. Update transformers: pip install --upgrade transformers")
            print(f"      2. Or ensure qwenvl/Qwen2-VL-Finetune/src is in your PYTHONPATH")
            print(f"      3. Or install from source: pip install git+https://github.com/huggingface/transformers.git")
            print(f"   Exiting...")
            return
    
    print(f"\n🔄 Processing samples...")
    if use_qwen:
        print(f"   Using Qwen2.5-VL model for classification")
    elif use_openai:
        print(f"   Using OpenAI API for classification")
    else:
        print(f"   ⚠️  No classification method specified!")
        return
    
    results = []
    classification_stats = {i+1: 0 for i in range(10)}  # Count occurrences of each topic
    
    for idx, item in enumerate(tqdm(data, desc="Classifying")):
        # Find GPT response (caption)
        gpt_value = None
        for conv in item.get('conversations', []):
            if conv.get('from') == 'gpt':
                gpt_value = conv.get('value', '')
                break
        
        if not gpt_value:
            print(f"   ⚠️  No GPT value found for item {idx}")
            continue
        
        # Classify caption
        topic_numbers = classify_caption(
            gpt_value,
            use_openai,
            openai_key,
            use_qwen,
            qwen_model,
            qwen_processor,
            device
        )
        
        # Update statistics
        for topic_num in topic_numbers:
            if 1 <= topic_num <= 10:
                classification_stats[topic_num] += 1
        
        # Create result entry
        new_item = item.copy()
        new_item['topic_numbers'] = topic_numbers
        new_item['topic_names'] = [TOPICS[t-1] for t in topic_numbers if 1 <= t <= 10]
        
        results.append(new_item)
    
    # Print statistics
    print(f"\n📊 Classification Statistics:")
    print(f"   Total samples classified: {len(results)}")
    print(f"   Topic distribution:")
    for i, topic in enumerate(TOPICS):
        topic_num = i + 1
        count = classification_stats[topic_num]
        percentage = (count / len(results) * 100) if len(results) > 0 else 0
        print(f"      {topic_num}. {topic}: {count} ({percentage:.1f}%)")
    
    # Save results
    print(f"\n💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processing complete!")
    print(f"   Processed: {len(results)} samples")
    print(f"   Output file: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify How2Sign video captions into topics")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--use-openai", action="store_true",
                        help="Use OpenAI API for classification (requires --openai-key)")
    parser.add_argument("--openai-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--use-qwen", action="store_true",
                        help="Use Qwen2.5-VL model for classification")
    parser.add_argument("--qwen-model-path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Path to Qwen2.5-VL model (default: Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run model on (default: cuda)")
    
    args = parser.parse_args()
    
    # Get OpenAI key from args or environment
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    
    process_json_file(
        args.input,
        args.output,
        max_samples=args.max_samples,
        use_openai=args.use_openai,
        openai_key=openai_key,
        use_qwen=args.use_qwen,
        qwen_model_path=args.qwen_model_path,
        device=args.device
    )

