#!/usr/bin/env python3
"""
GPT-4 Vision / Gemini Evaluation Script for How2Sign
Uses OpenAI GPT-4 Vision API or Google Gemini API to process video frames and answer questions
"""

import os
import sys
import json
import re
import cv2
import numpy as np
import base64
from PIL import Image
import io
import warnings
import argparse
from datetime import datetime
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

warnings.filterwarnings("ignore")


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


def encode_image_to_base64(image, max_size=1024):
    """
    Encode PIL Image to base64 string.
    Optionally resize if image is too large.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height) in pixels
    
    Returns:
        base64 encoded string
    """
    # Resize if too large (to save API costs and ensure compatibility)
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


def call_gpt4_vision_api(
    frames,
    prompt,
    api_key,
    model="gpt-4o",
    max_tokens=10240,
    temperature=0.7,
    detail="low"
):
    """
    Call GPT-4 Vision API with multiple frames.
    
    Args:
        frames: List of PIL Image objects
        prompt: Text prompt/question
        api_key: OpenAI API key
        model: Model to use (default: gpt-4o)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        detail: Image detail level ("low", "high", or "auto")
    
    Returns:
        Response text from the model
    """
    # Validate API key format before attempting to use it
    if not api_key or not isinstance(api_key, str):
        raise Exception("OpenAI API key is required and must be a string")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("OpenAI SDK not installed. Please install with: pip install openai")
    
    # Create system instruction (similar to Gemini)
    system_instruction_text = "Your knowledge cutoff date is January 2025."
    
    # Prepare content: start with prompt text, then add images (similar to Gemini)
    content = []
    
    # Add prompt text first (similar to Gemini's content_parts = [prompt])
    content.append({
        "type": "text",
        "text": prompt
    })
    
    # Add all frames as images (similar to Gemini's content_parts.append(frame))
    for idx, frame in enumerate(frames):
        img_base64 = encode_image_to_base64(frame, max_size=1024)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}",
                "detail": detail
            }
        })
    
    # Prepare messages with system instruction (similar to Gemini's system_instruction)
    messages = [
        {
            "role": "system",
            "content": system_instruction_text
        },
        {
            "role": "user",
            "content": content
        }
    ]
    
    try:
        # Call API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract text response
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            raise Exception("No text response received from GPT-4 Vision API")
    
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages for common issues (similar to Gemini)
        if "401" in error_msg or "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
            raise Exception(
                f"❌ GPT-4 Vision API authentication failed: {error_msg}\n\n"
                f"Please verify:\n"
                f"1. ✅ Your API key is correct and active\n"
                f"2. ✅ Get a valid API key from: https://platform.openai.com/api-keys\n"
                f"3. ⚠️  Make sure it's a valid OpenAI API key"
            )
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise Exception(f"GPT-4 Vision API rate limit exceeded: {error_msg}")
        elif "quota" in error_msg.lower():
            raise Exception(f"GPT-4 Vision API quota exceeded: {error_msg}")
        else:
            raise Exception(f"GPT-4 Vision API call failed: {error_msg}")


def call_gemini_api(
    frames,
    prompt,
    api_key,
    model="gemini-1.5-pro",
    max_tokens=128,
    temperature=0.7
):
    """
    Call Google Gemini API with multiple frames.
    
    Args:
        frames: List of PIL Image objects
        prompt: Text prompt/question
        api_key: Google Gemini API key
        model: Model to use (default: gemini-1.5-pro)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0-1.0)
    
    Returns:
        Response text from the model
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Google Generative AI SDK not installed. Please install with: pip install google-generativeai")
    
    # Validate API key format before attempting to use it
    if not api_key or not isinstance(api_key, str):
        raise Exception("Gemini API key is required and must be a string")
    
    # Configure API key
    # Note: Must use API key from Google AI Studio (https://aistudio.google.com/app/apikey)
    # NOT from Google Cloud Console
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise Exception(
            f"Failed to configure Gemini API key: {str(e)}\n"
            f"Please ensure you're using an API key from Google AI Studio: https://aistudio.google.com/app/apikey"
        )
    
    # Create system instruction
    system_instruction_text = "Your knowledge cutoff date is January 2025."
    
    # Initialize model with system instruction
    try:
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction_text
        )
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
            raise Exception(
                f"Gemini API authentication failed. This usually means:\n"
                f"1. ❌ Your API key is invalid or expired\n"
                f"2. ❌ You're using a Google Cloud API key instead of Google AI Studio key\n\n"
                f"✅ Solution: Get a new API key from Google AI Studio:\n"
                f"   https://aistudio.google.com/app/apikey\n\n"
                f"Original error: {error_msg}"
            )
        else:
            raise Exception(f"Failed to initialize Gemini model: {error_msg}")
    
    # Prepare content: start with prompt text, then add images
    content_parts = [prompt]
    
    # Add all frames as images (Gemini accepts PIL Images directly)
    for frame in frames:
        # Resize if too large (Gemini has limits on image size)
        width, height = frame.size
        max_size = 2048  # Gemini supports larger images than GPT-4
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        content_parts.append(frame)
    
    try:
        # Configure generation parameters
        # Build safety settings using dictionary format (most compatible)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Build generation config as dictionary
        # Note: seed is not supported in GenerationConfig, removed
        # Use function parameters (max_tokens, temperature) instead of hardcoded values
        generation_config_dict = {
            "temperature": temperature,
            "top_p": 0.95,
            "max_output_tokens": max_tokens,
        }
        
        # Prepare arguments for generate_content (system_instruction is set during model creation)
        generate_kwargs = {
            "generation_config": generation_config_dict,
            "safety_settings": safety_settings,
        }
        
        # Try to use types if available for better type safety
        try:
            # Try to use proper types if they exist
            if hasattr(genai, 'types'):
                # Try to create GenerationConfig using types
                try:
                    # Note: seed is not a valid field for GenerationConfig
                    # Use function parameters instead of hardcoded values
                    generation_config = genai.types.GenerationConfig(
                        temperature=temperature,
                        top_p=0.95,
                        max_output_tokens=max_tokens,
                    )
                    generate_kwargs["generation_config"] = generation_config
                except:
                    pass  # Fall back to dict
                
                # Try to use SafetySetting types if available
                try:
                    if hasattr(genai.types, 'HarmCategory') and hasattr(genai.types, 'HarmBlockThreshold'):
                        safety_settings_typed = [
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                            },
                            {
                                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
                            }
                        ]
                        generate_kwargs["safety_settings"] = safety_settings_typed
                except:
                    pass  # Fall back to string dict
                
                # Try to add thinking_config if supported
                try:
                    if hasattr(genai.types, 'ThinkingConfig'):
                        thinking_config = genai.types.ThinkingConfig(thinking_level="HIGH")
                        generate_kwargs["thinking_config"] = thinking_config
                except:
                    pass  # Thinking config not supported for this model
        except:
            pass  # Types not available, use dict format
        
        # Generate content with rate limit retry logic
        max_retries = 3
        retry_delay = 30  # Wait 30 seconds on rate limit
        
        for attempt in range(max_retries):
            try:
                response = gemini_model.generate_content(
                    content_parts,
                    **generate_kwargs
                )
                
                # Extract text response
                if hasattr(response, 'text') and response.text:
                    return response.text.strip()
                else:
                    # Handle cases where response might have parts
                    if hasattr(response, 'parts'):
                        text_parts = [part.text for part in response.parts if hasattr(part, 'text') and part.text]
                        if text_parts:
                            return " ".join(text_parts).strip()
                    raise Exception("No text response received from Gemini API")
            
            except Exception as e:
                error_msg = str(e)
                # Check for rate limit error (429)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️  Rate limit detected, waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception(f"Gemini API rate limit exceeded after {max_retries} attempts: {error_msg}")
                else:
                    # For other errors, raise immediately
                    raise
    
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages for common issues
        if "401" in error_msg or "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
            # Check for the specific OAuth2 error that indicates wrong key type
            if "oauth2" in error_msg.lower() or "api keys are not supported" in error_msg.lower():
                raise Exception(
                    f"❌ Authentication Error: OAuth2 Required\n"
                    f"The API key you're using requires OAuth2 authentication, but this script uses the simpler Google AI Studio API.\n\n"
                    f"💡 Quick Fix (Recommended):\n"
                    f"   Get a Google AI Studio API key (free, no OAuth2 needed):\n"
                    f"   1. Visit: https://aistudio.google.com/app/apikey\n"
                    f"   2. Click 'Create API Key'\n"
                    f"   3. Copy the key and set: export GEMINI_API_KEY='your-key'\n\n"
                    f"📝 Original error: {error_msg[:200]}"
                )
            else:
                raise Exception(
                    f"❌ Gemini API authentication failed: {error_msg}\n\n"
                    f"Please verify:\n"
                    f"1. ✅ Your API key is correct and active\n"
                    f"2. ✅ Get a valid API key from: https://aistudio.google.com/app/apikey\n"
                    f"3. ⚠️  Make sure it's a valid Gemini API key"
                )
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise Exception(f"Gemini API rate limit exceeded: {error_msg}")
        elif "quota" in error_msg.lower():
            raise Exception(f"Gemini API quota exceeded: {error_msg}")
        else:
            raise Exception(f"Gemini API call failed: {error_msg}")


def parse_model_output_to_json(model_output):
    """
    Parse model output text into JSON format with statement1, statement2, statement3.
    
    Args:
        model_output: Raw text output from the model
    
    Returns:
        dict: Parsed JSON with keys statement1, statement2, statement3, or None if parsing fails
        str: Error message if parsing fails, None otherwise
    """
    if not model_output or model_output.startswith("ERROR"):
        return None, "Output is empty or error"
    
    try:
        # Try to parse as JSON first (in case model already returns JSON)
        try:
            parsed = json.loads(model_output)
            if isinstance(parsed, dict) and "statement1" in parsed and "statement2" in parsed:
                # Ensure statement3 exists
                if "statement3" not in parsed:
                    parsed["statement3"] = ""
                return parsed, None
        except json.JSONDecodeError:
            pass  # Not JSON, continue with text parsing
        
        # Parse text format - look for Statement 1, Statement 2, Statement 3
        # Handle various formats: Statement1, Statement 1, statement1, statement 1, etc.
        result = {
            "statement1": "",
            "statement2": "",
            "statement3": ""
        }
        
        # Try different patterns - handle case-insensitive matching
        # Pattern matches: Statement1, Statement 1, statement1, statement 1, etc.
        # Also handles with or without colon and spaces
        patterns = [
            # Match Statement1 or Statement 1 (with optional space, colon, and spaces)
            (r"(?i)statement\s*1\s*:?\s*(.+?)(?=(?i)statement\s*[23]|$)", "statement1"),
            (r"(?i)statement\s*2\s*:?\s*(.+?)(?=(?i)statement\s*3|$)", "statement2"),
            (r"(?i)statement\s*3\s*:?\s*(.+?)(?=(?i)statement\s*[123]|$)", "statement3"),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, model_output, re.DOTALL | re.MULTILINE)
            if match:
                text = match.group(1).strip()
                # Remove brackets if present [description]
                text = re.sub(r'^\[|\]$', '', text).strip()
                # Remove leading/trailing quotes if present
                text = re.sub(r'^["\']|["\']$', '', text).strip()
                result[key] = text
        
        # If we found at least statement1 and statement2, consider it successful
        if result["statement1"] or result["statement2"]:
            return result, None
        else:
            return None, "Could not extract statements from output"
    
    except Exception as e:
        return None, f"Parsing error: {str(e)}"


def process_single_sample(args, source, idx, total_samples, api_key, question_prompt, use_gemini, results_lock):
    """
    Process a single sample (extract frames and call API).
    Thread-safe function for parallel processing.
    
    Returns:
        dict: Result dictionary with video, model_output, and ground_truth
    """
    try:
        video_file = source["video"]
        video_path = os.path.join(args.video_folder, video_file)
        
        # Extract ground truth from conversations
        conversations = source.get('conversations', [])
        if len(conversations) >= 2:
            ground_truth = conversations[1].get('value', '')
        else:
            ground_truth = source.get('answer', source.get('ground_truth', ''))
        
        if not os.path.exists(video_path):
            return {
                "video": video_file,
                "model_output": "ERROR: Video not found",
                "ground_truth": ground_truth,
                "idx": idx,
                "statements": {
                    "statement1": "",
                    "statement2": "",
                    "statement3": ""
                },
                "parse_error": "Video not found"
            }
        
        # Extract frames from video
        try:
            frames_output_dir = None
            if args.save_frames:
                frames_output_dir = os.path.join(args.out_dir, "extracted_frames")
                os.makedirs(frames_output_dir, exist_ok=True)
            
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
            
        except Exception as e:
            return {
                "video": video_file,
                "model_output": f"ERROR: Frame extraction failed: {str(e)}",
                "ground_truth": ground_truth,
                "idx": idx,
                "statements": {
                    "statement1": "",
                    "statement2": "",
                    "statement3": ""
                },
                "parse_error": f"Frame extraction failed: {str(e)}"
            }
        
        # Call API (GPT-4 or Gemini)
        try:
            if use_gemini:
                output = call_gemini_api(
                    frames=frames,
                    prompt=question_prompt,
                    api_key=api_key,
                    model=args.model,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
            else:
                output = call_gpt4_vision_api(
                    frames=frames,
                    prompt=question_prompt,
                    api_key=api_key,
                    model=args.model,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    detail=args.image_detail
                )
        
        except Exception as e:
            output = f"ERROR: API call failed: {str(e)}"
        
        # Parse output to JSON format
        parsed_output, parse_error = parse_model_output_to_json(output)
        
        result = {
            "video": video_file,
            "model_output": output,  # Keep original output for debugging
            "ground_truth": ground_truth,
            "idx": idx
        }
        
        if parsed_output:
            # Add parsed JSON output
            result["statements"] = parsed_output
        else:
            # Add error information but continue processing
            result["statements"] = {
                "statement1": "",
                "statement2": "",
                "statement3": ""
            }
            result["parse_error"] = parse_error or "Failed to parse output"
            # Print warning but don't stop
            print(f"\n⚠️  [{idx}/{total_samples}] Failed to parse output for {video_file}: {parse_error}")
            print(f"   Raw output: {output[:200]}...")  # Print first 200 chars
        
        return result
    
    except Exception as e:
        conversations = source.get('conversations', [])
        if len(conversations) >= 2:
            gt = conversations[1].get('value', 'unknown')
        else:
            gt = source.get('answer', source.get('ground_truth', 'unknown'))
        return {
            "video": source.get("video", "unknown"),
            "model_output": f"ERROR: {str(e)}",
            "ground_truth": gt,
            "idx": idx,
            "statements": {
                "statement1": "",
                "statement2": "",
                "statement3": ""
            },
            "parse_error": f"Processing error: {str(e)}"
        }


def save_results_to_file(results, output_path, lock=None):
    """
    Save results to JSON file in a thread-safe manner.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
        lock: Optional threading lock for thread-safe writing
    """
    def _write():
        # Write to temporary file first, then rename (atomic operation)
        temp_path = output_path + ".tmp"
        with open(temp_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        # Atomic rename
        os.replace(temp_path, output_path)
    
    if lock:
        with lock:
            _write()
    else:
        _write()


def eval_model(args):
    """Main evaluation function."""
    
    # Determine API type and check API key
    use_gemini = args.api_type.lower() == "gemini" if args.api_type else False
    
    if use_gemini:
        api_key = args.api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ ERROR: Gemini API key not provided!")
            print("   Please provide --api-key or set GEMINI_API_KEY/GOOGLE_API_KEY environment variable")
            print("   Get your API key from: https://aistudio.google.com/app/apikey")
            return
        
        print(f"🔑 Using Gemini API key: {api_key[:10]}...")
    else:
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ ERROR: OpenAI API key not provided!")
            print("   Please provide --api-key or set OPENAI_API_KEY environment variable")
            return
        print(f"🔑 Using OpenAI API key: {api_key[:10]}...")
    
    print(f"🤖 API Type: {args.api_type or 'OpenAI'}")
    print(f"🤖 Model: {args.model}\n")
    
    # Load test data
    print(f"📂 Loading test data from: {args.question_file}")
    with open(args.question_file, 'r', encoding='utf-8') as f:
        # Try to load as JSON first (array format)
        try:
            data_dict = json.load(f)
            if not isinstance(data_dict, list):
                # If it's a single object, wrap it in a list
                data_dict = [data_dict]
        except json.JSONDecodeError:
            # Try JSONL format (one JSON object per line)
            f.seek(0)
            data_dict = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data_dict.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Failed to parse line: {line[:100]}... Error: {e}")
                        continue
    
    if args.max_samples:
        data_dict = data_dict[:args.max_samples]
        print(f"   Limited to {args.max_samples} samples")
    
    print(f"   Total samples: {len(data_dict)}\n")
    
    if len(data_dict) == 0:
        print("❌ ERROR: No data loaded!")
        return
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create output file path early for periodic saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"gpt4v_results_{timestamp}.json"
    output_path = os.path.join(args.out_dir, output_file)
    
    results = []
    references = []
    predictions = []
    
    # Lock for thread-safe file writing
    file_write_lock = threading.Lock()
    
    # Batch save interval (save every N samples)
    save_interval = 10  # Save every 5 samples
    
    print(f"🎬 Starting evaluation...")
    print(f"💾 Results will be saved periodically to: {output_path}")
    print(f"   (Saving every {save_interval} samples to prevent data loss)\n")
    print(f"{'='*70}\n")
    
    # Define prompt/question
    # question_prompt = """You must answer based on the entire video clip,
    # Base your judgment only on visible hand motion and spatial position, not on sign language meaning.
    # Choose the answer strictly from the provided options.
    # If the condition never occurs in the video, choose the corresponding "none" option.

    # Considering the entire video, did any hand move to a position higher than the shoulder level?

    # Definition: a hand is above the shoulder if it is visibly higher than the shoulder line for a short continuous period.

    # Answer with one option only, no other words:
    # none / left / right / both
    # """
    question_prompt = """
            You are an ASL motion-description annotator.

            Describe the most important hand movements in the video using exactly TWO statements.

            Format:
            Statement 1: [Describe the most significant hand movement, including handshape, palm orientation, location, movement path, and finger positions for both hands if relevant]
            Statement 2: [Describe the second most important hand movement or interaction, including handshape, palm orientation, location, movement path, and hand interaction if relevant]
            Statement 3: [Describe if two hands touch each other or not]
            
            Rules:
            - Focus on the TWO most important/distinctive movements in the video.
            - DO NOT infer meaning; only describe observable motion.
            - Use factual, objective language only.
            - Each statement should be a complete sentence describing one movement.
            - FORBIDDEN WORDS: 'indicate', 'suggests', 'seems', 'appears', 'may', 'might', 'could', 'possibly', 'probably', 'likely', 'represents', 'signifies', 'means', 'communicates', 'expresses', 'implies', 'conveys', 'shows that', 'demonstrates that'.
            """
    
    
    # Allow custom prompt override
    if args.prompt:
        question_prompt = args.prompt
    
    # Thread-safe lock for results
    results_lock = threading.Lock()
    
    # Process samples - use multithreading for Gemini, sequential for GPT-4
    if use_gemini:
        # Use multithreading for Gemini API
        # Get max_workers from args, default to 5 if not specified
        if args.max_workers is not None:
            max_workers = min(args.max_workers, len(data_dict))
        else:
            max_workers = min(5, len(data_dict))  # Default: 5 concurrent threads
        print(f"🚀 Using multithreading with {max_workers} workers for Gemini API")
        print(f"   Rate limit handling: Auto-wait 30s on 429 errors\n")
        
        # Prepare sample data with indices
        sample_data = [(args, source, idx + 1, len(data_dict), api_key, question_prompt, use_gemini, results_lock) 
                      for idx, source in enumerate(data_dict)]
        
        # Process samples in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_single_sample, *sample_args): sample_args[2] 
                for sample_args in sample_data
            }
            
            # Collect results as they complete and save periodically
            sample_results = {}
            completed_count = 0
            
            for future in tqdm(as_completed(future_to_sample), total=len(data_dict), desc="Evaluating"):
                idx = future_to_sample[future]
                try:
                    result = future.result()
                    sample_results[idx] = result
                except Exception as e:
                    print(f"\n⚠️  [{idx}/{len(data_dict)}] Thread error: {e}")
                    sample_results[idx] = {
                        "video": "unknown",
                        "model_output": f"ERROR: Thread error: {str(e)}",
                        "ground_truth": "unknown",
                        "idx": idx,
                        "statements": {
                            "statement1": "",
                            "statement2": "",
                            "statement3": ""
                        },
                        "parse_error": f"Thread error: {str(e)}"
                    }
                
                completed_count += 1
                
                # Periodic save as results come in (save every N completed samples)
                if completed_count % save_interval == 0 or completed_count == len(data_dict):
                    # Build current results in order (thread-safe)
                    with file_write_lock:
                        # Rebuild results list from all completed samples
                        sorted_indices = sorted([i for i in sample_results.keys()])
                        current_results = []
                        current_references = []
                        current_predictions = []
                        
                        for sorted_idx in sorted_indices:
                            result = sample_results[sorted_idx]
                            current_references.append(result["ground_truth"])
                            current_predictions.append(result["model_output"])
                            
                            # Build result dict with parsed statements
                            result_dict = {
                                "video": result["video"],
                                "model_output": result["model_output"],
                                "ground_truth": result["ground_truth"]
                            }
                            
                            # Add parsed statements if available
                            if "statements" in result:
                                result_dict["statements"] = result["statements"]
                            if "parse_error" in result:
                                result_dict["parse_error"] = result["parse_error"]
                            
                            current_results.append(result_dict)
                        
                        # Update global lists (replace, not append, to avoid duplicates)
                        results.clear()
                        results.extend(current_results)
                        references.clear()
                        references.extend(current_references)
                        predictions.clear()
                        predictions.extend(current_predictions)
                        
                        # Save to file
                        save_results_to_file(results, output_path, None)  # Lock already acquired
                    
                    if completed_count % save_interval == 0:
                        print(f"\n💾 Progress saved: {completed_count}/{len(data_dict)} samples to {output_path}")
        
        # Final processing: ensure all results are in the correct order
        sorted_results = [sample_results[i] for i in sorted(sample_results.keys())]
        
        # Update final results list
        results.clear()
        references.clear()
        predictions.clear()
        for result in sorted_results:
            references.append(result["ground_truth"])
            predictions.append(result["model_output"])
            
            # Build result dict with parsed statements
            result_dict = {
                "video": result["video"],
                "model_output": result["model_output"],
                "ground_truth": result["ground_truth"]
            }
            
            # Add parsed statements if available
            if "statements" in result:
                result_dict["statements"] = result["statements"]
            if "parse_error" in result:
                result_dict["parse_error"] = result["parse_error"]
            
            results.append(result_dict)
            
            # Print first 10 examples
            idx = result.get("idx", 0)
            if idx <= 10:
                print(f"\n{'─'*70}")
                print(f"[{idx}/{len(data_dict)}] {result['video']}")
                print(f"Ground truth: {result['ground_truth']}")
                print(f"Prediction:   {result['model_output']}")
                
                # Check if it's learning
                output = result['model_output']
                gt = result['ground_truth']
                if output.lower() == gt.lower():
                    print(f"✅ EXACT MATCH!")
                elif any(word in output.lower() for word in gt.lower().split()):
                    print(f"✅ Partial match (has some words)")
                else:
                    print(f"⚠️  No obvious match")
    
    else:
        # Sequential processing for GPT-4 (or other APIs)
        for idx, source in enumerate(tqdm(data_dict, desc="Evaluating"), 1):
            try:
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
                        "ground_truth": ground_truth,
                        "statements": {
                            "statement1": "",
                            "statement2": "",
                            "statement3": ""
                        },
                        "parse_error": "Video not found"
                    })
                    continue
                
                # Extract frames from video
                try:
                    frames_output_dir = None
                    if args.save_frames:
                        frames_output_dir = os.path.join(args.out_dir, "extracted_frames")
                        os.makedirs(frames_output_dir, exist_ok=True)
                    
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
                    
                    if idx <= 3:  # Print for first few samples
                        print(f"\n📹 [{idx}/{len(data_dict)}] Extracted {len(frames)} frames from {video_file}")
                        if saved_paths:
                            print(f"   💾 Saved {len(saved_paths)} frames to: {frames_output_dir}")
                    
                except Exception as e:
                    print(f"\n⚠️  [{idx}/{len(data_dict)}] Failed to extract frames: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "video": video_file,
                        "model_output": f"ERROR: Frame extraction failed: {str(e)}",
                        "ground_truth": ground_truth,
                        "statements": {
                            "statement1": "",
                            "statement2": "",
                            "statement3": ""
                        },
                        "parse_error": f"Frame extraction failed: {str(e)}"
                    })
                    continue
                
                # Call API (GPT-4)
                try:
                    if idx <= 3:  # Print for first few samples
                        print(f"   🤖 Calling GPT-4 Vision API with {len(frames)} frames...")
                    
                    output = call_gpt4_vision_api(
                        frames=frames,
                        prompt=question_prompt,
                        api_key=api_key,
                        model=args.model,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        detail=args.image_detail
                    )
                    
                    # Add small delay to avoid rate limiting
                    if idx < len(data_dict):
                        time.sleep(0.5)
                    
                except Exception as e:
                    print(f"\n⚠️  [{idx}/{len(data_dict)}] API call failed: {e}")
                    import traceback
                    traceback.print_exc()
                    output = f"ERROR: API call failed: {str(e)}"
                
                # Parse output to JSON format
                parsed_output, parse_error = parse_model_output_to_json(output)
                
                # Store results
                references.append(ground_truth)
                predictions.append(output)
                
                # Build result dict with parsed statements
                result_dict = {
                    "video": video_file,
                    "model_output": output,
                    "ground_truth": ground_truth
                }
                
                # Add parsed statements if available
                if parsed_output:
                    result_dict["statements"] = parsed_output
                else:
                    # Add error information but continue processing
                    result_dict["statements"] = {
                        "statement1": "",
                        "statement2": "",
                        "statement3": ""
                    }
                    result_dict["parse_error"] = parse_error or "Failed to parse output"
                    # Print warning but don't stop
                    print(f"\n⚠️  [{idx}/{len(data_dict)}] Failed to parse output for {video_file}: {parse_error}")
                    print(f"   Raw output: {output[:200]}...")  # Print first 200 chars
                
                results.append(result_dict)
                
                # Periodic save to prevent data loss
                if idx % save_interval == 0 or idx == len(data_dict):
                    save_results_to_file(results, output_path, file_write_lock)
                    if idx % save_interval == 0:
                        print(f"💾 Progress saved: {idx}/{len(data_dict)} samples")
                
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
                    "ground_truth": gt,
                    "statements": {
                        "statement1": "",
                        "statement2": "",
                        "statement3": ""
                    },
                    "parse_error": f"Processing error: {str(e)}"
                })
                continue
    
    # Wrap final operations in try-finally to ensure results are saved
    try:
        # Final save (ensure all results are saved)
        save_results_to_file(results, output_path, file_write_lock)
        
        print(f"\n{'='*70}")
        print(f"✅ Final results saved: {output_path}")
        print(f"   Total samples: {len(results)}")
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
                    # Try alternative path
                    alt_path = '/home/mh2803/projects/sign_language_llm/evaluation'
                    if os.path.exists(alt_path):
                        sys.path.append(alt_path)
                
                from ssvp_evaluation import comprehensive_evaluation, print_evaluation_results
                
                eval_results = comprehensive_evaluation(references, predictions)
                api_name = "Gemini" if use_gemini else "GPT-4 Vision"
                print_evaluation_results(eval_results, api_name)
                
                # Save metrics
                eval_file = os.path.join(args.out_dir, f"metrics_{timestamp}.json")
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f"\n📊 Metrics saved: {eval_file}")
                
            except Exception as e:
                print(f"\n⚠️  Evaluation metrics error: {e}")
                import traceback
                traceback.print_exc()
        
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
    
    except Exception as e:
        # Handle unexpected errors
        print(f"\n❌ Unexpected error during final operations: {e}")
        import traceback
        traceback.print_exc()
        # Still try to save results if we have any
        if results:
            try:
                save_results_to_file(results, output_path, file_write_lock)
                print(f"\n💾 Emergency save: Saved {len(results)} results to {output_path}")
            except Exception as save_error:
                print(f"\n⚠️  Failed to save results: {save_error}")
    finally:
        # Final save attempt (in case of any unhandled errors)
        if results:
            try:
                save_results_to_file(results, output_path, file_write_lock)
            except Exception as save_error:
                print(f"\n⚠️  Final save attempt failed: {save_error}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-4 Vision or Gemini on How2Sign test set")
    parser.add_argument("--api-type", type=str, default="openai", choices=["openai", "gemini"],
                       help="API type to use: 'openai' or 'gemini' (default: openai)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (or set OPENAI_API_KEY/GEMINI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model to use. OpenAI: gpt-4o, gpt-4-vision-preview. Gemini: gemini-1.5-pro, gemini-1.5-flash (default: gpt-4o)")
    parser.add_argument("--video-folder", type=str, required=True,
                       help="Folder containing test videos")
    parser.add_argument("--question-file", type=str, required=True,
                       help="JSON/JSONL file with test questions")
    parser.add_argument("--out-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                       help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (0.0-2.0)")
    parser.add_argument("--video-fps", type=float, default=None,
                       help="Target FPS for frame extraction (if None, use num-frames)")
    parser.add_argument("--num-frames", type=int, default=6,
                       help="Number of frames to extract from video (default: 6)")
    parser.add_argument("--image-detail", type=str, default="low",
                       choices=["low", "high", "auto"],
                       help="Image detail level for GPT-4 Vision (default: low, 'high' costs more)")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save extracted frames to disk")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Custom prompt/question (overrides default)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of worker threads for Gemini API (default: 5, only used for Gemini)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.video_folder):
        print(f"❌ Video folder not found: {args.video_folder}")
        return
    if not os.path.exists(args.question_file):
        print(f"❌ Question file not found: {args.question_file}")
        return
    
    eval_model(args)


if __name__ == "__main__":
    main()

