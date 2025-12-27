#!/usr/bin/env python3
"""
Generate QA pairs from GPT-4V/GPT-5 evaluation results using GPT-5/GPT-4o
Takes statements from evaluation results and generates visual grounding QA pairs
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from openai import AzureOpenAI
except ImportError:
    print("OpenAI SDK not installed. Please install with: pip install openai")
    sys.exit(1)


# Visual grounding QA prompt template
visual_grounding_qa_prompt = """
You are generating question–answer (QA) pairs for training a vision-language model.

You are given one or more action description statements describing hand movements.

Your task is to generate QA pairs that test visual grounding:

1. Hand selection QA:
   - If both "left hand" and "right hand" are mentioned in the statements, generate two separate QA pairs, one for each hand.  
     - The answer for the left hand question must be "Left hand".  
     - The answer for the right hand question must be "Right hand".  
   - If only one hand is mentioned, generate one Hand selection QA for that hand.

2. Binary visual property QA:
   - Generate one Yes/No question about an observable visual feature based on the statements.

STRICT REQUIREMENTS:

1. Each question MUST include an explicit list of answer options.
2. The answer MUST be copied EXACTLY from the provided options.
3. Answers must be extremely short and belong to a closed set.
4. Do NOT include explanations or extra words in the answers.
5. The question text MUST NOT contain:
  - Any answer option (e.g., Left hand, Right hand, Both hands)
  - Any synonym or paraphrase of the answer options
6. At least one QA pair MUST be counterfactual (i.e., the correct answer should be "No").

ALLOWED OPTION SETS:

- Hand selection:
  Options: Left hand / Right hand / Both hands

- Binary visual property:
  Options: Yes / No

QUESTION DESIGN RULES:

- Questions must focus ONLY on observable physical properties (hand selection, motion, handshape, orientation).  
- Do NOT ask about meaning, intent, or interpretation.

OUTPUT FORMAT (follow exactly):

Q1: <question>
Options: <options>
A1: <answer>

Q2: <question>
Options: <options>
A2: <answer>

[Optional Q3 if both hands are present]:
Q3: <question>
Options: <options>
A3: <answer>

Here are the action description statements:
{statements}
"""


# QA pair evaluation prompt template
qa_evaluation_prompt = """
# Task
Evaluate the following QA pairs based **strictly** on the provided ASL action descriptions. 

# Action Descriptions
{statements}

# QA Pairs to Evaluate
{qa_pairs}

# Instructions
For each QA pair, provide:
1. **Judgment**: (Correct or Incorrect)
2. **Reason**: A brief explanation citing the specific part of the statement that supports or contradicts the QA.

OUTPUT FORMAT (follow exactly):

For Q1:
Judgment: <Correct or Incorrect>
Reason: <explanation>

For Q2:
Judgment: <Correct or Incorrect>
Reason: <explanation>

[Continue for all QA pairs...]
"""


def call_gpt_api(
    prompt,
    api_key,
    model="gpt-4o",
    max_tokens=1024,
    temperature=1,
    azure_endpoint=None,
    api_version=None,
    deployment=None
):
    """
    Call Azure OpenAI GPT API.
    
    Args:
        prompt: Text prompt
        api_key: Azure OpenAI API key
        model: Model name (default: gpt-4o)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: Azure API version
        deployment: Azure deployment name
    
    Returns:
        Response text from the model
    """
    if not api_key or not isinstance(api_key, str):
        raise Exception("Azure OpenAI API key is required and must be a string")
    
    # Get Azure configuration from parameters or environment variables
    endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT', 'https://dil-research-3.openai.azure.com/')
    api_ver = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    deployment_name = deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', model)
    
    # Initialize Azure OpenAI client
    try:
        client = AzureOpenAI(
            api_version=api_ver,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
    except Exception as e:
        raise Exception(f"Failed to initialize Azure OpenAI client: {str(e)}")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        # Call Azure OpenAI API
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract text response
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            raise Exception("No text response received from Azure OpenAI API")
    
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
            raise Exception(
                f"❌ Azure OpenAI API authentication failed: {error_msg}\n\n"
                f"Please verify:\n"
                f"1. ✅ Your Azure API key is correct and active\n"
                f"2. ✅ Your Azure endpoint is correct: {endpoint}\n"
                f"3. ✅ Your deployment name is correct: {deployment_name}\n"
                f"4. ✅ Your API version is correct: {api_ver}"
            )
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise Exception(f"Azure OpenAI API rate limit exceeded: {error_msg}")
        else:
            raise Exception(f"Azure OpenAI API call failed: {error_msg}")


def parse_qa_output(output_text):
    """
    Parse GPT output into QA pairs.
    
    Args:
        output_text: Raw text output from GPT
    
    Returns:
        list: List of QA pair dictionaries, each with 'question', 'options', 'answer'
    """
    qa_pairs = []
    
    if not output_text or output_text.startswith("ERROR"):
        return qa_pairs
    
    # Pattern to match Q/A pairs
    # Format: Q1: <question>\nOptions: <options>\nA1: <answer>
    # More flexible pattern that handles variations
    pattern = r'Q\d+:\s*(.+?)\s*Options:\s*(.+?)\s*A\d+:\s*(.+?)(?=\n\s*Q\d+:|$)'
    
    matches = re.finditer(pattern, output_text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    
    for match in matches:
        question = match.group(1).strip()
        options = match.group(2).strip()
        answer = match.group(3).strip()
        
        # Clean up question and answer - remove extra whitespace
        question = re.sub(r'\s+', ' ', question).strip()
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove any extra formatting from question
        question = question.rstrip('?.').strip()
        if not question.endswith('?'):
            question += '?'
        
        # Clean up answer - remove quotes if present
        answer = re.sub(r'^["\']|["\']$', '', answer).strip()
        
        qa_pairs.append({
            "question": question,
            "options": options,
            "answer": answer
        })
    
    return qa_pairs


def parse_evaluation_output(evaluation_text, num_qa_pairs):
    """
    Parse GPT evaluation output into judgments and reasons.
    
    Args:
        evaluation_text: Raw text output from GPT evaluation
        num_qa_pairs: Number of QA pairs to parse
    
    Returns:
        list: List of evaluation dictionaries, each with 'judgment' and 'reason'
    """
    evaluations = []
    
    if not evaluation_text or evaluation_text.startswith("ERROR"):
        # Return default evaluations if parsing fails
        return [{"judgment": "Unknown", "reason": "Evaluation parsing failed"} for _ in range(num_qa_pairs)]
    
    # Pattern to match evaluation for each QA pair
    # Format: For Q1:\nJudgment: Correct/Incorrect\nReason: ...
    # Also handle variations like "Q1:" or "Question 1:" etc.
    pattern = r'(?:For\s+)?(?:Q|Question)\s*\d+[:\-]?\s*Judgment:\s*(Correct|Incorrect)\s*Reason:\s*(.+?)(?=(?:For\s+)?(?:Q|Question)\s*\d+[:\-]?|$)'
    
    matches = re.finditer(pattern, evaluation_text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    
    for match in matches:
        judgment = match.group(1).strip()
        reason = match.group(2).strip()
        
        # Clean up reason - remove extra whitespace
        reason = re.sub(r'\s+', ' ', reason).strip()
        # Remove leading/trailing punctuation that might be part of formatting
        reason = reason.strip('.,;:')
        
        evaluations.append({
            "judgment": judgment,
            "reason": reason
        })
    
    # If we didn't get enough evaluations, pad with defaults
    while len(evaluations) < num_qa_pairs:
        evaluations.append({
            "judgment": "Unknown",
            "reason": "Evaluation not found in output"
        })
    
    # Return only the number we need
    return evaluations[:num_qa_pairs]


def evaluate_qa_pairs(statements_text, qa_pairs, args, api_key):
    """
    Evaluate QA pairs using GPT API.
    
    Args:
        statements_text: Combined statements text
        qa_pairs: List of QA pair dictionaries
        args: Command line arguments
        api_key: API key
    
    Returns:
        list: List of evaluation dictionaries
    """
    if not qa_pairs:
        return []
    
    # Format QA pairs for evaluation prompt
    qa_pairs_text = ""
    for idx, qa in enumerate(qa_pairs, 1):
        qa_pairs_text += f"Q{idx}: {qa['question']}\n"
        qa_pairs_text += f"Options: {qa['options']}\n"
        qa_pairs_text += f"Answer: {qa['answer']}\n\n"
    
    # Format evaluation prompt
    prompt = qa_evaluation_prompt.format(
        statements=statements_text.strip(),
        qa_pairs=qa_pairs_text.strip()
    )
    
    try:
        # Call GPT API for evaluation
        evaluation_output = call_gpt_api(
            prompt=prompt,
            api_key=api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            azure_endpoint=args.azure_endpoint,
            api_version=args.azure_api_version,
            deployment=args.azure_deployment
        )
        
        # Parse evaluation results
        evaluations = parse_evaluation_output(evaluation_output, len(qa_pairs))
        return evaluations
    
    except Exception as e:
        # Return default evaluations on error
        return [{"judgment": "Error", "reason": f"Evaluation failed: {str(e)}"} for _ in qa_pairs]


def generate_touch_qa_pair(statement3):
    """
    Generate touch/not touch QA pair directly from statement3.
    
    Args:
        statement3: String containing "touch" or "not touch"
    
    Returns:
        dict: QA pair dictionary
    """
    # Normalize statement3
    statement3_lower = statement3.lower().strip()
    
    # Determine answer
    if "touch" in statement3_lower and "not" not in statement3_lower:
        answer = "touch"
    elif "not touch" in statement3_lower or ("not" in statement3_lower and "touch" in statement3_lower):
        answer = "not touch"
    else:
        # Default to "not touch" if unclear
        answer = "not touch"
    
    return {
        "question": "Answer only \"touch\" or \"not touch\" - whether two hands visibly touch each other or not",
        "options": "touch / not touch",
        "answer": answer
    }


def process_single_video(args, result_item, idx, total, api_key):
    """
    Process a single video to generate QA pairs.
    Thread-safe function for parallel processing.
    
    Returns:
        dict: Result dictionary with video and qa_pairs
    """
    try:
        video_file = result_item.get("video", "unknown")
        statements = result_item.get("statements", {})
        
        statement1 = statements.get("statement1", "")
        statement2 = statements.get("statement2", "")
        statement3 = statements.get("statement3", "")
        
        # Skip if no statements available
        if not statement1 and not statement2:
            return {
                "video": video_file,
                "qa_pairs": [],
                "error": "No statements available"
            }
        
        qa_pairs = []
        statements_text = ""  # Initialize statements_text
        
        # Generate QA pairs from statement1 and statement2 using GPT
        if statement1 or statement2:
            # Combine statements
            if statement1:
                statements_text += f"Statement 1: {statement1}\n"
            if statement2:
                statements_text += f"Statement 2: {statement2}\n"
            
            # Format prompt
            prompt = visual_grounding_qa_prompt.format(statements=statements_text.strip())
            
            try:
                # Call GPT API
                gpt_output = call_gpt_api(
                    prompt=prompt,
                    api_key=api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    azure_endpoint=args.azure_endpoint,
                    api_version=args.azure_api_version,
                    deployment=args.azure_deployment
                )
                
                # Parse QA pairs from GPT output
                parsed_qa_pairs = parse_qa_output(gpt_output)
                qa_pairs.extend(parsed_qa_pairs)
                
            except Exception as e:
                error_msg = f"GPT API call failed: {str(e)}"
                print(f"\n⚠️  [{idx}/{total}] {video_file}: {error_msg}")
                # Continue to add touch QA pair even if GPT fails
                # qa_pairs remains [] if GPT fails, but we still add touch QA pair below
        
        # Add touch/not touch QA pair from statement3
        if statement3:
            touch_qa = generate_touch_qa_pair(statement3)
            qa_pairs.append(touch_qa)
        
        # Evaluate QA pairs if any were generated and evaluation is enabled
        if qa_pairs and not args.no_evaluate:
            try:
                # Prepare statements text for evaluation (include statement3 if available)
                eval_statements_text = statements_text if statement1 or statement2 else ""
                if statement3:
                    if eval_statements_text:
                        eval_statements_text += f"\nStatement 3: {statement3}"
                    else:
                        eval_statements_text = f"Statement 3: {statement3}"
                
                # Evaluate QA pairs
                evaluations = evaluate_qa_pairs(eval_statements_text, qa_pairs, args, api_key)
                
                # Add evaluations to each QA pair
                for qa_pair, evaluation in zip(qa_pairs, evaluations):
                    qa_pair["evaluation"] = evaluation
            
            except Exception as e:
                error_msg = f"QA evaluation failed: {str(e)}"
                print(f"\n⚠️  [{idx}/{total}] {video_file}: {error_msg}")
                # Add default evaluations if evaluation fails
                for qa_pair in qa_pairs:
                    qa_pair["evaluation"] = {
                        "judgment": "Error",
                        "reason": error_msg
                    }
        
        return {
            "video": video_file,
            "qa_pairs": qa_pairs
        }
    
    except Exception as e:
        return {
            "video": result_item.get("video", "unknown"),
            "qa_pairs": [],
            "error": str(e)
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from GPT-4V/GPT-5 evaluation results using GPT-5/GPT-4o"
    )
    parser.add_argument("--api-key", type=str, default=None,
                       help="Azure OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model to use: gpt-4o, gpt-5 (default: gpt-4o)")
    parser.add_argument("--azure-endpoint", type=str, default=None,
                       help="Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT env var)")
    parser.add_argument("--azure-api-version", type=str, default=None,
                       help="Azure OpenAI API version (or set AZURE_OPENAI_API_VERSION env var)")
    parser.add_argument("--azure-deployment", type=str, default=None,
                       help="Azure OpenAI deployment name (or set AZURE_OPENAI_DEPLOYMENT env var)")
    parser.add_argument("--input-file", type=str, required=True,
                       help="Input JSON file with GPT-4V/GPT-5 evaluation results")
    parser.add_argument("--output-file", type=str, required=None,
                       help="Output JSON file for QA pairs (default: auto-generated)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Max tokens to generate (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--max-workers", type=int, default=5,
                       help="Maximum number of worker threads (default: 5)")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save results every N samples (default: 10)")
    parser.add_argument("--no-evaluate", action="store_true",
                       help="Disable QA pair evaluation (evaluation is enabled by default)")
    
    args = parser.parse_args()
    
    # Check API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: Azure OpenAI API key not provided!")
        print("   Please provide --api-key or set OPENAI_API_KEY environment variable")
        return
    
    print(f"🔑 Using Azure OpenAI API key: {api_key[:10]}...")
    
    # Display Azure configuration
    endpoint = args.azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT', 'https://dil-research-3.openai.azure.com/')
    api_ver = args.azure_api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    deployment = args.azure_deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', args.model)
    print(f"   Azure Endpoint: {endpoint}")
    print(f"   Azure API Version: {api_ver}")
    print(f"   Azure Deployment: {deployment}")
    print(f"🤖 Model: {args.model}")
    if args.no_evaluate:
        print(f"⚠️  QA pair evaluation: DISABLED\n")
    else:
        print(f"✅ QA pair evaluation: ENABLED\n")
    
    # Load input file
    print(f"📂 Loading input file: {args.input_file}")
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    if not isinstance(input_data, list):
        input_data = [input_data]
    
    if args.max_samples:
        input_data = input_data[:args.max_samples]
        print(f"   Limited to {args.max_samples} samples")
    
    print(f"   Total samples: {len(input_data)}\n")
    
    if len(input_data) == 0:
        print("❌ ERROR: No data loaded!")
        return
    
    # Set output file path
    if args.output_file:
        output_path = args.output_file
    else:
        # Auto-generate output filename
        input_dir = os.path.dirname(args.input_file)
        input_basename = os.path.basename(args.input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"qa_pairs_{timestamp}.json"
        output_path = os.path.join(input_dir, output_filename)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"💾 Output file: {output_path}\n")
    
    # Initialize results
    results = []
    
    # Lock for thread-safe file writing
    file_write_lock = threading.Lock()
    
    print(f"🎬 Starting QA pair generation...")
    print(f"   Saving every {args.save_interval} samples\n")
    print(f"{'='*70}\n")
    
    # Process samples using multithreading
    max_workers = min(args.max_workers, len(input_data))
    print(f"🚀 Using multithreading with {max_workers} workers\n")
    
    # Prepare sample data with indices
    sample_data = [(args, item, idx + 1, len(input_data), api_key) 
                  for idx, item in enumerate(input_data)]
    
    # Process samples in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_video, *sample_args): sample_args[2] 
            for sample_args in sample_data
        }
        
        # Collect results as they complete and save periodically
        sample_results = {}
        completed_count = 0
        
        for future in tqdm(as_completed(future_to_idx), total=len(input_data), desc="Generating QA pairs"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                sample_results[idx] = result
                completed_count += 1
                
                # Periodic save
                if completed_count % args.save_interval == 0 or completed_count == len(input_data):
                    # Build current results in order (thread-safe)
                    with file_write_lock:
                        sorted_indices = sorted([i for i in sample_results.keys()])
                        current_results = [sample_results[i] for i in sorted_indices]
                        
                        # Update global results
                        results.clear()
                        results.extend(current_results)
                        
                        # Save to file
                        save_results_to_file(results, output_path, None)  # Lock already acquired
                    
                    if completed_count % args.save_interval == 0:
                        print(f"\n💾 Progress saved: {completed_count}/{len(input_data)} samples")
            
            except Exception as e:
                print(f"\n⚠️  [{idx}/{len(input_data)}] Thread error: {e}")
                sample_results[idx] = {
                    "video": "unknown",
                    "qa_pairs": [],
                    "error": f"Thread error: {str(e)}"
                }
    
    # Final processing: ensure all results are in the correct order
    sorted_results = [sample_results[i] for i in sorted(sample_results.keys())]
    
    # Update final results
    results.clear()
    results.extend(sorted_results)
    
    # Final save
    save_results_to_file(results, output_path, file_write_lock)
    
    print(f"\n{'='*70}")
    print(f"✅ QA pairs generated successfully!")
    print(f"   Output file: {output_path}")
    print(f"   Total samples: {len(results)}")
    
    # Calculate statistics
    total_qa_pairs = sum(len(r.get("qa_pairs", [])) for r in results)
    avg_qa_pairs = total_qa_pairs / len(results) if results else 0
    successful = len([r for r in results if not r.get("error")])
    
    print(f"   Successful: {successful}/{len(results)}")
    print(f"   Total QA pairs: {total_qa_pairs}")
    print(f"   Average QA pairs per video: {avg_qa_pairs:.1f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

