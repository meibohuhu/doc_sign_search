#!/usr/bin/env python3
"""
Add prior prompt to randomly selected N entries in a JSONL file
"""

import json
import random
from pathlib import Path

def add_prior_prompt_to_jsonl(
    input_file: str,
    output_file: str,
    num_samples: int = 8000,
    prior_prompt: str = "Focus on the signer's hand shapes, movements, and facial expressions",
    random_seed: int = 42
):
    """
    Add prior prompt to randomly selected N entries in JSONL file, keep all other entries unchanged
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        num_samples: Number of samples to randomly select and add prior prompt to (default: 8000)
        prior_prompt: Prior prompt text to add before the main prompt
        random_seed: Random seed for reproducibility (default: 42)
    """
    print(f"Reading input file: {input_file}")
    
    # First pass: read all entries to get total count
    print("Step 1: Reading all entries...")
    all_entries = []
    valid_indices = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                # Check if entry has human conversation
                conversations = entry.get('conversations', [])
                has_human = any(conv.get('from') == 'human' for conv in conversations)
                
                if has_human:
                    all_entries.append(entry)
                    valid_indices.append(len(all_entries) - 1)
                else:
                    all_entries.append(entry)
                    print(f"⚠️  Warning: No human conversation found in line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON on line {line_num}: {e}")
                continue
    
    total_count = len(all_entries)
    valid_count = len(valid_indices)
    
    print(f"   Total entries: {total_count}")
    print(f"   Entries with human conversation: {valid_count}")
    
    # Randomly select indices to modify
    if num_samples > valid_count:
        print(f"⚠️  Warning: Requested {num_samples} samples but only {valid_count} valid entries available.")
        print(f"   Using all {valid_count} valid entries.")
        selected_indices = set(valid_indices)
    else:
        print(f"Step 2: Randomly selecting {num_samples} entries...")
        random.seed(random_seed)
        selected_indices = set(random.sample(valid_indices, num_samples))
        print(f"   Selected {len(selected_indices)} entries (seed={random_seed})")
    
    # Second pass: modify selected entries and write output
    print("Step 3: Processing and writing output...")
    modified_count = 0
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, entry in enumerate(all_entries):
            if idx in selected_indices:
                # Find and modify human prompt
                conversations = entry.get('conversations', [])
                
                for conv in conversations:
                    if conv.get('from') == 'human':
                        old_value = conv.get('value', '')
                        
                        # Extract the main prompt (everything after <video>\n)
                        if old_value.startswith('<video>\n'):
                            main_prompt = old_value[len('<video>\n'):].strip()
                            # Combine prior prompt and main prompt into one sentence
                            new_value = f"<video>\n{prior_prompt}. {main_prompt}"
                        else:
                            # If format is different, just prepend prior prompt
                            new_value = f"<video>\n{prior_prompt}. {old_value}"
                        
                        conv['value'] = new_value
                        modified_count += 1
                        break
            
            # Write entry (modified or unchanged)
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Processing complete:")
    print(f"   Total entries: {total_count}")
    print(f"   Entries with prior prompt: {modified_count}")
    print(f"   Entries unchanged: {total_count - modified_count}")
    print(f"   Output file: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add prior prompt to first N entries in JSONL file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=8000,
        help='Number of samples to process (default: 8000)'
    )
    parser.add_argument(
        '--prior-prompt',
        type=str,
        default="Focus on the signer's hand shapes, movements, and facial expressions",
        help='Prior prompt text to add before the main prompt'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    add_prior_prompt_to_jsonl(args.input, args.output, args.num_samples, args.prior_prompt, args.random_seed)

if __name__ == '__main__':
    main()

