#!/usr/bin/env python3
"""
Simplify hand selection answers from "Left hand" / "Right hand" / "Both hands" 
to "Left" / "Right" / "Both" for a portion of the data.
This helps the model focus on distinguishing left/right/both rather than just learning "hand".
"""

import json
import sys
import argparse
import random
from pathlib import Path


def simplify_hand_answers(input_file: str, output_file: str = None, ratio: float = 0.5, seed: int = 42):
    """
    Simplify hand selection answers for a portion of the data.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input file)
        ratio: Ratio of entries to modify (0.0 to 1.0, default: 0.5)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read input file
    print(f"📖 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} entries")
    
    # Mapping for answer simplification
    answer_mapping = {
        "Left hand": "Left",
        "Right hand": "Right",
        "Both hands": "Both"
    }
    
    # Mapping for options simplification
    options_mapping = {
        "Left hand / Right hand / Both hands": "Left / Right / Both"
    }
    
    # Find entries that need modification
    candidate_indices = []
    for idx, entry in enumerate(data):
        conversations = entry.get("conversations", [])
        if len(conversations) >= 2:
            gpt_answer = conversations[1].get("value", "").strip()
            human_value = conversations[0].get("value", "")
            
            # Check if answer is one of the hand selection answers
            if gpt_answer in answer_mapping:
                # Check if options mention hand selection
                if "Left hand / Right hand / Both hands" in human_value:
                    candidate_indices.append(idx)
    
    print(f"   Found {len(candidate_indices)} hand selection entries")
    
    # Select subset to modify
    num_to_modify = int(len(candidate_indices) * ratio)
    indices_to_modify = random.sample(candidate_indices, num_to_modify)
    indices_to_modify_set = set(indices_to_modify)
    
    print(f"   Will modify {num_to_modify} entries ({ratio*100:.1f}%)")
    
    # Modify entries
    modified_count = 0
    for idx in candidate_indices:
        if idx in indices_to_modify_set:
            entry = data[idx]
            conversations = entry.get("conversations", [])
            
            # Update answer
            old_answer = conversations[1].get("value", "").strip()
            if old_answer in answer_mapping:
                conversations[1]["value"] = answer_mapping[old_answer]
            
            # Update options in human value
            human_value = conversations[0].get("value", "")
            for old_options, new_options in options_mapping.items():
                if old_options in human_value:
                    conversations[0]["value"] = human_value.replace(old_options, new_options)
            
            modified_count += 1
    
    # Write output file
    output_path = output_file if output_file else input_file
    print(f"\n💾 Writing output file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MODIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total entries: {len(data)}")
    print(f"Hand selection entries found: {len(candidate_indices)}")
    print(f"Entries modified: {modified_count}")
    print(f"Modification ratio: {ratio*100:.1f}%")
    print(f"\n✅ Modified file saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Simplify hand selection answers for a portion of the data"
    )
    parser.add_argument("input_file", type=str,
                       help="Input JSON file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file path (if not specified, overwrites input file)")
    parser.add_argument("--ratio", "-r", type=float, default=0.5,
                       help="Ratio of entries to modify (0.0 to 1.0, default: 0.5)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    if args.ratio < 0 or args.ratio > 1:
        print("❌ Error: ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"❌ Error: {args.input_file} is not a valid file")
        sys.exit(1)
    
    simplify_hand_answers(args.input_file, args.output, args.ratio, args.seed)


if __name__ == "__main__":
    main()

