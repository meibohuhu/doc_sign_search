#!/usr/bin/env python3
"""
Extract a balanced sample of QA pairs from multiple instructional JSON files.
Proportions:
- Hand selection answers (Left hand/Left/Right hand/Right/Both hands/Both): 400
- Yes/No answers: 200
- Touch/not touch answers: 200
Total: 800 entries
"""

import json
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict


def categorize_entry(entry):
    """
    Categorize an entry based on its answer type.
    
    Returns:
        'hand_selection', 'yes_no', 'touch', or None
    """
    conversations = entry.get("conversations", [])
    if len(conversations) < 2:
        return None
    
    answer = conversations[1].get("value", "").strip()
    
    # Hand selection answers
    hand_answers = ["Left hand", "Right hand", "Both hands", "Left", "Right", "Both"]
    if answer in hand_answers:
        return "hand_selection"
    
    # Yes/No answers
    if answer in ["Yes", "No"]:
        return "yes_no"
    
    # Touch answers
    if answer in ["touch", "not touch"]:
        return "touch"
    
    return None


def extract_balanced_sample(input_files: list, output_file: str, 
                           hand_selection_count: int = 400,
                           yes_no_count: int = 200,
                           touch_count: int = 200,
                           seed: int = 42):
    """
    Extract a balanced sample from multiple JSON files.
    
    Args:
        input_files: List of input JSON file paths
        output_file: Path to output JSON file
        hand_selection_count: Number of hand selection entries to extract
        yes_no_count: Number of Yes/No entries to extract
        touch_count: Number of touch entries to extract
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load all entries from all files
    all_entries = []
    for input_file in input_files:
        print(f"📖 Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   Found {len(data)} entries")
        all_entries.extend(data)
    
    print(f"\n📊 Total entries loaded: {len(all_entries)}")
    
    # Categorize all entries
    categorized = defaultdict(list)
    uncategorized = []
    
    for entry in all_entries:
        category = categorize_entry(entry)
        if category:
            categorized[category].append(entry)
        else:
            uncategorized.append(entry)
    
    print(f"\n📈 Categorization:")
    print(f"   Hand selection: {len(categorized['hand_selection'])}")
    print(f"   Yes/No: {len(categorized['yes_no'])}")
    print(f"   Touch: {len(categorized['touch'])}")
    print(f"   Uncategorized: {len(uncategorized)}")
    
    # Sample from each category
    sampled_entries = []
    
    # Sample hand selection
    hand_available = len(categorized['hand_selection'])
    if hand_available >= hand_selection_count:
        sampled_hand = random.sample(categorized['hand_selection'], hand_selection_count)
    else:
        sampled_hand = categorized['hand_selection']
        print(f"   ⚠️  Warning: Only {hand_available} hand selection entries available, requested {hand_selection_count}")
    sampled_entries.extend(sampled_hand)
    print(f"   ✓ Sampled {len(sampled_hand)} hand selection entries")
    
    # Sample Yes/No
    yes_no_available = len(categorized['yes_no'])
    if yes_no_available >= yes_no_count:
        sampled_yes_no = random.sample(categorized['yes_no'], yes_no_count)
    else:
        sampled_yes_no = categorized['yes_no']
        print(f"   ⚠️  Warning: Only {yes_no_available} Yes/No entries available, requested {yes_no_count}")
    sampled_entries.extend(sampled_yes_no)
    print(f"   ✓ Sampled {len(sampled_yes_no)} Yes/No entries")
    
    # Sample touch
    touch_available = len(categorized['touch'])
    if touch_available >= touch_count:
        sampled_touch = random.sample(categorized['touch'], touch_count)
    else:
        sampled_touch = categorized['touch']
        print(f"   ⚠️  Warning: Only {touch_available} touch entries available, requested {touch_count}")
    sampled_entries.extend(sampled_touch)
    print(f"   ✓ Sampled {len(sampled_touch)} touch entries")
    
    # Shuffle final sample
    random.shuffle(sampled_entries)
    
    # Write output file
    print(f"\n💾 Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_entries, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total entries sampled: {len(sampled_entries)}")
    print(f"  - Hand selection: {len(sampled_hand)}")
    print(f"  - Yes/No: {len(sampled_yes_no)}")
    print(f"  - Touch: {len(sampled_touch)}")
    print(f"\n✅ Sample file saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract a balanced sample of QA pairs from multiple JSON files"
    )
    parser.add_argument("input_files", nargs='+', type=str,
                       help="Input JSON files (one or more)")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--hand-selection", type=int, default=400,
                       help="Number of hand selection entries (default: 400)")
    parser.add_argument("--yes-no", type=int, default=200,
                       help="Number of Yes/No entries (default: 200)")
    parser.add_argument("--touch", type=int, default=200,
                       help="Number of touch entries (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate input files
    for input_file in args.input_files:
        if not Path(input_file).is_file():
            print(f"❌ Error: {input_file} is not a valid file")
            sys.exit(1)
    
    extract_balanced_sample(
        args.input_files,
        args.output,
        args.hand_selection,
        args.yes_no,
        args.touch,
        args.seed
    )


if __name__ == "__main__":
    main()

