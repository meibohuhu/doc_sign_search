#!/usr/bin/env python3
"""
Fix touch question format in instructional JSON files.
Replace: "Answer only \"touch\" or \"not touch\" - whether two hands visibly touch each other or not"
With: "Do the two hands visibly touch each other in the video?"
"""

import json
import sys
import argparse
from pathlib import Path


def fix_touch_questions(input_file: str, output_file: str = None):
    """
    Fix touch question format in the JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input file)
    """
    # Read input file
    print(f"📖 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} entries")
    
    # Old pattern
    old_pattern = "Answer only \"touch\" or \"not touch\" - whether two hands visibly touch each other or not"
    new_question = "Do the two hands visibly touch each other in the video?"
    
    # Fix entries
    fixed_count = 0
    for entry in data:
        conversations = entry.get("conversations", [])
        if len(conversations) >= 2:
            human_value = conversations[0].get("value", "")
            
            # Check if it contains the old pattern
            if old_pattern in human_value:
                # Replace the old question with the new one
                # The format is: "<video>\n[TASK: Visual Grounding]\n{old_question}\nAnswer with one of: touch / not touch."
                # We want to replace {old_question} with {new_question}
                new_value = human_value.replace(old_pattern, new_question)
                conversations[0]["value"] = new_value
                fixed_count += 1
    
    # Write output file
    output_path = output_file if output_file else input_file
    print(f"\n💾 Writing output file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FIX SUMMARY")
    print("=" * 60)
    print(f"Total entries: {len(data)}")
    print(f"Fixed entries: {fixed_count}")
    print(f"\n✅ Fixed file saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fix touch question format in instructional JSON files"
    )
    parser.add_argument("input_file", type=str,
                       help="Input JSON file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file path (if not specified, overwrites input file)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"❌ Error: {args.input_file} is not a valid file")
        sys.exit(1)
    
    fix_touch_questions(args.input_file, args.output)


if __name__ == "__main__":
    main()

