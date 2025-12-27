#!/usr/bin/env python3
"""
Remove task tags ([TASK: Translation], [TASK: Visual Grounding]) from merged JSON files.
"""

import json
import sys
import argparse
from pathlib import Path
import re


def remove_task_tags(input_file: str, output_file: str):
    """
    Remove task tags from the JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Read input file
    print(f"📖 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} entries")
    
    # Task tags to remove
    task_tags = [
        "[TASK: Translation]",
        "[TASK: Visual Grounding]"
    ]
    
    # Process entries
    modified_count = 0
    for entry in data:
        conversations = entry.get("conversations", [])
        if len(conversations) >= 1:
            human_value = conversations[0].get("value", "")
            original_value = human_value
            
            # Remove task tags
            for tag in task_tags:
                # Remove the tag and the newline after it
                human_value = human_value.replace(f"\n{tag}\n", "\n")
                # Also handle case where tag is at the start
                human_value = human_value.replace(f"{tag}\n", "")
                # Handle case with trailing newline before tag
                human_value = re.sub(r'\n\s*' + re.escape(tag) + r'\s*\n', '\n', human_value)
            
            if human_value != original_value:
                conversations[0]["value"] = human_value
                modified_count += 1
    
    # Write output file
    print(f"\n💾 Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MODIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total entries: {len(data)}")
    print(f"Entries modified: {modified_count}")
    print(f"\n✅ Modified file saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Remove task tags from merged JSON files"
    )
    parser.add_argument("input_file", type=str,
                       help="Input JSON file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file path (default: adds '_no_tags' suffix)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"❌ Error: {args.input_file} is not a valid file")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        output_file = str(input_path.parent / f"{input_path.stem}_no_tags{input_path.suffix}")
    else:
        output_file = args.output
    
    remove_task_tags(args.input_file, output_file)


if __name__ == "__main__":
    main()

