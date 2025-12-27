#!/usr/bin/env python3
"""
Add [TASK: Translation] tag to ALL entries in segmented_train_val_combined.json
regardless of the question format.
"""

import json
import sys
import argparse
from pathlib import Path


def add_tags_to_all_entries(input_file: str, output_file: str):
    """
    Add [TASK: Translation] tag to all entries.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Read input file
    print(f"📖 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} total entries")
    
    # Process all entries
    modified_count = 0
    
    for entry in data:
        conversations = entry.get("conversations", [])
        if len(conversations) >= 1:
            human_value = conversations[0].get("value", "")
            
            # Add tag if not present
            if "[TASK: Translation]" not in human_value:
                # Add the tag after <video>\n
                if human_value.startswith("<video>\n"):
                    new_human_value = human_value.replace(
                        "<video>\n",
                        "<video>\n[TASK: Translation]\n",
                        1  # Only replace first occurrence
                    )
                    conversations[0]["value"] = new_human_value
                    modified_count += 1
                else:
                    # If doesn't start with <video>, add both
                    new_human_value = f"<video>\n[TASK: Translation]\n{human_value}"
                    conversations[0]["value"] = new_human_value
                    modified_count += 1
    
    # Write output file
    print(f"\n💾 Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total entries: {len(data)}")
    print(f"Entries modified: {modified_count}")
    print(f"Entries already with tag: {len(data) - modified_count}")
    print(f"\n✅ File saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Add [TASK: Translation] tag to ALL entries"
    )
    parser.add_argument("input_file", type=str,
                       help="Input JSON file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"❌ Error: {args.input_file} is not a valid file")
        sys.exit(1)
    
    add_tags_to_all_entries(args.input_file, args.output)


if __name__ == "__main__":
    main()

