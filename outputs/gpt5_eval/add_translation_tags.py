#!/usr/bin/env python3
"""
Extract all translation entries from segmented_train_val_combined.json
and add [TASK: Translation] tag if not present.
"""

import json
import sys
import argparse
from pathlib import Path


def add_translation_tags(input_file: str, output_file: str):
    """
    Extract translation entries and ensure they have [TASK: Translation] tag.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Read input file
    print(f"📖 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} total entries")
    
    # Extract translation entries
    translation_entries = []
    target_question = "Translate the American Sign Language in this video to English."
    target_question_with_tag = "<video>\n[TASK: Translation]\nTranslate the American Sign Language in this video to English."
    target_question_without_tag = "<video>\nTranslate the American Sign Language in this video to English."
    
    for entry in data:
        conversations = entry.get("conversations", [])
        if len(conversations) >= 2:
            human_value = conversations[0].get("value", "")
            
            # Check if this is a translation entry
            if target_question in human_value:
                # Create a copy
                new_entry = {
                    "id": entry.get("id", ""),
                    "video": entry.get("video", ""),
                    "conversations": []
                }
                
                # Add tag if not present
                if "[TASK: Translation]" not in human_value:
                    # Replace old format with new format
                    new_human_value = human_value.replace(
                        target_question_without_tag,
                        target_question_with_tag
                    )
                    # If replacement didn't work, try adding the tag
                    if new_human_value == human_value:
                        if human_value.startswith("<video>\n"):
                            new_human_value = human_value.replace(
                                "<video>\n",
                                "<video>\n[TASK: Translation]\n"
                            )
                        else:
                            new_human_value = f"<video>\n[TASK: Translation]\n{target_question}"
                else:
                    new_human_value = human_value
                
                # Add human message
                new_entry["conversations"].append({
                    "from": "human",
                    "value": new_human_value
                })
                
                # Add GPT response
                if len(conversations) >= 2:
                    new_entry["conversations"].append(conversations[1])
                
                translation_entries.append(new_entry)
    
    # Write output file
    print(f"\n💾 Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translation_entries, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total translation entries extracted: {len(translation_entries)}")
    print(f"\n✅ File saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract all translation entries and add [TASK: Translation] tag"
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
    
    add_translation_tags(args.input_file, args.output)


if __name__ == "__main__":
    main()

