#!/usr/bin/env python3
"""
Convert QA pairs JSON format to instructional data format for training.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any


def extract_video_basename(video_path: str) -> str:
    """
    Extract basename from video path.
    e.g., "abc_123.mp4" -> "abc_123"
    """
    return os.path.splitext(video_path)[0]


def format_question(question: str, options: str = None) -> str:
    """
    Format question with task prefix and options.
    
    Args:
        question: The question text
        options: Optional options string (e.g., "Left hand / Right hand / Both hands")
    
    Returns:
        Formatted question string
    """
    formatted = f"<video>\n[TASK: Visual Grounding]\n{question}"
    
    if options:
        formatted += f"\nAnswer with one of: {options}."
    
    return formatted


def convert_qa_pairs_to_instructional(input_file: str, output_file: str):
    """
    Convert QA pairs format to instructional data format.
    
    Args:
        input_file: Path to input QA pairs JSON file
        output_file: Path to output instructional JSON file
    """
    # Read input file
    print(f"📖 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} videos")
    
    # Convert format
    instructional_data = []
    total_qa_pairs = 0
    
    for video_entry in data:
        video_name = video_entry.get("video", "")
        if not video_name:
            continue
        
        # Extract basename (remove .mp4 extension)
        video_basename = os.path.splitext(video_name)[0]
        qa_pairs = video_entry.get("qa_pairs", [])
        
        # Create one entry per QA pair
        for idx, qa_pair in enumerate(qa_pairs, start=1):
            question = qa_pair.get("question", "")
            options = qa_pair.get("options", "")
            answer = qa_pair.get("answer", "")
            
            # Skip if essential fields are missing
            if not question or not answer:
                continue
            
            # Create unique ID
            entry_id = f"{video_basename}_q{idx}"
            
            # Format question
            formatted_question = format_question(question, options if options else None)
            
            # Create instructional entry
            entry = {
                "id": entry_id,
                "video": video_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": formatted_question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            
            instructional_data.append(entry)
            total_qa_pairs += 1
    
    # Write output file
    print(f"\n💾 Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instructional_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Input videos: {len(data)}")
    print(f"Output entries: {len(instructional_data)}")
    print(f"Total QA pairs converted: {total_qa_pairs}")
    print(f"\n✅ Converted file saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert QA pairs JSON format to instructional data format"
    )
    parser.add_argument("input_file", type=str,
                       help="Input QA pairs JSON file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output instructional JSON file path")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"❌ Error: {args.input_file} is not a valid file")
        sys.exit(1)
    
    convert_qa_pairs_to_instructional(args.input_file, args.output)


if __name__ == "__main__":
    main()

