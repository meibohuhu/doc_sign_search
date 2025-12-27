#!/usr/bin/env python3
"""
Merge multiple QA pairs JSON files into a single JSON file.
If the same video appears in multiple files, merge their QA pairs.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def merge_qa_json_files(input_dir: str, output_file: str, deduplicate_qa: bool = True):
    """
    Merge all JSON files in a directory into a single JSON file.
    
    Args:
        input_dir: Directory containing JSON files to merge
        output_file: Path to output merged JSON file
        deduplicate_qa: If True, remove duplicate QA pairs for the same video
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"❌ Error: {input_dir} is not a valid directory")
        return
    
    # Find all JSON files
    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"⚠️  Warning: No JSON files found in {input_dir}")
        return
    
    print(f"📁 Found {len(json_files)} JSON file(s) in {input_dir}")
    print()
    
    # Dictionary to store video -> qa_pairs mapping
    video_data = defaultdict(lambda: {"qa_pairs": []})
    total_videos = 0
    total_qa_pairs = 0
    duplicate_videos = set()
    
    # Process each JSON file
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_videos = len(data)
            file_qa_pairs = sum(len(entry.get("qa_pairs", [])) for entry in data)
            total_videos += file_videos
            total_qa_pairs += file_qa_pairs
            
            print(f"  Videos: {file_videos}, QA pairs: {file_qa_pairs}")
            
            # Process each video entry
            for entry in data:
                video_name = entry.get("video", "unknown")
                qa_pairs = entry.get("qa_pairs", [])
                
                if video_name in video_data:
                    duplicate_videos.add(video_name)
                    # Merge QA pairs
                    if deduplicate_qa:
                        # Create a set of QA pair signatures to avoid duplicates
                        existing_signatures = set()
                        for qa in video_data[video_name]["qa_pairs"]:
                            sig = (qa.get("question", ""), qa.get("answer", ""))
                            existing_signatures.add(sig)
                        
                        # Add new QA pairs that don't already exist
                        for qa in qa_pairs:
                            sig = (qa.get("question", ""), qa.get("answer", ""))
                            if sig not in existing_signatures:
                                video_data[video_name]["qa_pairs"].append(qa)
                                existing_signatures.add(sig)
                    else:
                        # Simply append all QA pairs
                        video_data[video_name]["qa_pairs"].extend(qa_pairs)
                else:
                    # First occurrence of this video
                    video_data[video_name] = {
                        "video": video_name,
                        "qa_pairs": qa_pairs.copy()
                    }
            
            print()
        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {e}")
            print()
    
    # Convert to list format
    merged_data = []
    merged_qa_pairs = 0
    for video_name in sorted(video_data.keys()):
        entry = {
            "video": video_name,
            "qa_pairs": video_data[video_name]["qa_pairs"]
        }
        merged_data.append(entry)
        merged_qa_pairs += len(entry["qa_pairs"])
    
    # Write merged output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("=" * 60)
    print("📊 MERGE SUMMARY")
    print("=" * 60)
    print(f"Total JSON files processed: {len(json_files)}")
    print(f"Total videos in input files: {total_videos}")
    print(f"Total QA pairs in input files: {total_qa_pairs}")
    print(f"Unique videos in merged file: {len(merged_data)}")
    print(f"Total QA pairs in merged file: {merged_qa_pairs}")
    if duplicate_videos:
        print(f"Videos appearing in multiple files: {len(duplicate_videos)}")
        print(f"  (QA pairs from duplicate videos were merged)")
    print(f"\n✅ Merged file saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple QA pairs JSON files into a single file"
    )
    parser.add_argument("input_dir", type=str,
                       help="Directory containing JSON files to merge")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--no-deduplicate", action="store_true",
                       help="Don't remove duplicate QA pairs when merging videos")
    
    args = parser.parse_args()
    
    deduplicate = not args.no_deduplicate
    
    merge_qa_json_files(args.input_dir, args.output, deduplicate_qa=deduplicate)


if __name__ == "__main__":
    main()

