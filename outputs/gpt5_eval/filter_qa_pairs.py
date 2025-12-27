#!/usr/bin/env python3
"""
Filter QA pairs from JSON files based on criteria:
1. Remove QA pairs with judgment="Incorrect"
2. Remove QA pairs with "first" or "second" in the question (case-insensitive)
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re


def should_keep_qa_pair(qa_pair: Dict[str, Any]) -> bool:
    """
    Determine if a QA pair should be kept based on filtering criteria.
    
    Args:
        qa_pair: A dictionary containing question, options, answer, and evaluation
    
    Returns:
        True if the QA pair should be kept, False otherwise
    """
    # Filter 1: Remove if judgment is "Incorrect"
    evaluation = qa_pair.get("evaluation", {})
    judgment = evaluation.get("judgment", "")
    if judgment == "Incorrect":
        return False
    
    # Filter 2: Remove if question contains "first" or "second" (case-insensitive)
    question = qa_pair.get("question", "").lower()
    if "first" in question or "second" in question:
        return False
    
    return True


def filter_qa_pairs_file(input_file: str, output_file: str = None, dry_run: bool = False) -> Dict[str, int]:
    """
    Filter QA pairs in a JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input file)
        dry_run: If True, only report statistics without modifying files
    
    Returns:
        Dictionary with statistics: {"total_videos", "total_qa_before", "total_qa_after", "removed"}
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        "total_videos": len(data),
        "total_qa_before": 0,
        "total_qa_after": 0,
        "removed_incorrect": 0,
        "removed_first_second": 0,
        "removed_total": 0
    }
    
    # Filter QA pairs for each video
    filtered_data = []
    for video_entry in data:
        video_name = video_entry.get("video", "unknown")
        qa_pairs = video_entry.get("qa_pairs", [])
        
        stats["total_qa_before"] += len(qa_pairs)
        
        # Filter QA pairs
        filtered_qa_pairs = []
        for qa_pair in qa_pairs:
            # Track why it was removed
            evaluation = qa_pair.get("evaluation", {})
            judgment = evaluation.get("judgment", "")
            question = qa_pair.get("question", "").lower()
            
            if judgment == "Incorrect":
                stats["removed_incorrect"] += 1
                stats["removed_total"] += 1
                continue
            
            if "first" in question or "second" in question:
                stats["removed_first_second"] += 1
                stats["removed_total"] += 1
                continue
            
            # Keep this QA pair
            filtered_qa_pairs.append(qa_pair)
        
        stats["total_qa_after"] += len(filtered_qa_pairs)
        
        # Keep video entry if it has at least one QA pair
        if filtered_qa_pairs:
            filtered_entry = {
                "video": video_name,
                "qa_pairs": filtered_qa_pairs
            }
            filtered_data.append(filtered_entry)
    
    # Write output file
    if not dry_run:
        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    return stats


def filter_directory(directory: str, pattern: str = "*.json", dry_run: bool = False):
    """
    Filter all JSON files in a directory.
    
    Args:
        directory: Directory containing JSON files
        pattern: File pattern to match (default: "*.json")
        dry_run: If True, only report statistics without modifying files
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"❌ Error: {directory} is not a valid directory")
        return
    
    json_files = list(dir_path.glob(pattern))
    if not json_files:
        print(f"⚠️  Warning: No JSON files found in {directory} matching pattern {pattern}")
        return
    
    print(f"📁 Found {len(json_files)} JSON file(s) in {directory}")
    if dry_run:
        print("🔍 DRY RUN MODE: Files will not be modified")
    print()
    
    total_stats = {
        "total_files": len(json_files),
        "total_videos": 0,
        "total_qa_before": 0,
        "total_qa_after": 0,
        "removed_incorrect": 0,
        "removed_first_second": 0,
        "removed_total": 0
    }
    
    for json_file in sorted(json_files):
        print(f"Processing: {json_file.name}")
        try:
            stats = filter_qa_pairs_file(str(json_file), dry_run=dry_run)
            
            # Accumulate statistics
            total_stats["total_videos"] += stats["total_videos"]
            total_stats["total_qa_before"] += stats["total_qa_before"]
            total_stats["total_qa_after"] += stats["total_qa_after"]
            total_stats["removed_incorrect"] += stats["removed_incorrect"]
            total_stats["removed_first_second"] += stats["removed_first_second"]
            total_stats["removed_total"] += stats["removed_total"]
            
            print(f"  Videos: {stats['total_videos']}")
            print(f"  QA pairs: {stats['total_qa_before']} → {stats['total_qa_after']} "
                  f"(removed {stats['removed_total']}: {stats['removed_incorrect']} Incorrect, "
                  f"{stats['removed_first_second']} first/second)")
            print()
        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {e}")
            print()
    
    # Print summary
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_stats['total_files']}")
    print(f"Total videos: {total_stats['total_videos']}")
    print(f"Total QA pairs: {total_stats['total_qa_before']} → {total_stats['total_qa_after']}")
    print(f"Removed: {total_stats['removed_total']}")
    print(f"  - Incorrect judgment: {total_stats['removed_incorrect']}")
    print(f"  - Contains 'first'/'second': {total_stats['removed_first_second']}")
    if total_stats['total_qa_before'] > 0:
        removal_rate = 100 * total_stats['removed_total'] / total_stats['total_qa_before']
        print(f"Removal rate: {removal_rate:.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Filter QA pairs from JSON files based on judgment and question content"
    )
    parser.add_argument("input", type=str,
                       help="Input JSON file or directory containing JSON files")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file (only used when input is a single file)")
    parser.add_argument("--pattern", "-p", type=str, default="*.json",
                       help="File pattern for directory processing (default: *.json)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode: report statistics without modifying files")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        print(f"📄 Processing single file: {args.input}")
        if args.dry_run:
            print("🔍 DRY RUN MODE: File will not be modified")
        print()
        
        stats = filter_qa_pairs_file(args.input, args.output, dry_run=args.dry_run)
        
        print("📊 Statistics:")
        print(f"  Videos: {stats['total_videos']}")
        print(f"  QA pairs: {stats['total_qa_before']} → {stats['total_qa_after']}")
        print(f"  Removed: {stats['removed_total']}")
        print(f"    - Incorrect judgment: {stats['removed_incorrect']}")
        print(f"    - Contains 'first'/'second': {stats['removed_first_second']}")
        if stats['total_qa_before'] > 0:
            removal_rate = 100 * stats['removed_total'] / stats['total_qa_before']
            print(f"  Removal rate: {removal_rate:.2f}%")
        
        if not args.dry_run:
            output_path = args.output if args.output else args.input
            print(f"\n✅ Filtered file saved to: {output_path}")
    
    elif input_path.is_dir():
        # Directory processing
        filter_directory(str(input_path), args.pattern, dry_run=args.dry_run)
    
    else:
        print(f"❌ Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

