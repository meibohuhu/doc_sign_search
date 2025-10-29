#!/usr/bin/env python3
"""
Filter videos from JSON based on duration from CSV file.
Extracts videos with duration < 20 seconds using how2sign_realigned_train.csv
"""

import json
import csv
import sys
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Filter JSON dataset by video duration from CSV file'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/how2sign_realigned_train.csv',
        help='Path to CSV file with video timing information'
    )
    parser.add_argument(
        '--json_input',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_corrupted_removed.json',
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--json_output',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_lt20s.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=20.0,
        help='Maximum duration in seconds (default: 20.0)'
    )
    return parser.parse_args()

def load_duration_from_csv(csv_file):
    """
    Load video durations from CSV file.
    Returns dict mapping SENTENCE_NAME -> duration
    """
    duration_map = {}
    
    print(f"📖 Reading CSV file: {csv_file}")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        count = 0
        for row in reader:
            sentence_name = row.get('SENTENCE_NAME', '').strip()
            start = float(row.get('START_REALIGNED', 0))
            end = float(row.get('END_REALIGNED', 0))
            
            if sentence_name and end > start:
                duration = end - start
                duration_map[sentence_name] = duration
                count += 1
        
        print(f"✅ Loaded {count} video durations from CSV")
    
    return duration_map

def filter_json_by_duration(json_input, json_output, duration_map, max_duration=20.0):
    """
    Filter JSON entries by duration from CSV.
    Only keeps entries that:
    1. Exist in JSON (already satisfied since we read from JSON)
    2. Have duration info in CSV
    3. Have duration < max_duration
    """
    print(f"📖 Reading JSON file: {json_input}")
    with open(json_input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Total entries in JSON: {len(data)}")
    
    # Create set of valid video IDs from CSV for faster lookup
    valid_csv_ids = set(duration_map.keys())
    print(f"📊 Total entries in CSV: {len(valid_csv_ids)}")
    
    filtered = []
    matched = 0
    missing_duration = 0
    too_long = 0
    missing_id = 0
    
    for item in data:
        video_id = item.get('id', '')
        
        # Skip entries without ID
        if not video_id:
            missing_id += 1
            continue
        
        # Verify entry exists in CSV (must be in both JSON and CSV)
        if video_id not in valid_csv_ids:
            missing_duration += 1
            continue
        
        # Get duration from CSV
        duration = duration_map.get(video_id)
        
        if duration is None:
            missing_duration += 1
            continue
        
        matched += 1
        
        # Filter by duration: must be < max_duration
        if duration < max_duration:
            filtered.append(item)
        else:
            too_long += 1
    
    print(f"\n📊 Filtering Statistics:")
    print(f"  - Total in JSON: {len(data)}")
    print(f"  - Entries without ID: {missing_id}")
    print(f"  - Not found in CSV (filtered out): {missing_duration}")
    print(f"  - Found in both JSON and CSV: {matched}")
    print(f"  - Duration < {max_duration}s (KEPT): {len(filtered)}")
    print(f"  - Duration >= {max_duration}s (filtered out): {too_long}")
    
    # Verify: All filtered entries must exist in both JSON and CSV
    print(f"\n🔍 Verification:")
    all_valid = True
    for item in filtered:
        video_id = item.get('id', '')
        if video_id not in valid_csv_ids:
            print(f"  ❌ ERROR: {video_id} not in CSV but included in output!")
            all_valid = False
        duration = duration_map.get(video_id, 0)
        if duration >= max_duration:
            print(f"  ❌ ERROR: {video_id} has duration {duration}s >= {max_duration}s but included in output!")
            all_valid = False
    
    if all_valid:
        print(f"  ✅ All {len(filtered)} entries are valid (exist in both JSON and CSV, duration < {max_duration}s)")
    
    # Save filtered JSON
    print(f"\n💾 Saving filtered JSON to: {json_output}")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Successfully saved {len(filtered)} entries to {json_output}")
    print(f"   All entries are verified to exist in both JSON and CSV, with duration < {max_duration}s")
    
    return filtered

def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("🎬 Video Duration Filter")
    print("=" * 80)
    print(f"CSV file: {args.csv_file}")
    print(f"JSON input: {args.json_input}")
    print(f"JSON output: {args.json_output}")
    print(f"Max duration: {args.max_duration} seconds")
    print("=" * 80)
    
    # Load durations from CSV
    duration_map = load_duration_from_csv(args.csv_file)
    
    if not duration_map:
        print("❌ Error: No durations found in CSV file!")
        sys.exit(1)
    
    # Filter JSON by duration
    filtered_data = filter_json_by_duration(
        args.json_input,
        args.json_output,
        duration_map,
        args.max_duration
    )
    
    print("\n" + "=" * 80)
    print("✅ Filtering complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

