#!/usr/bin/env python3
"""
Filter videos from JSON based on duration from TSV file.
Only keeps videos with duration < 10 seconds.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

def load_durations_from_tsv(tsv_file):
    """
    Load video durations from TSV file.
    Format: video_name<TAB>duration<TAB>text
    Returns: dict mapping video_name -> duration (float)
    """
    print(f"📖 Reading durations from TSV: {tsv_file}")
    durations = {}
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading TSV"), 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            video_name = parts[0].strip()
            try:
                duration = float(parts[1].strip())
                durations[video_name] = duration
            except (ValueError, IndexError):
                continue
    
    print(f"✅ Loaded {len(durations)} video durations")
    return durations

def filter_json_by_duration(json_input, json_output, durations, max_duration=10.0):
    """
    Filter JSON entries by duration from TSV.
    Only keeps entries where video duration < max_duration.
    """
    print(f"\n📖 Reading JSON file: {json_input}")
    with open(json_input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Total entries in JSON: {len(data)}")
    
    filtered = []
    matched = 0
    missing_duration = 0
    too_long = 0
    
    print(f"\n🔄 Filtering videos with duration < {max_duration}s...")
    for entry in tqdm(data, desc="Filtering entries"):
        video_file = entry.get('video', '')
        
        if not video_file:
            continue
        
        # Get duration from TSV
        duration = durations.get(video_file)
        
        if duration is None:
            missing_duration += 1
            continue
        
        matched += 1
        
        # Filter by duration: must be < max_duration
        if duration < max_duration:
            filtered.append(entry)
        else:
            too_long += 1
    
    # Save filtered dataset
    print(f"\n💾 Saving filtered JSON to: {json_output}")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\n✅ Filtering complete!")
    print(f"   Total in JSON: {len(data)}")
    print(f"   Videos with duration info: {matched}")
    print(f"   Duration < {max_duration}s (KEPT): {len(filtered)}")
    print(f"   Duration >= {max_duration}s (filtered out): {too_long}")
    print(f"   Missing duration info: {missing_duration}")
    print(f"   Kept: {len(filtered)/len(data)*100:.2f}% of original data")
    
    if filtered:
        # Show duration statistics of filtered videos
        filtered_durations = [durations.get(entry['video'], 0) for entry in filtered if entry.get('video') in durations]
        if filtered_durations:
            print(f"\n📊 Filtered videos duration statistics:")
            print(f"   Min: {min(filtered_durations):.2f}s")
            print(f"   Max: {max(filtered_durations):.2f}s")
            print(f"   Mean: {sum(filtered_durations)/len(filtered_durations):.2f}s")
            print(f"   Median: {sorted(filtered_durations)[len(filtered_durations)//2]:.2f}s")
    
    return len(filtered)

def main():
    parser = argparse.ArgumentParser(
        description='Filter JSON dataset by video duration from TSV file'
    )
    parser.add_argument(
        '--tsv_file',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/manifests/train.tsv',
        help='Path to TSV file with video timing information'
    )
    parser.add_argument(
        '--json_input',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/vanshika/asl_test/train_ssvp_updated_diverse.json',
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--json_output',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/vanshika/asl_test/train_ssvp_updated_diverse_under_10s.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=10.0,
        help='Maximum duration in seconds (default: 10.0)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    tsv_path = Path(args.tsv_file)
    json_path = Path(args.json_input)
    
    if not tsv_path.exists():
        print(f"❌ Error: TSV file not found: {tsv_path}")
        return 1
    
    if not json_path.exists():
        print(f"❌ Error: JSON file not found: {json_path}")
        return 1
    
    print("="*70)
    print("FILTERING VIDEOS BY DURATION (< 10s)")
    print("="*70)
    print(f"TSV file: {args.tsv_file}")
    print(f"Input JSON: {args.json_input}")
    print(f"Output JSON: {args.json_output}")
    print(f"Max duration: {args.max_duration}s")
    print()
    
    try:
        # Load durations
        durations = load_durations_from_tsv(args.tsv_file)
        
        # Filter JSON
        count = filter_json_by_duration(
            args.json_input,
            args.json_output,
            durations,
            args.max_duration
        )
        
        print("\n" + "="*70)
        print(f"✅ Success! Saved {count} entries to {args.json_output}")
        print("="*70)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
