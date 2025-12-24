#!/usr/bin/env python3
"""
Convert openasl-nad-only.tsv to merged_train.json format.
"""

import json
import csv
import random
from pathlib import Path


def convert_time_format(time_str, for_id=False):
    """
    Convert time format from TSV to JSON format.
    
    TSV format: 00:00:06.000
    JSON id format: 00_00_06_000 (underscores, milliseconds with underscore)
    JSON video format: 00_00_06.000 (underscores, milliseconds with dot)
    
    Args:
        time_str: Time string in format "00:00:06.000"
        for_id: If True, use format for id (with underscore before milliseconds)
                If False, use format for video (with dot before milliseconds)
    
    Returns:
        Converted time string
    """
    # Split by colon and dot
    parts = time_str.replace(':', '_').split('.')
    if len(parts) == 2:
        time_part = parts[0]  # 00_00_06
        millis = parts[1]     # 000
        if for_id:
            return f"{time_part}_{millis}"  # 00_00_06_000
        else:
            return f"{time_part}.{millis}"  # 00_00_06.000
    else:
        # Fallback if format is unexpected
        return time_str.replace(':', '_')


def convert_tsv_to_json(tsv_file, output_file, sample_size=None, random_seed=42):
    """
    Convert TSV file to JSON format matching merged_train.json.
    
    Args:
        tsv_file: Path to input TSV file
        output_file: Path to output JSON file
        sample_size: If specified, randomly sample this many entries. If None, use all entries.
        random_seed: Random seed for reproducibility
    """
    print(f"Reading TSV file: {tsv_file}")
    
    # First, read all rows
    all_rows = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            all_rows.append(row)
    
    print(f"Total entries in TSV: {len(all_rows)}")
    
    # Sample if requested
    if sample_size is not None:
        if sample_size > len(all_rows):
            print(f"⚠️  Warning: Requested sample size ({sample_size}) is larger than total entries ({len(all_rows)}). Using all entries.")
            selected_rows = all_rows
        else:
            random.seed(random_seed)
            selected_rows = random.sample(all_rows, sample_size)
            print(f"Randomly sampled {len(selected_rows)} entries (seed={random_seed})")
    else:
        selected_rows = all_rows
    
    results = []
    
    for row_idx, row in enumerate(selected_rows):
        yid = row['yid']
        start_time = row['start']
        end_time = row['end']
        raw_text = row['raw-text']
        
        # Convert time formats
        start_id_format = convert_time_format(start_time, for_id=True)
        end_id_format = convert_time_format(end_time, for_id=True)
        start_video_format = convert_time_format(start_time, for_id=False)
        end_video_format = convert_time_format(end_time, for_id=False)
        
        # Create id: yid-start_time-end_time (with underscores)
        entry_id = f"{yid}-{start_id_format}-{end_id_format}"
        
        # Create video: yid-start_time-end_time.mp4 (with dots for milliseconds)
        video_name = f"{yid}-{start_video_format}-{end_video_format}.mp4"
        
        # Create conversations structure
        entry = {
            "id": entry_id,
            "video": video_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "<video>\nTranslate the American Sign Language in this video to English."
                },
                {
                    "from": "gpt",
                    "value": raw_text
                }
            ]
        }
        
        results.append(entry)
        
        if (row_idx + 1) % 1000 == 0:
            print(f"  Processed {row_idx + 1} entries...")
    
    print(f"\nTotal entries to write: {len(results)}")
    print(f"Writing to JSON file: {output_file}")
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Conversion complete! Output written to {output_file}")
    
    # Print a sample entry for verification
    if results:
        print(f"\nSample entry:")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))


def main():
    tsv_file = Path("/home/mh2803/projects/sign_language_llm/openasl/openasl-no-part2.tsv")
    output_file = Path("/home/mh2803/projects/sign_language_llm/openasl/openasl-no-part2-sampled.json")
    sample_size = 17000  # Randomly sample 17000 entries
    
    if not tsv_file.exists():
        print(f"❌ Error: {tsv_file} not found")
        return
    
    convert_tsv_to_json(tsv_file, output_file, sample_size=sample_size)


if __name__ == '__main__':
    main()
