#!/usr/bin/env python3
"""
Convert dailymoth and openasl TSV files to JSON format and merge them.
"""

import json
import csv
import os
from pathlib import Path

def convert_dailymoth_tsv_to_json(tsv_path, output_json_path, video_base_path):
    """Convert dailymoth train.tsv to JSON format."""
    data = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            video_filename = parts[0]
            duration = parts[1]
            text = parts[2]
            
            # Remove .mp4 extension for id
            video_id = video_filename.replace('.mp4', '')
            
            entry = {
                "id": video_id,
                "video": video_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\nTranslate the American Sign Language in this video to English."
                    },
                    {
                        "from": "gpt",
                        "value": text
                    }
                ]
            }
            data.append(entry)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(data)} dailymoth entries to {output_json_path}")
    return data


def convert_openasl_tsv_to_json(tsv_path, output_json_path, video_base_path):
    """Convert openasl openasl-v1.0.tsv to JSON format (only train split, using raw-text)."""
    data = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Only process train split
            if row.get('split', '').strip() != 'train':
                continue
            
            vid = row.get('vid', '').strip()
            raw_text = row.get('raw-text', '').strip()
            
            if not vid or not raw_text:
                continue
            
            # Extract video filename from vid (format: Mci9oyb5V2E-00:00:06.000-00:00:06.589)
            # We need to construct the video filename
            # The vid format suggests the video file might be named differently
            # Let's use the vid as the video identifier
            video_filename = f"{vid}.mp4"
            video_id = vid.replace(':', '_').replace('.', '_')
            
            entry = {
                "id": video_id,
                "video": video_filename,
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
            data.append(entry)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(data)} openasl entries to {output_json_path}")
    return data


def merge_json_files(json_paths, output_json_path):
    """Merge multiple JSON files into one."""
    merged_data = []
    
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Merged {len(merged_data)} entries to {output_json_path}")
    return merged_data


if __name__ == "__main__":
    # File paths
    dailymoth_tsv = "/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/manifests/train.tsv"
    openasl_tsv = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/openasl-v1.0.tsv"
    
    output_dir = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval"
    dailymoth_json = os.path.join(output_dir, "dailymoth_train.json")
    openasl_json = os.path.join(output_dir, "openasl_train.json")
    merged_json = os.path.join(output_dir, "dailymoth_openasl_merged.json")
    
    # Convert dailymoth
    print("Converting dailymoth TSV to JSON...")
    dailymoth_data = convert_dailymoth_tsv_to_json(dailymoth_tsv, dailymoth_json, None)
    
    # Convert openasl
    print("\nConverting openasl TSV to JSON (train split only)...")
    openasl_data = convert_openasl_tsv_to_json(openasl_tsv, openasl_json, None)
    
    # Merge
    print("\nMerging JSON files...")
    merged_data = merge_json_files([dailymoth_json, openasl_json], merged_json)
    
    print(f"\n✅ Complete! Total entries: {len(merged_data)}")
    print(f"   - Dailymoth: {len(dailymoth_data)} entries")
    print(f"   - OpenASL: {len(openasl_data)} entries")

