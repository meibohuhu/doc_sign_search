#!/usr/bin/env python3
"""
Filter dailymoth_openasl_merged.json to only keep clips that appear in 
openasl-v1.0_filtered.csv or dailymoth_train.json
"""

import json
import csv

def convert_vid_to_json_id(vid):
    """Convert CSV vid format to JSON id format.
    
    CSV: Mci9oyb5V2E-00:00:06.000-00:00:06.589
    JSON: Mci9oyb5V2E-00_00_06_000-00_00_06_589
    """
    # Replace : with _ and . with _
    return vid.replace(':', '_').replace('.', '_')

def load_openasl_ids(csv_path):
    """Load vid IDs from openasl-v1.0_filtered.csv and convert to JSON id format."""
    ids = set()
    print(f"Loading OpenASL IDs from: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get('vid', '').strip()
            if vid:
                json_id = convert_vid_to_json_id(vid)
                ids.add(json_id)
                # Also add the original vid in case it's used directly
                ids.add(vid)
    
    print(f"   Loaded {len(ids)} OpenASL IDs")
    return ids

def load_dailymoth_ids(json_path):
    """Load clip IDs from dailymoth_train.json."""
    ids = set()
    print(f"Loading DailyMoth IDs from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            entry_id = entry.get('id', '').strip()
            if entry_id:
                ids.add(entry_id)
    
    print(f"   Loaded {len(ids)} DailyMoth IDs")
    return ids

def filter_merged_json(merged_json_path, openasl_csv_path, dailymoth_json_path, output_path=None):
    """Filter merged JSON to only keep clips from specified sources."""
    
    # Load valid IDs from both sources (union)
    openasl_ids = load_openasl_ids(openasl_csv_path)
    dailymoth_ids = load_dailymoth_ids(dailymoth_json_path)
    
    # Combine both sets (union)
    valid_ids = openasl_ids | dailymoth_ids
    print(f"\nTotal unique valid IDs (union): {len(valid_ids)}")
    
    # Load merged JSON
    print(f"\nLoading merged JSON: {merged_json_path}")
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original entries: {len(data)}")
    
    # Filter entries
    filtered_data = []
    removed_count = 0
    openasl_kept = 0
    dailymoth_kept = 0
    
    for entry in data:
        entry_id = entry.get('id', '').strip()
        
        if entry_id in valid_ids:
            filtered_data.append(entry)
            if entry_id in openasl_ids:
                openasl_kept += 1
            if entry_id in dailymoth_ids:
                dailymoth_kept += 1
        else:
            removed_count += 1
    
    # Write filtered data
    if output_path is None:
        output_path = merged_json_path
    
    print(f"\nWriting filtered JSON to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Filtered JSON file created: {output_path}")
    print(f"   Original entries: {len(data)}")
    print(f"   Removed entries: {removed_count}")
    print(f"   Remaining entries: {len(filtered_data)}")
    print(f"   - From OpenASL: {openasl_kept}")
    print(f"   - From DailyMoth: {dailymoth_kept}")


if __name__ == "__main__":
    merged_json_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/dailymoth_openasl_merged.json"
    openasl_csv_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/openasl-v1.0_filtered.csv"
    dailymoth_json_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/dailymoth_train.json"
    
    filter_merged_json(merged_json_path, openasl_csv_path, dailymoth_json_path)






