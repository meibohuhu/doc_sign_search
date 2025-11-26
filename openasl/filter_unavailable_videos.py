#!/usr/bin/env python3
"""
Filter out clips from unavailable videos in openasl-v1.0.tsv
"""

import csv

def filter_unavailable_videos(tsv_path, unavailable_list_path, output_path):
    """Filter out rows where yid is in the unavailable video list."""
    
    # Read unavailable video IDs
    unavailable_videos = set()
    with open(unavailable_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            video_id = line.strip()
            if video_id:
                unavailable_videos.add(video_id)
    
    print(f"Loaded {len(unavailable_videos)} unavailable video IDs")
    
    # Read TSV and filter
    filtered_rows = []
    removed_count = 0
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        
        for row in reader:
            yid = row.get('yid', '').strip()
            if yid not in unavailable_videos:
                filtered_rows.append(row)
            else:
                removed_count += 1
    
    # Write filtered data to CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"✅ Filtered CSV file created: {output_path}")
    print(f"   Original rows: {len(filtered_rows) + removed_count}")
    print(f"   Removed rows: {removed_count}")
    print(f"   Remaining rows: {len(filtered_rows)}")


if __name__ == "__main__":
    tsv_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/openasl-v1.0.tsv"
    unavailable_list_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/unavailable_video_list.txt"
    output_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/openasl-v1.0_filtered.csv"
    
    filter_unavailable_videos(tsv_path, unavailable_list_path, output_path)

