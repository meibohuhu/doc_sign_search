#!/usr/bin/env python3
"""
Filter segmented_test_videos.json to remove videos over 30 seconds.
Uses how2sign_realigned_test.csv to get video durations.
"""

import json
import csv
from collections import defaultdict

# File paths
csv_file = "how2sign_realigned_test.csv"
json_file = "segmented_test_videos.json"
output_file = "segmented_test_videos_filtered.json"

# Load CSV to get video durations
print("Loading CSV file...")
video_durations = {}
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        sentence_id = row['SENTENCE_ID']
        video_name = row['SENTENCE_NAME']
        start_time = float(row['START_REALIGNED'])
        end_time = float(row['END_REALIGNED'])
        
        # Calculate duration for this sentence segment
        duration = end_time - start_time
        
        # Track the maximum duration for each video
        # Using SENTENCE_NAME as the key (e.g., "g3kFAmcBpFc_13-3-rgb_front")
        if video_name not in video_durations or duration > video_durations[video_name]:
            video_durations[video_name] = duration

print(f"Loaded durations for {len(video_durations)} videos from CSV")

# Load JSON file
print(f"Loading JSON file...")
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Original JSON contains {len(data)} videos")

# Filter videos
filtered_data = []
videos_over_30s = []
videos_under_30s = []

for video_entry in data:
    video_id = video_entry['id']
    
    # Get duration from CSV, default to 0 if not found
    duration = video_durations.get(video_id, 0)
    
    if duration <= 30.0:
        filtered_data.append(video_entry)
        videos_under_30s.append(video_id)
    else:
        videos_over_30s.append((video_id, duration))

print(f"\nFiltering results:")
print(f"  Videos under or equal to 30s: {len(videos_under_30s)}")
print(f"  Videos over 30s: {len(videos_over_30s)}")

# Show some examples of videos over 30s
if videos_over_30s:
    print(f"\nSample videos over 30s:")
    for video_id, duration in videos_over_30s[:10]:
        print(f"  {video_id}: {duration:.2f}s")

# Save filtered JSON
print(f"\nSaving filtered data to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"✅ Done! Filtered JSON saved to {output_file}")
print(f"   Removed {len(videos_over_30s)} videos over 30 seconds")

# Save list of removed videos
removed_file = "removed_videos_over_30s.txt"
with open(removed_file, 'w', encoding='utf-8') as f:
    f.write("Videos removed (over 30 seconds):\n")
    f.write("=" * 80 + "\n\n")
    for video_id, duration in videos_over_30s:
        f.write(f"{video_id}: {duration:.2f}s\n")

print(f"   List of removed videos saved to {removed_file}")

