#!/usr/bin/env python3
"""
Filter dailymoth_openasl_merged.json to only keep entries where the video file
exists in either cropped_videos_nonad or dailymoth folders.
"""

import json
import os
from pathlib import Path

def convert_video_name_to_filename(video_name):
    """Convert JSON video name to actual filename.
    
    JSON format: Mci9oyb5V2E-00:00:06.000-00:00:06.589.mp4
    File format: Mci9oyb5V2E-00_00_06.000-00_00_06.589.mp4 (colon -> underscore)
    """
    # Replace colons with underscores in the time segments
    # Pattern: VIDEO_ID-00:00:06.000-00:00:06.589.mp4 -> VIDEO_ID-00_00_06.000-00_00_06.589.mp4
    parts = video_name.split('-', 1)
    if len(parts) == 2 and ':' in parts[1]:
        video_id = parts[0]
        time_part = parts[1]
        # Replace colons with underscores in time segments
        time_part = time_part.replace(':', '_')
        return f"{video_id}-{time_part}"
    return video_name

def load_video_files_from_folder(folder_path):
    """Load all video filenames from a folder into a set."""
    video_files = set()
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder does not exist: {folder_path}")
        return video_files
    
    print(f"Loading video files from: {folder_path}")
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix == '.mp4':
            video_files.add(file_path.name)
    
    print(f"   Loaded {len(video_files)} video files")
    return video_files

def filter_json_by_video_files(json_path, nonad_folder, dailymoth_folder, output_path=None):
    """Filter JSON to only keep entries where video file exists."""
    
    # Load video files from both folders
    nonad_videos = load_video_files_from_folder(nonad_folder)
    dailymoth_videos = load_video_files_from_folder(dailymoth_folder)
    
    # Combine both sets (union)
    all_videos = nonad_videos | dailymoth_videos
    print(f"\nTotal unique video files: {len(all_videos)}")
    
    # Load JSON
    print(f"\nLoading JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original entries: {len(data)}")
    
    # Filter entries
    filtered_data = []
    removed_count = 0
    nonad_count = 0
    dailymoth_count = 0
    not_found_count = 0
    
    for entry in data:
        video_name = entry.get('video', '').strip()
        
        if not video_name:
            removed_count += 1
            not_found_count += 1
            continue
        
        # Check if video exists as-is (for dailymoth)
        if video_name in all_videos:
            filtered_data.append(entry)
            if video_name in dailymoth_videos:
                dailymoth_count += 1
            if video_name in nonad_videos:
                nonad_count += 1
        else:
            # Try converting format (for openasl: colon -> underscore)
            converted_name = convert_video_name_to_filename(video_name)
            if converted_name in all_videos:
                filtered_data.append(entry)
                if converted_name in nonad_videos:
                    nonad_count += 1
                if converted_name in dailymoth_videos:
                    dailymoth_count += 1
            else:
                # Video file not found
                removed_count += 1
                not_found_count += 1
    
    # Write filtered data
    if output_path is None:
        output_path = json_path
    
    print(f"\nWriting filtered JSON to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Filtered JSON file created: {output_path}")
    print(f"   Original entries: {len(data)}")
    print(f"   Removed entries (video not found): {removed_count}")
    print(f"   Remaining entries: {len(filtered_data)}")
    print(f"   - From nonad folder: {nonad_count}")
    print(f"   - From dailymoth folder: {dailymoth_count}")


if __name__ == "__main__":
    json_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/dailymoth_openasl_merged.json"
    nonad_folder = "/shared/rc/llm-gen-agent/mhu/videos/open_asl/cropped_videos_openasl/cropped_videos_nonad"
    dailymoth_folder = "/shared/rc/llm-gen-agent/mhu/videos/open_asl/cropped_videos_openasl/dailymoth"
    
    filter_json_by_video_files(json_path, nonad_folder, dailymoth_folder)






