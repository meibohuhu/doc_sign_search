#!/usr/bin/env python3
"""
Check how many videos from openasl-no-nad.tsv are in the cropped_videos_nonad folder.
"""

import csv
from pathlib import Path

def convert_vid_to_filename(vid):
    """Convert TSV vid format to actual filename.
    
    TSV format: LyjRsur4oTo-00:14:59.875-00:15:20.875
    File format: LyjRsur4oTo-00_14_59.875-00_15_20.875.mp4 (colon -> underscore, add .mp4)
    """
    # Replace colons with underscores and add .mp4 extension
    filename = vid.replace(':', '_') + '.mp4'
    return filename

def check_videos_in_folder(tsv_path, folder_path):
    """Check how many videos from TSV exist in the folder."""
    
    # Load all video files from the folder
    folder = Path(folder_path)
    video_files = set()
    
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return
    
    print(f"Loading video files from: {folder_path}")
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix == '.mp4':
            video_files.add(file_path.name)
    
    print(f"   Found {len(video_files)} video files in folder")
    
    # Read TSV and check videos
    print(f"\nReading TSV file: {tsv_path}")
    found_count = 0
    not_found_count = 0
    total_count = 0
    not_found_list = []
    found_list = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            vid = row.get('vid', '').strip()
            if vid:
                total_count += 1
                filename = convert_vid_to_filename(vid)
                
                if filename in video_files:
                    found_count += 1
                    found_list.append(vid)
                else:
                    not_found_count += 1
                    not_found_list.append(vid)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"   Total videos in TSV: {total_count}")
    print(f"   Videos found in folder: {found_count}")
    print(f"   Videos NOT found in folder: {not_found_count}")
    print(f"   Match rate: {found_count/total_count*100:.2f}%" if total_count > 0 else "   Match rate: N/A")
    print(f"{'='*70}")
    
    # Save lists to files
    output_dir = Path(tsv_path).parent
    
    # Save missing videos list
    missing_file = output_dir / "missing_videos_from_nonad_folder.txt"
    with open(missing_file, 'w', encoding='utf-8') as f:
        for vid in not_found_list:
            f.write(f"{vid}\n")
    print(f"\n📝 Missing videos list saved to: {missing_file}")
    
    # Save found videos list
    found_file = output_dir / "found_videos_in_nonad_folder.txt"
    with open(found_file, 'w', encoding='utf-8') as f:
        for vid in found_list:
            f.write(f"{vid}\n")
    print(f"📝 Found videos list saved to: {found_file}")
    
    return not_found_list, found_list


if __name__ == "__main__":
    tsv_path = "/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/openasl/openasl-no-nad.tsv"
    folder_path = "/shared/rc/llm-gen-agent/mhu/videos/open_asl/cropped_videos_openasl/cropped_videos_nonad"
    
    check_videos_in_folder(tsv_path, folder_path)

