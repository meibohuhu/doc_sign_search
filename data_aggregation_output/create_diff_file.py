#!/usr/bin/env python3
"""
Create a new file containing video IDs that are in notstrict file but not in stage2 file
"""
from pathlib import Path

def read_video_ids(file_path):
    """Read video IDs from file"""
    video_ids = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            video_id = line.strip()
            if video_id:
                video_ids.add(video_id)
    return video_ids

def main():
    script_dir = Path(__file__).parent
    
    # Input files
    notstrict_file = script_dir / "youtube_video_ids_stage2_notstrict.txt"
    stage2_file = script_dir / "youtube_video_ids_stage2.txt"
    
    # Output file
    output_file = script_dir / "youtube_video_ids_diff.txt"
    
    # Read video IDs from both files
    print(f"Reading video IDs from: {notstrict_file}")
    notstrict_ids = read_video_ids(notstrict_file)
    print(f"Found {len(notstrict_ids)} video IDs in notstrict file")
    
    print(f"Reading video IDs from: {stage2_file}")
    stage2_ids = read_video_ids(stage2_file)
    print(f"Found {len(stage2_ids)} video IDs in stage2 file")
    
    # Find IDs that are in notstrict but not in stage2
    diff_ids = notstrict_ids - stage2_ids
    print(f"\nFound {len(diff_ids)} video IDs in notstrict but not in stage2")
    
    # Write to output file (preserve order from notstrict file)
    print(f"Writing to: {output_file}")
    with open(notstrict_file, 'r', encoding='utf-8') as f:
        notstrict_lines = [line.strip() for line in f if line.strip()]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        written_count = 0
        for line in notstrict_lines:
            video_id = line.strip()
            if video_id in diff_ids:
                f.write(f"{video_id}\n")
                written_count += 1
    
    print(f"✅ Successfully wrote {written_count} video IDs to {output_file}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()

