#!/usr/bin/env python3
"""
Filter video_ids that belong to Sign1NEWS channel from youtube-asl_metadata.csv
"""

import csv
from pathlib import Path
import sys

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

def main():
    input_file = Path("/home/mh2803/projects/sign_language_llm/youtube-asl_metadata.csv")
    output_file = Path("/home/mh2803/projects/sign_language_llm/sign1news_video_ids.txt")
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        return
    
    print("=" * 60)
    print("Filtering Sign1NEWS channel videos")
    print("=" * 60)
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print("=" * 60)
    
    video_ids = []
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            channel = row.get('channel', '').strip()
            video_id = row.get('video_id', '').strip()
            
            # Check if channel is Sign1NEWS (case-insensitive)
            if channel.lower() == 'sign1news':
                video_ids.append(video_id)
    
    # Save video IDs
    with open(output_file, 'w', encoding='utf-8') as f:
        for video_id in video_ids:
            f.write(f"{video_id}\n")
    
    # Print summary
    print(f"\n✅ Filtering complete!")
    print(f"   Total videos in CSV: {total_count:,}")
    print(f"   Sign1NEWS videos: {len(video_ids):,}")
    print(f"   Output saved to: {output_file}")
    print("=" * 60)
    
    # Show first 10 video IDs as sample
    if video_ids:
        print(f"\n📋 Sample video IDs (first 10):")
        for vid in video_ids[:10]:
            print(f"   - {vid}")
        if len(video_ids) > 10:
            print(f"   ... and {len(video_ids) - 10} more")

if __name__ == '__main__':
    main()

