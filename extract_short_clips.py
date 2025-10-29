#!/usr/bin/env python3
"""
Extract video clips < 10s from How2Sign CSV, filter corrupted videos, 
and convert to JSON format similar to segmented_train_videos.json
"""

import csv
import json
import os

def load_corrupted_videos_list(corrupted_file_path):
    """Load the list of corrupted video files"""
    corrupted_videos = set()
    
    if os.path.exists(corrupted_file_path):
        with open(corrupted_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.endswith('.mp4') and not line.startswith('Corrupted'):
                    corrupted_videos.add(line)
    
    print(f"📋 Loaded {len(corrupted_videos)} corrupted video files")
    return corrupted_videos

def extract_short_clips(csv_file_path, corrupted_videos, max_duration=10.0):
    """Extract video clips shorter than max_duration seconds"""
    
    print(f"📊 Extracting clips shorter than {max_duration} seconds...")
    
    short_clips = []
    total_processed = 0
    filtered_corrupted = 0
    filtered_long = 0
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        for row in reader:
            total_processed += 1
            
            try:
                # Extract video information
                video_name = row['VIDEO_NAME']
                sentence_id = row['SENTENCE_ID']
                sentence_name = row['SENTENCE_NAME']
                start_time = float(row['START_REALIGNED'])
                end_time = float(row['END_REALIGNED'])
                sentence_text = row['SENTENCE']
                
                # Calculate duration
                duration = end_time - start_time
                
                # Check if duration is within limit
                if duration >= max_duration:
                    filtered_long += 1
                    continue
                
                # Check if video is corrupted
                video_filename = f"{sentence_name}.mp4"
                if video_filename in corrupted_videos:
                    filtered_corrupted += 1
                    continue
                
                # Create JSON entry
                clip_entry = {
                    "id": sentence_name,
                    "video": video_filename,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<video>\nTranslate the American Sign Language in this video to English."
                        },
                        {
                            "from": "gpt", 
                            "value": sentence_text
                        }
                    ]
                }
                
                short_clips.append(clip_entry)
                
            except (ValueError, KeyError) as e:
                print(f"⚠️  Skipping row {total_processed} due to error: {e}")
                continue
    
    print(f"\n📈 EXTRACTION SUMMARY:")
    print(f"📋 Total rows processed: {total_processed:,}")
    print(f"📋 Short clips (< {max_duration}s): {len(short_clips):,}")
    print(f"📋 Filtered (corrupted): {filtered_corrupted:,}")
    print(f"📋 Filtered (too long): {filtered_long:,}")
    print(f"📋 Success rate: {len(short_clips)/total_processed*100:.1f}%")
    
    return short_clips

def save_json_file(clips, output_file_path):
    """Save clips to JSON file"""
    
    print(f"\n💾 Saving {len(clips):,} clips to {output_file_path}...")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully saved to {output_file_path}")
    
    # Calculate file size
    file_size = os.path.getsize(output_file_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.2f} MB")

def analyze_duration_distribution(clips):
    """Analyze duration distribution of extracted clips"""
    
    if not clips:
        print("❌ No clips to analyze")
        return
    
    durations = []
    for clip in clips:
        # Extract duration from video filename or calculate from conversations
        # For now, we'll estimate based on typical clip lengths
        durations.append(5.0)  # Placeholder - would need actual duration calculation
    
    print(f"\n📊 DURATION ANALYSIS OF EXTRACTED CLIPS:")
    print(f"📋 Total clips: {len(clips):,}")
    print(f"📋 All clips are < 10 seconds (as filtered)")

def main():
    # File paths
    csv_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/how2sign_realigned_train.csv"
    corrupted_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/corrupted_videos_list.txt"
    output_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/short_clips_under_10s.json"
    
    print("🔍 EXTRACTING SHORT VIDEO CLIPS (< 10s)")
    print("=" * 50)
    
    # Load corrupted videos list
    corrupted_videos = load_corrupted_videos_list(corrupted_file)
    
    # Extract short clips
    short_clips = extract_short_clips(csv_file, corrupted_videos, max_duration=10.0)
    
    if short_clips:
        # Save to JSON file
        save_json_file(short_clips, output_file)
        
        # Analyze duration distribution
        analyze_duration_distribution(short_clips)
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Total clips extracted: {len(short_clips):,}")
    else:
        print("❌ No clips found matching criteria")

if __name__ == "__main__":
    main()

