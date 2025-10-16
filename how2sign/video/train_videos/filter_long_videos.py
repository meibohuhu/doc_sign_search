#!/usr/bin/env python3
"""
Filter how2sign training data to remove very long videos that cause OOM.
This keeps 96.6% of the data while preventing OOM at 12fps.
"""

import json
import csv

# Configuration
MAX_DURATION = 30.0  # seconds - keeps 99.0% of data, removes 256 longest videos
INPUT_JSON = "segmented_train_videos.json"
OUTPUT_JSON = "segmented_train_videos_filtered.json"
CSV_FILE = "how2sign_realigned_train.csv"

def load_durations_from_csv(csv_file):
    """Load video durations from CSV."""
    durations = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sentence_name = row['SENTENCE_NAME']
            start = float(row['START_REALIGNED'])
            end = float(row['END_REALIGNED'])
            duration = end - start
            durations[sentence_name] = duration
    return durations

def filter_dataset(input_json, output_json, durations, max_duration):
    """Filter JSON dataset to remove long videos."""
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset: {len(data)} samples")
    
    filtered_data = []
    removed_count = 0
    removed_durations = []
    
    for item in data:
        # Extract video filename (without extension)
        video_file = item.get('video', '')
        video_name = video_file.replace('.mp4', '').replace('.avi', '')
        
        # Check duration
        duration = durations.get(video_name, 0)
        
        if duration <= max_duration:
            filtered_data.append(item)
        else:
            removed_count += 1
            removed_durations.append(duration)
    
    # Save filtered dataset
    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\nFiltered dataset: {len(filtered_data)} samples")
    print(f"Removed: {removed_count} samples ({removed_count/len(data)*100:.2f}%)")
    print(f"Kept: {len(filtered_data)/len(data)*100:.2f}% of data")
    
    if removed_durations:
        print(f"\nRemoved video statistics:")
        print(f"  Min duration: {min(removed_durations):.2f}s")
        print(f"  Max duration: {max(removed_durations):.2f}s")
        print(f"  Mean duration: {sum(removed_durations)/len(removed_durations):.2f}s")
    
    print(f"\nFiltered dataset saved to: {output_json}")

if __name__ == "__main__":
    print("="*70)
    print("HOW2SIGN DATASET FILTERING FOR OOM PREVENTION")
    print("="*70)
    print(f"\nMax duration threshold: {MAX_DURATION}s")
    print(f"At 12 FPS, max frames: {int(MAX_DURATION * 12)}")
    print()
    
    # Load durations
    print("Loading video durations from CSV...")
    durations = load_durations_from_csv(CSV_FILE)
    print(f"Loaded {len(durations)} video durations")
    
    # Filter dataset
    print("\nFiltering dataset...")
    filter_dataset(INPUT_JSON, OUTPUT_JSON, durations, MAX_DURATION)
    
    print("\n" + "="*70)
    print("✅ Done! Use the filtered JSON in your training script.")
    print("="*70)

