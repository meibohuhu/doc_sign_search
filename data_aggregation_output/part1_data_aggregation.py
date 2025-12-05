#!/usr/bin/env python3
"""
Part 1: Data Aggregation (Video-Level)
Aggregates captions for How2Sign and YouTube datasets.
"""

import pandas as pd
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import sys

def aggregate_how2sign_videos(csv_path):
    """
    Aggregate How2Sign sentences by VIDEO_ID.
    
    Args:
        csv_path: Path to how2sign_realigned_train.csv
        
    Returns:
        Dictionary mapping video_id → full_text
    """
    print("=" * 60)
    print("Part 1.1: How2Sign Video Text Aggregation")
    print("=" * 60)
    
    # Read CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, sep='\t')
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Group sentences by VIDEO_ID
    how2sign_videos = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping sentences"):
        video_id = row['VIDEO_ID']
        sentence = row['SENTENCE']
        if pd.notna(sentence):  # Skip NaN sentences
            how2sign_videos[video_id].append(str(sentence))
    
    # Combine into full video text
    how2sign_full_texts = {}
    for video_id, sentences in tqdm(how2sign_videos.items(), desc="Combining sentences"):
        full_text = " ".join(sentences)
        how2sign_full_texts[video_id] = full_text
    
    print(f"\n✅ Aggregated {len(how2sign_full_texts)} unique videos")
    print(f"   Average text length: {sum(len(t) for t in how2sign_full_texts.values()) / len(how2sign_full_texts):.0f} characters")
    print(f"   Average word count: {sum(len(t.split()) for t in how2sign_full_texts.values()) / len(how2sign_full_texts):.0f} words")
    
    return how2sign_full_texts


def aggregate_youtube_videos(csv_path):
    """
    Aggregate YouTube video text from title, description, and captions.
    
    Args:
        csv_path: Path to youtube-asl_metadata.csv
        
    Returns:
        Dictionary mapping video_id → {text, title, description, caption_only, metadata}
    """
    print("\n" + "=" * 60)
    print("Part 1.2: YouTube Video Text Aggregation")
    print("=" * 60)
    
    # Read CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    youtube_videos = {}
    failed_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_id = row['video_id']
        
        # Get title and description (handle NaN)
        title = str(row['title']) if pd.notna(row['title']) else ""
        description = str(row['description']) if pd.notna(row['description']) else ""
        
        # Parse caption_timestamps JSON
        caption_text = ""
        caption_only = ""
        
        if pd.notna(row['caption_timestamps']):
            try:
                caption_json = json.loads(row['caption_timestamps'])
                if isinstance(caption_json, list):
                    caption_text = " ".join([seg.get('text', '') for seg in caption_json if 'text' in seg])
                    caption_only = caption_text
            except (json.JSONDecodeError, TypeError) as e:
                failed_count += 1
                if failed_count <= 5:  # Print first 5 errors
                    tqdm.write(f"⚠️  Failed to parse captions for {video_id}: {e}")
        
        # Combine: title + description + captions
        full_text = f"{title} {description} {caption_text}".strip()
        
        # Store metadata
        youtube_videos[video_id] = {
            'text': full_text,
            'title': title,
            'description': description,
            'caption_only': caption_only,
            'metadata': {
                'channel': str(row.get('channel', '')) if pd.notna(row.get('channel')) else "",
                'duration': row.get('duration', 0),
                'view_count': row.get('view_count', 0),
                'upload_date': str(row.get('upload_date', '')) if pd.notna(row.get('upload_date')) else "",
                'has_captions': row.get('has_captions', False),
            }
        }
    
    if failed_count > 0:
        print(f"\n⚠️  Failed to parse captions for {failed_count} videos")
    
    # Calculate statistics
    valid_texts = [v['text'] for v in youtube_videos.values() if v['text'].strip()]
    print(f"\n✅ Aggregated {len(youtube_videos)} videos")
    print(f"   Videos with text: {len(valid_texts)}")
    if valid_texts:
        print(f"   Average text length: {sum(len(t) for t in valid_texts) / len(valid_texts):.0f} characters")
        print(f"   Average word count: {sum(len(t.split()) for t in valid_texts) / len(valid_texts):.0f} words")
    
    return youtube_videos


def main():
    # Input paths
    how2sign_csv = Path("/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/how2sign_realigned_train.csv")
    youtube_csv = Path("/home/mh2803/projects/sign_language_llm/youtube-asl_metadata.csv")
    
    # Output paths
    output_dir = Path("/home/mh2803/projects/sign_language_llm/data_aggregation_output")
    output_dir.mkdir(exist_ok=True)
    
    how2sign_output = output_dir / "how2sign_video_texts.json"
    youtube_output = output_dir / "youtube_video_texts.json"
    
    # Process How2Sign
    if not how2sign_csv.exists():
        print(f"❌ Error: {how2sign_csv} not found")
        sys.exit(1)
    
    how2sign_full_texts = aggregate_how2sign_videos(how2sign_csv)
    
    # Save How2Sign results
    print(f"\n💾 Saving How2Sign results to {how2sign_output}...")
    with open(how2sign_output, 'w', encoding='utf-8') as f:
        json.dump(how2sign_full_texts, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(how2sign_full_texts)} videos")
    
    # Process YouTube
    if not youtube_csv.exists():
        print(f"❌ Error: {youtube_csv} not found")
        sys.exit(1)
    
    youtube_videos = aggregate_youtube_videos(youtube_csv)
    
    # Save YouTube results
    print(f"\n💾 Saving YouTube results to {youtube_output}...")
    with open(youtube_output, 'w', encoding='utf-8') as f:
        json.dump(youtube_videos, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(youtube_videos)} videos")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Part 1: Data Aggregation Complete!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"   - How2Sign: {how2sign_output}")
    print(f"   - YouTube: {youtube_output}")
    print("=" * 60)


if __name__ == '__main__':
    main()

