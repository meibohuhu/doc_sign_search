#!/usr/bin/env python3
"""
Merge two JSON datasets with 70/30 split.
70% from How2Sign, 30% from DailyMoth.
"""

import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def load_json(file_path):
    """Load JSON file."""
    print(f"📖 Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   Loaded {len(data)} entries")
    return data

def sample_entries(data, ratio, seed=None):
    """
    Randomly sample entries from dataset.
    
    Args:
        data: List of entries
        ratio: Fraction to sample (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Sampled entries
    """
    if seed is not None:
        random.seed(seed)
    
    n_samples = int(len(data) * ratio)
    sampled = random.sample(data, n_samples)
    return sampled

def merge_datasets(
    how2sign_path,
    dailymoth_path,
    output_path,
    how2sign_ratio=0.7,
    dailymoth_ratio=0.3,
    seed=42,
    shuffle_final=True
):
    """
    Merge two datasets with specified ratios.
    
    Args:
        how2sign_path: Path to How2Sign JSON file
        dailymoth_path: Path to DailyMoth JSON file
        output_path: Path to output merged JSON file
        how2sign_ratio: Fraction of How2Sign to use (default: 0.7)
        dailymoth_ratio: Fraction of DailyMoth to use (default: 0.3)
        seed: Random seed for reproducibility
        shuffle_final: Whether to shuffle the final merged dataset
    """
    print("="*70)
    print("MERGING DATASETS: 70% How2Sign + 30% DailyMoth")
    print("="*70)
    print()
    
    # Load datasets
    how2sign_data = load_json(how2sign_path)
    dailymoth_data = load_json(dailymoth_path)
    
    print(f"\n📊 Dataset sizes:")
    print(f"   How2Sign: {len(how2sign_data)} entries")
    print(f"   DailyMoth: {len(dailymoth_data)} entries")
    
    # Sample entries
    print(f"\n🔄 Sampling entries...")
    print(f"   Sampling {how2sign_ratio*100:.0f}% from How2Sign...")
    how2sign_sampled = sample_entries(how2sign_data, how2sign_ratio, seed=seed)
    
    print(f"   Sampling {dailymoth_ratio*100:.0f}% from DailyMoth...")
    # Use different seed for DailyMoth to ensure independent sampling
    dailymoth_sampled = sample_entries(dailymoth_data, dailymoth_ratio, seed=seed+1)
    
    print(f"\n📊 Sampled sizes:")
    print(f"   How2Sign: {len(how2sign_sampled)} entries ({len(how2sign_sampled)/len(how2sign_data)*100:.1f}%)")
    print(f"   DailyMoth: {len(dailymoth_sampled)} entries ({len(dailymoth_sampled)/len(dailymoth_data)*100:.1f}%)")
    
    # Merge datasets
    print(f"\n🔀 Merging datasets...")
    merged_data = how2sign_sampled + dailymoth_sampled
    
    # Shuffle final dataset
    if shuffle_final:
        print(f"   Shuffling merged dataset...")
        random.seed(seed+2)  # Different seed for final shuffle
        random.shuffle(merged_data)
    
    # Calculate final ratio
    total = len(merged_data)
    how2sign_pct = len(how2sign_sampled) / total * 100
    dailymoth_pct = len(dailymoth_sampled) / total * 100
    
    print(f"\n📊 Final merged dataset:")
    print(f"   Total entries: {total}")
    print(f"   How2Sign: {len(how2sign_sampled)} entries ({how2sign_pct:.1f}%)")
    print(f"   DailyMoth: {len(dailymoth_sampled)} entries ({dailymoth_pct:.1f}%)")
    
    # Save merged dataset
    print(f"\n💾 Saving merged dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Success! Saved {total} entries to {output_path}")
    print("="*70)
    
    return merged_data

def main():
    parser = argparse.ArgumentParser(
        description='Merge two JSON datasets with 70/30 split'
    )
    parser.add_argument(
        '--how2sign_json',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_corrupted_removed.json',
        help='Path to How2Sign JSON file'
    )
    parser.add_argument(
        '--dailymoth_json',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/vanshika/asl_test/train_ssvp_updated_diverse_under_10s.json',
        help='Path to DailyMoth JSON file'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='/home/mh2803/projects/sign_language_llm/vanshika/asl_test/train_merged_70_30.json',
        help='Path to output merged JSON file'
    )
    parser.add_argument(
        '--how2sign_ratio',
        type=float,
        default=0.7,
        help='Fraction of How2Sign to use (default: 0.7)'
    )
    parser.add_argument(
        '--dailymoth_ratio',
        type=float,
        default=0.3,
        help='Fraction of DailyMoth to use (default: 0.3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the final merged dataset'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    how2sign_path = Path(args.how2sign_json)
    dailymoth_path = Path(args.dailymoth_json)
    
    if not how2sign_path.exists():
        print(f"❌ Error: How2Sign file not found: {how2sign_path}")
        return 1
    
    if not dailymoth_path.exists():
        print(f"❌ Error: DailyMoth file not found: {dailymoth_path}")
        return 1
    
    try:
        merge_datasets(
            args.how2sign_json,
            args.dailymoth_json,
            args.output_json,
            args.how2sign_ratio,
            args.dailymoth_ratio,
            args.seed,
            shuffle_final=not args.no_shuffle
        )
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
