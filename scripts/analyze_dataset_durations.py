#!/usr/bin/env python3
"""
Analyze duration distribution for DailyMoth and How2Sign datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dailymoth_durations(tsv_file):
    """Analyze DailyMoth dataset durations from TSV file"""
    print("📊 Analyzing DailyMoth dataset...")
    
    # Read TSV file (tab-separated)
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['video_name', 'duration', 'text'])
    
    # Convert duration to float
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    
    # Remove rows with invalid durations
    df = df.dropna(subset=['duration'])
    
    print(f"   Total samples: {len(df)}")
    print(f"   Duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")
    print(f"   Average duration: {df['duration'].mean():.2f}s")
    print(f"   Median duration: {df['duration'].median():.2f}s")
    
    return df['duration'].values

def analyze_how2sign_durations(csv_file):
    """Analyze How2Sign dataset durations from CSV file"""
    print("📊 Analyzing How2Sign dataset...")
    
    # Read CSV file (tab-separated)
    df = pd.read_csv(csv_file, sep='\t')
    
    # Calculate duration from START_REALIGNED and END_REALIGNED
    df['duration'] = df['END_REALIGNED'] - df['START_REALIGNED']
    
    # Remove rows with invalid durations
    df = df.dropna(subset=['duration'])
    df = df[df['duration'] > 0]  # Remove negative or zero durations
    
    print(f"   Total samples: {len(df)}")
    print(f"   Duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")
    print(f"   Average duration: {df['duration'].mean():.2f}s")
    print(f"   Median duration: {df['duration'].median():.2f}s")
    
    return df['duration'].values

def categorize_durations(durations, dataset_name):
    """Categorize durations into bins and return statistics"""
    print(f"\n🎯 Duration Distribution for {dataset_name}:")
    print("=" * 50)
    
    # Define bins (user requested: 5s, 5-10s, 10-20s, 20-30s, >30s)
    bins = [0, 5, 10, 20, 30, float('inf')]
    labels = ['<5s', '5-10s', '10-20s', '20-30s', '>30s']
    
    # Categorize
    categories = pd.cut(durations, bins=bins, labels=labels, right=False)
    category_counts = categories.value_counts().sort_index()
    
    # Calculate percentages
    total = len(durations)
    percentages = (category_counts / total * 100).round(1)
    
    # Print results
    for label in labels:
        count = category_counts.get(label, 0)
        pct = percentages.get(label, 0)
        print(f"   {label:>6}: {count:>6} samples ({pct:>5.1f}%)")
    
    print(f"   {'Total':>6}: {total:>6} samples (100.0%)")
    
    # Additional statistics
    print(f"\n📈 Additional Statistics:")
    print(f"   Average: {np.mean(durations):.2f}s")
    print(f"   Median:  {np.median(durations):.2f}s")
    print(f"   Std Dev: {np.std(durations):.2f}s")
    print(f"   Min:     {np.min(durations):.2f}s")
    print(f"   Max:     {np.max(durations):.2f}s")
    
    return category_counts, percentages

def main():
    # File paths
    dailymoth_file = "/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/manifests/train.tsv"
    how2sign_file = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/how2sign_realigned_train.csv"
    
    print("🔍 Dataset Duration Analysis")
    print("=" * 60)
    
    # Analyze DailyMoth
    try:
        dailymoth_durations = analyze_dailymoth_durations(dailymoth_file)
        dailymoth_counts, dailymoth_pcts = categorize_durations(dailymoth_durations, "DailyMoth")
    except Exception as e:
        print(f"❌ Error analyzing DailyMoth: {e}")
        dailymoth_durations = None
    
    print("\n" + "=" * 60)
    
    # Analyze How2Sign
    try:
        how2sign_durations = analyze_how2sign_durations(how2sign_file)
        how2sign_counts, how2sign_pcts = categorize_durations(how2sign_durations, "How2Sign")
    except Exception as e:
        print(f"❌ Error analyzing How2Sign: {e}")
        how2sign_durations = None
    
    # Comparison
    if dailymoth_durations is not None and how2sign_durations is not None:
        print("\n" + "=" * 60)
        print("🔄 COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"{'Metric':<20} {'DailyMoth':<15} {'How2Sign':<15}")
        print("-" * 50)
        print(f"{'Total Samples':<20} {len(dailymoth_durations):<15} {len(how2sign_durations):<15}")
        print(f"{'Average Duration':<20} {np.mean(dailymoth_durations):<15.2f} {np.mean(how2sign_durations):<15.2f}")
        print(f"{'Median Duration':<20} {np.median(dailymoth_durations):<15.2f} {np.median(how2sign_durations):<15.2f}")
        print(f"{'Min Duration':<20} {np.min(dailymoth_durations):<15.2f} {np.min(how2sign_durations):<15.2f}")
        print(f"{'Max Duration':<20} {np.max(dailymoth_durations):<15.2f} {np.max(how2sign_durations):<15.2f}")
        
        print(f"\n{'Duration Range':<20} {'DailyMoth %':<15} {'How2Sign %':<15}")
        print("-" * 50)
        for label in ['<5s', '5-10s', '10-20s', '20-30s', '>30s']:
            dailymoth_pct = dailymoth_pcts.get(label, 0)
            how2sign_pct = how2sign_pcts.get(label, 0)
            print(f"{label:<20} {dailymoth_pct:<15.1f} {how2sign_pct:<15.1f}")

if __name__ == "__main__":
    main()
