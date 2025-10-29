#!/usr/bin/env python3
"""
Compare video duration statistics between How2Sign and DailyMoth datasets
Using the exact categories requested: <5s, 5-10s, 10-20s, 20-30s, >30s
"""

import csv
import statistics

def analyze_dataset_durations(file_path, dataset_name, file_type='tsv'):
    """Analyze video duration statistics from TSV file"""
    
    print(f"📊 Loading {dataset_name} data...")
    
    durations = []
    total_rows = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        if dataset_name == "How2Sign":
            # How2Sign is TSV with START_REALIGNED and END_REALIGNED columns
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                total_rows += 1
                try:
                    start = float(row['START_REALIGNED'])
                    end = float(row['END_REALIGNED'])
                    duration = end - start
                    if duration > 0:
                        durations.append(duration)
                except (ValueError, KeyError):
                    continue
        else:  # DailyMoth TSV
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                total_rows += 1
                try:
                    duration = float(row[1])  # Duration is in second column
                    if duration > 0:
                        durations.append(duration)
                except (ValueError, IndexError):
                    continue
    
    print(f"📋 Total rows processed: {total_rows}")
    print(f"📋 Valid video clips: {len(durations)}")
    
    if not durations:
        print(f"❌ No valid durations found in {dataset_name}!")
        return None
    
    # Calculate statistics
    mean_duration = statistics.mean(durations)
    std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
    median_duration = statistics.median(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    # Calculate percentiles manually
    sorted_durations = sorted(durations)
    n = len(sorted_durations)
    p90_duration = sorted_durations[int(0.9 * n)]
    p95_duration = sorted_durations[int(0.95 * n)]
    p99_duration = sorted_durations[int(0.99 * n)]
    
    # Duration categories as requested
    less_than_5s = sum(1 for d in durations if d < 5)
    between_5_10s = sum(1 for d in durations if 5 <= d < 10)
    between_10_20s = sum(1 for d in durations if 10 <= d < 20)
    between_20_30s = sum(1 for d in durations if 20 <= d <= 30)
    over_30s = sum(1 for d in durations if d > 30)
    total_clips = len(durations)
    
    return {
        'dataset_name': dataset_name,
        'total_clips': total_clips,
        'mean': mean_duration,
        'std': std_duration,
        'median': median_duration,
        'min': min_duration,
        'max': max_duration,
        'p90': p90_duration,
        'p95': p95_duration,
        'p99': p99_duration,
        'less_than_5s': less_than_5s,
        'between_5_10s': between_5_10s,
        'between_10_20s': between_10_20s,
        'between_20_30s': between_20_30s,
        'over_30s': over_30s
    }

def print_comparison_table(stats1, stats2):
    """Print a comparison table between two datasets"""
    
    print(f"\n{'='*80}")
    print(f"📊 DATASET COMPARISON: {stats1['dataset_name']} vs {stats2['dataset_name']}")
    print(f"{'='*80}")
    
    # Basic statistics comparison
    print(f"\n📈 BASIC STATISTICS")
    print(f"{'Metric':<20} {'How2Sign':<15} {'DailyMoth':<15} {'Difference':<15}")
    print(f"{'-'*65}")
    print(f"{'Total clips':<20} {stats1['total_clips']:,<15} {stats2['total_clips']:,<15} {stats1['total_clips']-stats2['total_clips']:,<15}")
    print(f"{'Mean (s)':<20} {stats1['mean']:<15.2f} {stats2['mean']:<15.2f} {stats1['mean']-stats2['mean']:<15.2f}")
    print(f"{'Std Dev (s)':<20} {stats1['std']:<15.2f} {stats2['std']:<15.2f} {stats1['std']-stats2['std']:<15.2f}")
    print(f"{'Median (s)':<20} {stats1['median']:<15.2f} {stats2['median']:<15.2f} {stats1['median']-stats2['median']:<15.2f}")
    print(f"{'90th %ile (s)':<20} {stats1['p90']:<15.2f} {stats2['p90']:<15.2f} {stats1['p90']-stats2['p90']:<15.2f}")
    print(f"{'Min (s)':<20} {stats1['min']:<15.2f} {stats2['min']:<15.2f} {stats1['min']-stats2['min']:<15.2f}")
    print(f"{'Max (s)':<20} {stats1['max']:<15.2f} {stats2['max']:<15.2f} {stats1['max']-stats2['max']:<15.2f}")
    
    # Duration distribution comparison
    print(f"\n📊 DURATION DISTRIBUTION COMPARISON")
    print(f"{'Category':<15} {'How2Sign':<20} {'DailyMoth':<20} {'Difference':<15}")
    print(f"{'-'*70}")
    
    categories = [
        ('< 5s', 'less_than_5s'),
        ('5-10s', 'between_5_10s'),
        ('10-20s', 'between_10_20s'),
        ('20-30s', 'between_20_30s'),
        ('> 30s', 'over_30s')
    ]
    
    for cat_name, cat_key in categories:
        count1 = stats1[cat_key]
        count2 = stats2[cat_key]
        pct1 = count1 / stats1['total_clips'] * 100
        pct2 = count2 / stats2['total_clips'] * 100
        
        print(f"{cat_name:<15} {count1:,} ({pct1:.1f}%){'':<8} {count2:,} ({pct2:.1f}%){'':<8} {count1-count2:,} ({pct1-pct2:+.1f}%)")
    
    # Summary insights
    print(f"\n🎯 KEY INSIGHTS")
    print(f"{'-'*50}")
    
    if stats1['std'] > stats2['std']:
        print(f"• {stats1['dataset_name']} has more variable durations (std: {stats1['std']:.2f}s vs {stats2['std']:.2f}s)")
    else:
        print(f"• {stats2['dataset_name']} has more variable durations (std: {stats2['std']:.2f}s vs {stats1['std']:.2f}s)")
    
    if stats1['max'] > stats2['max']:
        print(f"• {stats1['dataset_name']} has longer maximum duration ({stats1['max']:.1f}s vs {stats2['max']:.1f}s)")
    else:
        print(f"• {stats2['dataset_name']} has longer maximum duration ({stats2['max']:.1f}s vs {stats1['max']:.1f}s)")
    
    # Most common category for each dataset
    cat_counts1 = [stats1['less_than_5s'], stats1['between_5_10s'], stats1['between_10_20s'], stats1['between_20_30s'], stats1['over_30s']]
    cat_counts2 = [stats2['less_than_5s'], stats2['between_5_10s'], stats2['between_10_20s'], stats2['between_20_30s'], stats2['over_30s']]
    
    cat_names = ['< 5s', '5-10s', '10-20s', '20-30s', '> 30s']
    max_cat1 = cat_names[cat_counts1.index(max(cat_counts1))]
    max_cat2 = cat_names[cat_counts2.index(max(cat_counts2))]
    
    print(f"• Most common category in {stats1['dataset_name']}: {max_cat1}")
    print(f"• Most common category in {stats2['dataset_name']}: {max_cat2}")

def main():
    # File paths
    how2sign_csv = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/how2sign_realigned_train.csv"
    dailymoth_tsv = "/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/manifests/train.tsv"
    
    # Analyze both datasets
    print("🔍 ANALYZING BOTH DATASETS")
    print("="*50)
    
    how2sign_stats = analyze_dataset_durations(how2sign_csv, "How2Sign", 'tsv')
    print()
    dailymoth_stats = analyze_dataset_durations(dailymoth_tsv, "DailyMoth", 'tsv')
    
    if how2sign_stats and dailymoth_stats:
        print_comparison_table(how2sign_stats, dailymoth_stats)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
