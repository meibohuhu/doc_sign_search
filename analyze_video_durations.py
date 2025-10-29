#!/usr/bin/env python3
"""
Analyze How2Sign video duration statistics from CSV file
Using only built-in Python libraries
"""

import csv
import statistics

def analyze_video_durations(csv_path):
    """Analyze video duration statistics from CSV file"""
    
    print("📊 Loading CSV data...")
    
    durations = []
    total_rows = 0
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        for row in reader:
            total_rows += 1
            try:
                start = float(row['START_REALIGNED'])
                end = float(row['END_REALIGNED'])
                duration = end - start
                
                if duration > 0:  # Only include valid durations
                    durations.append(duration)
            except (ValueError, KeyError) as e:
                continue
    
    print(f"📋 Total rows processed: {total_rows}")
    print(f"📋 Valid video clips: {len(durations)}")
    
    if not durations:
        print("❌ No valid durations found!")
        return
    
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
    
    print("\n🎬 VIDEO DURATION ANALYSIS")
    print("=" * 50)
    print(f"📈 Mean duration: {mean_duration:.2f} seconds")
    print(f"📈 Standard deviation: {std_duration:.2f} seconds")
    print(f"📈 Median duration: {median_duration:.2f} seconds")
    print(f"📈 Min duration: {min_duration:.2f} seconds")
    print(f"📈 Max duration: {max_duration:.2f} seconds")
    print(f"📈 90th percentile: {p90_duration:.2f} seconds")
    print(f"📈 95th percentile: {p95_duration:.2f} seconds")
    print(f"📈 99th percentile: {p99_duration:.2f} seconds")
    
    # Duration categories
    less_than_10s = sum(1 for d in durations if d < 10)
    between_10_30s = sum(1 for d in durations if 10 <= d <= 30)
    over_30s = sum(1 for d in durations if d > 30)
    total_clips = len(durations)
    
    print(f"\n📊 DURATION DISTRIBUTION")
    print("=" * 30)
    print(f"🎬 Less than 10s: {less_than_10s:,} clips ({less_than_10s/total_clips*100:.1f}%)")
    print(f"🎬 10s - 30s: {between_10_30s:,} clips ({between_10_30s/total_clips*100:.1f}%)")
    print(f"🎬 Over 30s: {over_30s:,} clips ({over_30s/total_clips*100:.1f}%)")
    print(f"🎬 Total clips: {total_clips:,}")
    
    # Additional breakdowns
    print(f"\n📊 ADDITIONAL BREAKDOWN")
    print("=" * 30)
    
    # Very short clips (< 5s)
    very_short = sum(1 for d in durations if d < 5)
    print(f"🎬 Very short (< 5s): {very_short:,} clips ({very_short/total_clips*100:.1f}%)")
    
    # Short clips (5-10s)
    short = sum(1 for d in durations if 5 <= d < 10)
    print(f"🎬 Short (5-10s): {short:,} clips ({short/total_clips*100:.1f}%)")
    
    # Medium clips (10-20s)
    medium = sum(1 for d in durations if 10 <= d < 20)
    print(f"🎬 Medium (10-20s): {medium:,} clips ({medium/total_clips*100:.1f}%)")
    
    # Long clips (20-30s)
    long_clips = sum(1 for d in durations if 20 <= d <= 30)
    print(f"🎬 Long (20-30s): {long_clips:,} clips ({long_clips/total_clips*100:.1f}%)")
    
    # Very long clips (> 30s)
    very_long = sum(1 for d in durations if d > 30)
    print(f"🎬 Very long (> 30s): {very_long:,} clips ({very_long/total_clips*100:.1f}%)")
    
    # Extreme cases
    over_60s = sum(1 for d in durations if d > 60)
    print(f"🎬 Over 60s: {over_60s:,} clips ({over_60s/total_clips*100:.1f}%)")
    
    return {
        'mean': mean_duration,
        'std': std_duration,
        'median': median_duration,
        'p90': p90_duration,
        'p95': p95_duration,
        'p99': p99_duration,
        'min': min_duration,
        'max': max_duration,
        'less_than_10s': less_than_10s,
        'between_10_30s': between_10_30s,
        'over_30s': over_30s,
        'total': total_clips
    }

if __name__ == "__main__":
    csv_path = "/home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/how2sign_realigned_train.csv"
    stats = analyze_video_durations(csv_path)