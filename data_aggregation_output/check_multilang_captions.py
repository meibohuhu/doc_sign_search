#!/usr/bin/env python3
"""
Check captions for videos with multiple caption languages in sign1news_videos_metadata.json
"""

import json
from pathlib import Path
from collections import defaultdict

def check_caption_issues(caption_timestamps_str):
    """
    Check for issues in caption timestamps.
    Returns a list of issue descriptions.
    """
    issues = []
    
    if not caption_timestamps_str or not caption_timestamps_str.strip():
        issues.append("Empty caption_timestamps")
        return issues
    
    try:
        captions = json.loads(caption_timestamps_str)
    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON: {str(e)}")
        return issues
    
    if not isinstance(captions, list):
        issues.append(f"Not a list: {type(captions)}")
        return issues
    
    if len(captions) == 0:
        issues.append("Empty captions list")
        return issues
    
    # Check for format markers (like "align:start position:0%")
    has_format_markers = False
    empty_text_count = 0
    duplicate_text_count = 0
    text_samples = []
    
    for i, caption in enumerate(captions):
        if not isinstance(caption, dict):
            issues.append(f"Caption {i} is not a dict")
            continue
        
        text = caption.get('text', '')
        start = caption.get('start', 0)
        end = caption.get('end', 0)
        
        # Check for format markers
        if 'align:start' in text.lower() or 'position:' in text.lower():
            has_format_markers = True
        
        # Check for empty or very short text
        if not text or not text.strip():
            empty_text_count += 1
        
        # Collect text samples
        if text and text.strip():
            text_samples.append(text.strip()[:50])  # First 50 chars
    
    if has_format_markers:
        issues.append("Contains format markers (align:start position:0%)")
    
    if empty_text_count > 0:
        issues.append(f"{empty_text_count} empty text entries")
    
    # Check for excessive duplicates
    if len(text_samples) > 0:
        unique_texts = len(set(text_samples))
        if unique_texts / len(text_samples) < 0.3:  # Less than 30% unique
            issues.append(f"High text duplication ({unique_texts}/{len(text_samples)} unique)")
    
    # Check for suspicious patterns
    if len(captions) > 0:
        # Check if all texts are very similar (potential auto-translation artifacts)
        sample_texts = [c.get('text', '').strip()[:20] for c in captions[:10] if c.get('text', '').strip()]
        if len(set(sample_texts)) == 1 and len(sample_texts) > 1:
            issues.append("All caption texts are identical")
    
    return issues

def main():
    input_file = Path("/home/mh2803/projects/sign_language_llm/data_aggregation_output/sign1news_videos_metadata.json")
    output_file = Path("/home/mh2803/projects/sign_language_llm/data_aggregation_output/multilang_caption_issues.json")
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        return
    
    print("=" * 60)
    print("Checking captions for videos with multiple languages")
    print("=" * 60)
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print("=" * 60)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    
    # Find videos with multiple languages
    multilang_videos = {}
    single_lang_count = 0
    
    for video_id, video_data in videos.items():
        caption_languages = video_data.get('caption_languages', '')
        
        # Check if multiple languages (contains comma)
        if ',' in caption_languages:
            multilang_videos[video_id] = video_data
        else:
            single_lang_count += 1
    
    print(f"\n📊 Statistics:")
    print(f"   Total videos: {len(videos)}")
    print(f"   Single language: {single_lang_count}")
    print(f"   Multiple languages: {len(multilang_videos)}")
    
    # Check captions for multilang videos
    print(f"\n🔍 Checking captions for {len(multilang_videos)} multilang videos...")
    
    issues_report = {
        'total_multilang': len(multilang_videos),
        'videos_with_issues': {},
        'issue_summary': defaultdict(int),
        'videos_ok': []
    }
    
    for video_id, video_data in multilang_videos.items():
        caption_timestamps = video_data.get('caption_timestamps', '')
        issues = check_caption_issues(caption_timestamps)
        
        if issues:
            issues_report['videos_with_issues'][video_id] = {
                'title': video_data.get('title', ''),
                'caption_languages': video_data.get('caption_languages', ''),
                'issues': issues
            }
            for issue in issues:
                issues_report['issue_summary'][issue] += 1
        else:
            issues_report['videos_ok'].append(video_id)
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(issues_report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)
    print(f"📊 Results:")
    print(f"   Videos with issues: {len(issues_report['videos_with_issues'])}")
    print(f"   Videos OK: {len(issues_report['videos_ok'])}")
    
    if issues_report['issue_summary']:
        print(f"\n📋 Issue Summary:")
        for issue, count in sorted(issues_report['issue_summary'].items(), key=lambda x: x[1], reverse=True):
            print(f"   - {issue}: {count} videos")
    
    # Show sample problematic videos
    if issues_report['videos_with_issues']:
        print(f"\n⚠️  Sample problematic videos (first 5):")
        for i, (video_id, info) in enumerate(list(issues_report['videos_with_issues'].items())[:5]):
            print(f"\n   {i+1}. {video_id}")
            print(f"      Title: {info['title'][:60]}...")
            print(f"      Languages: {info['caption_languages'][:80]}...")
            print(f"      Issues: {', '.join(info['issues'])}")
    
    print("=" * 60)
    print(f"\n📁 Full report saved to: {output_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()

