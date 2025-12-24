#!/usr/bin/env python3
"""
Stage 1: Rule-Based Filtering
Quick filtering to remove obviously noisy content (30-40% expected).
"""

import json
import re
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import sys

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  Warning: langdetect not available. Language detection will be skipped.")


def count_special_chars(text):
    """Count special characters (non-alphanumeric, non-space)."""
    return len(re.findall(r'[^a-zA-Z0-9\s]', text))


def has_excessive_repetition(text, min_repeat_length=10, max_repeats=3):
    """
    Check for excessive repetition in text.
    
    Args:
        text: Input text
        min_repeat_length: Minimum length of repeated substring
        max_repeats: Maximum allowed repeats before flagging
    
    Returns:
        True if excessive repetition detected
    """
    words = text.split()
    if len(words) < min_repeat_length * 2:
        return False
    
    # Check for repeated phrases
    for i in range(len(words) - min_repeat_length):
        phrase = ' '.join(words[i:i+min_repeat_length])
        # Count how many times this phrase appears
        count = text.count(phrase)
        if count > max_repeats:
            return True
    
    # Check for character-level repetition (e.g., "aaaaaa")
    char_pattern = r'(.)\1{20,}'  # Same character repeated 20+ times
    if re.search(char_pattern, text):
        return True
    
    return False


def is_english(text, min_length=50):
    """
    Detect if text is English.
    
    Args:
        text: Input text
        min_length: Minimum text length for reliable detection
    
    Returns:
        True if English, False otherwise
    """
    if not LANGDETECT_AVAILABLE:
        # Fallback: simple heuristic based on common English words
        common_english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i']
        text_lower = text.lower()
        word_count = sum(1 for word in common_english_words if word in text_lower)
        return word_count >= 3  # At least 3 common English words
    
    if len(text) < min_length:
        # Too short for reliable detection, assume English
        return True
    
    try:
        detected_lang = detect(text)
        return detected_lang == 'en'
    except LangDetectException:
        # If detection fails, use fallback heuristic
        common_english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i']
        text_lower = text.lower()
        word_count = sum(1 for word in common_english_words if word in text_lower)
        return word_count >= 3


def rule_based_filter(video_data):
    """
    Apply rule-based filtering to a video.
    
    Args:
        video_data: Dictionary with 'text', 'caption_only', etc.
    
    Returns:
        (keep: bool, reason: str)
    """
    text = video_data.get('text', '')
    caption = video_data.get('caption_only', '')
    
    # Handle empty text
    if not text or not text.strip():
        return False, "Empty text"
    
    # Get metadata once
    metadata = video_data.get('metadata', {})
    title = video_data.get('title', '').lower()
    channel = metadata.get('channel', '').lower()
    duration = metadata.get('duration', 0)
    
    # 1. Length checks
    caption_words = len(caption.split()) if caption else 0
    if caption_words < 20:  # Too short
        return False, f"Too short (caption: {caption_words} words)"
    
    # 1.5. Check caption density relative to video duration
    # Normal videos should have at least 5-10 words per minute of video
    # If duration is available and > 60 seconds, check caption density
    if duration > 60:  # Only check for videos longer than 1 minute
        duration_minutes = duration / 60.0
        words_per_minute = caption_words / duration_minutes if duration_minutes > 0 else 0
        
        # Filter out videos with very low caption density (< 5 words per minute)
        # This catches cases where caption is too short for the video length
        if words_per_minute < 5:
            return False, f"Low caption density ({words_per_minute:.1f} words/min, {caption_words} words for {duration_minutes:.1f} min)"
    
    # Check video duration (30 minutes = 1800 seconds)
    if duration > 1800:  # More than 30 minutes
        duration_min = duration / 60
        return False, f"Too long (duration: {duration_min:.1f} min)"
    
    # 2. Language detection
    if not is_english(text):
        return False, "Not English"
    
    # 3. Character quality checks
    if len(text) > 0:
        special_char_ratio = count_special_chars(text) / len(text)
        if special_char_ratio > 0.3:  # Too many special characters
            return False, f"Too many special chars (ratio: {special_char_ratio:.2f})"
    
    # 4. Check for problematic captions (format markers)
    # If caption_only contains format markers like "start position:0%", filter it out
    if caption:
        caption_lower = caption.lower()
        if 'start position:' in caption_lower or 'align:start' in caption_lower or 'position:0%' in caption_lower:
            return False, "Problematic caption (contains format markers)"
        
        # Check for excessive special markers like [demonstrates name sign], [shows], etc.
        # Count brackets with descriptive text inside (e.g., [demonstrates name sign], [shows something])
        bracket_markers = len(re.findall(r'\[[^\]]+\]', caption))
        # If caption has bracket markers, check if they're excessive
        if caption_words > 10 and bracket_markers >= 2:
            bracket_ratio = bracket_markers / caption_words
            if bracket_ratio > 0.05:  # More than 5% of content is markers
                return False, f"Too many special markers ({bracket_markers} markers in {caption_words} words, ratio: {bracket_ratio:.2f})"
        
        # Check for missing punctuation in long captions
        # If caption is long (>50 words) but has very few sentence-ending punctuation marks
        if caption_words > 50:
            # Count sentence-ending punctuation
            sentence_endings = len(re.findall(r'[.!?]', caption))
            # Normal text should have roughly 1 sentence ending per 15-20 words
            expected_sentences = caption_words / 20
            # Require at least 20% of expected punctuation, or at least 1 if caption is short
            min_required = max(1, int(expected_sentences * 0.2))
            if sentence_endings < min_required:
                return False, f"Missing punctuation (only {sentence_endings} sentence endings in {caption_words} words, expected at least {min_required})"
        
        # Check for escaped quotes (e.g., \"Oh oh!\") - these are JSON formatting artifacts
        # Normal captions should use regular quotes, not escaped quotes
        if '\\"' in caption or '\\\'' in caption:
            # Count escaped quotes
            escaped_quotes = caption.count('\\"') + caption.count('\\\'')
            # If there are many escaped quotes, it's likely a formatting issue
            if escaped_quotes > 2:
                return False, f"Contains escaped quotes (JSON formatting artifacts: {escaped_quotes} escaped quotes)"
        
        # Check for excessive URLs (more than 2 "www" occurrences)
        # Captions with many URLs are likely not pure content but include metadata/links
        www_count = caption.lower().count('www.')
        if www_count > 2:
            return False, f"Too many URLs ({www_count} 'www' occurrences)"
        
        # Check for excessive conversation/quotes (too many quotation marks)
        # Captions with many quotes are often transcripts of conversations rather than instructional content
        quote_chars = caption.count('"') + caption.count("'") + caption.count('"') + caption.count('"')
        if caption_words > 30 and quote_chars > 8:  # More than 8 quote characters in longer captions
            quote_ratio = quote_chars / caption_words
            if quote_ratio > 0.15:  # More than 15% quote characters
                return False, f"Too many quotes/conversation ({quote_chars} quote chars in {caption_words} words, ratio: {quote_ratio:.2f})"
        
        # Check for mostly uppercase words (like vocabulary lists or sign glosses)
        # Count words that are mostly uppercase (excluding single characters and numbers)
        words = caption.split()
        if len(words) > 10:  # Only check for longer captions
            uppercase_words = 0
            for w in words:
                # Remove punctuation and check if the alphabetic part is uppercase
                # This catches words like "TODAY,", "SIGNS:", "SHOW-you", etc.
                w_clean = re.sub(r'[^\w]', '', w)  # Remove non-word characters
                if len(w_clean) > 1:  # At least 2 characters
                    # Check if it's all uppercase letters (may contain numbers/hyphens)
                    if w_clean.isupper() and any(c.isalpha() for c in w_clean):
                        uppercase_words += 1
            
            uppercase_ratio = uppercase_words / len(words) if len(words) > 0 else 0
            # If more than 50% of words are uppercase, it's likely a vocabulary list or sign gloss
            if uppercase_ratio > 0.5:
                return False, f"Mostly uppercase words ({uppercase_words}/{len(words)} words are uppercase, ratio: {uppercase_ratio:.2f})"
    
    # 5. Keyword blacklist (music, news, etc.)
    
    # Filter out Sign1News channel
    if channel == 'sign1news':
        return False, "Sign1News channel"
    
    # Other keywords: check in full text
    blacklist_keywords = [
        "music video", "mv", "song",
        "commercial", "advertisement", "trailer"
    ]
    text_lower = text.lower()
    for kw in blacklist_keywords:
        if kw in text_lower:
            return False, f"Blacklisted keyword: {kw}"
    
    # 6. Repetition check
    if has_excessive_repetition(text):
        return False, "Excessive repetition"
    
    return True, "Passed"


def main():
    # Input and output paths
    input_file = Path("/home/mh2803/projects/sign_language_llm/data_aggregation_output/youtube_video_texts.json")
    output_dir = Path("/home/mh2803/projects/sign_language_llm/data_aggregation_output")
    output_file = output_dir / "youtube_video_texts_filtered_stage1.json"
    filtered_out_file = output_dir / "youtube_video_texts_filtered_out_stage1.json"
    stats_file = output_dir / "stage1_filtering_stats.json"
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        sys.exit(1)
    
    print("=" * 60)
    print("Stage 1: Rule-Based Filtering")
    print("=" * 60)
    print(f"📁 Input: {input_file}")
    print(f"📁 Output (passed): {output_file}")
    print(f"📁 Output (filtered out): {filtered_out_file}")
    print("=" * 60)
    
    # Load YouTube videos
    print(f"\n📖 Loading YouTube videos...")
    with open(input_file, 'r', encoding='utf-8') as f:
        youtube_videos = json.load(f)
    
    print(f"   Total videos: {len(youtube_videos)}")
    
    # Apply filtering
    print(f"\n🔍 Applying rule-based filters...")
    filtered_videos = {}
    filtered_out = {}
    stats = {
        'total': len(youtube_videos),
        'passed': 0,
        'filtered': 0,
        'reasons': {}
    }
    
    for video_id, video_data in tqdm(youtube_videos.items(), desc="Filtering"):
        keep, reason = rule_based_filter(video_data)
        
        if keep:
            filtered_videos[video_id] = video_data
            stats['passed'] += 1
        else:
            filtered_out[video_id] = {
                'reason': reason,
                'data': video_data
            }
            stats['filtered'] += 1
            stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
    
    # Save filtered videos (passed)
    print(f"\n💾 Saving filtered videos (passed)...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_videos, f, indent=2, ensure_ascii=False)
    
    # Save filtered out videos
    print(f"💾 Saving filtered out videos...")
    with open(filtered_out_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_out, f, indent=2, ensure_ascii=False)
    
    # Save statistics
    stats['filter_rate'] = stats['filtered'] / stats['total'] * 100
    stats['keep_rate'] = stats['passed'] / stats['total'] * 100
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Stage 1 Filtering Complete!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   Total videos: {stats['total']:,}")
    print(f"   ✅ Passed: {stats['passed']:,} ({stats['keep_rate']:.1f}%)")
    print(f"   ❌ Filtered: {stats['filtered']:,} ({stats['filter_rate']:.1f}%)")
    print(f"\n📋 Filter reasons:")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: x[1], reverse=True):
        print(f"   - {reason}: {count:,} ({count/stats['filtered']*100:.1f}%)")
    print("=" * 60)
    print(f"\n📁 Output files:")
    print(f"   - Passed videos: {output_file}")
    print(f"   - Filtered out videos: {filtered_out_file}")
    print(f"   - Statistics: {stats_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()

