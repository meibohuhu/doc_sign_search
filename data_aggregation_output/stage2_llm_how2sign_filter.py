#!/usr/bin/env python3
"""
Stage 2: LLM How2Sign Domain Matching
Enhanced filtering using LLM prompts to match How2Sign instructional video domain.
Replaces Stage 3 (embedding similarity) for faster processing.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import sys
import time
from typing import List, Dict, Optional

# LLM Provider Selection
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMFilter:
    """LLM-based filter for How2Sign domain matching."""
    
    def __init__(self, provider="openai", model="gpt-4o-mini", api_key=None, batch_size=20):
        """
        Initialize LLM filter.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4o-mini", "claude-3-haiku-20240307")
            api_key: API key (if None, uses environment variable)
            batch_size: Number of videos to process per batch
        """
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        
        # Setup API client
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Install with: pip install openai")
            self.client = openai.OpenAI(api_key='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' or os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies video captions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API based on provider."""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def valid_english_check(self, texts_batch: List[str]) -> List[bool]:
        """
        Check if captions are valid, fluent English suitable for training data.
        
        Args:
            texts_batch: List of video caption strings
            
        Returns:
            List of booleans (True = valid fluent English suitable for training)
        """
        prompt = f"""You are evaluating video captions for ASL (American Sign Language) translation training data, similar to the How2Sign dataset.

A caption is suitable ONLY if it meets ALL of these criteria:
1. **Fluent, grammatically correct English** - Well-structured sentences, proper grammar, natural flow
2. **Instructional/Educational content** - Tutorials, how-to guides, educational demonstrations, step-by-step instructions
3. **Similar to How2Sign** - Content that teaches skills, techniques, procedures (cooking, crafts, sports, personal care, home improvement, etc.)
4. **Clear, structured content** - Organized information that can be clearly translated to ASL
5. **Not conversational** - Avoid casual conversations, Q&A sessions, or back-and-forth dialogues
6. **Not narrative/story** - Avoid story-telling, personal anecdotes, or fictional narratives
7. **Not excessive quotes** - Avoid transcripts with many quoted conversations or dialogue

A caption is NOT suitable if it:
- Is a conversation or dialogue between people (Q&A, interviews, casual chat)
- Is a story, narrative, or personal anecdote
- Contains excessive quoted dialogue or conversations
- Is broken, unreadable, or machine-translated gibberish
- Contains mostly special characters, format markers, or timestamps
- Is mostly metadata, URLs, or non-content text
- Has excessive repetition (same phrase repeated many times)
- Is not instructional/educational in nature

Examples of SUITABLE content (like How2Sign):
- "In this clip I'm going to talk about how to buy a trophy. Most people will come into my show room and they'll ask me how much is a twenty inch trophy..."
- "Today I'm going to show you one of my grandmother's favorite things to make. This recipe is really fun and fast. I'll show you the ingredients..."
- "In the side position, we're going to be palming and thumbing the sen lines on the legs. Now make sure your client is comfortable..."

Examples of NOT SUITABLE content:
- Conversations: "Hi, how are you?" "I'm fine, thanks. What about you?" "I'm doing well..."
- Stories: "Once upon a time, there was a bird. The bird sat on her egg. The egg jumped..."
- Personal narratives: "Some of the days that are the hardest are when I start out first thing in the morning and things just don't go right..."

Be STRICT: Only mark "yes" if the caption is fluent English AND instructional/educational content suitable for ASL translation training, similar to How2Sign. Mark "no" for conversations, stories, or non-instructional content.

For each caption, output "yes" if it is suitable, or "no" if it is not.

Captions:
{chr(10).join([f"{i+1}. {text[:1000]}" for i, text in enumerate(texts_batch)])}

Output format: yes/no for each (one per line, only yes or no)"""

        try:
            response = self._call_llm(prompt)
            results = []
            for line in response.split('\n'):
                line = line.strip().lower()
                if line.startswith('yes'):
                    results.append(True)
                elif line.startswith('no'):
                    results.append(False)
                else:
                    # Default to False if unclear
                    results.append(False)
            
            # Ensure we have results for all items
            while len(results) < len(texts_batch):
                results.append(False)
            
            return results[:len(texts_batch)]
        except Exception as e:
            print(f"⚠️  Error in valid_english_check: {e}")
            # Default to all False on error
            return [False] * len(texts_batch)
    
    def basic_english_check(self, texts_batch: List[str]) -> List[bool]:
        """
        Basic English fluency check (less strict than valid_english_check).
        Only checks if captions are fluent English, without requiring instructional content.
        
        Args:
            texts_batch: List of video caption strings
            
        Returns:
            List of booleans (True = fluent English, regardless of content type)
        """
        prompt = f"""For each video caption below, determine if it is valid, fluent English.

A caption is fluent English if it:
- Is grammatically correct and readable
- Is coherent and understandable
- Has proper sentence structure
- Is natural English (not machine-translated gibberish or broken text)
- Contains meaningful content (not just noise, repeated text, or format markers)

A caption is NOT fluent English if it:
- Is completely broken or unreadable
- Is clearly machine-translated gibberish
- Contains mostly special characters, format markers, or timestamps
- Has excessive repetition (same phrase repeated many times)
- Is mostly metadata, URLs, or non-content text

Note: This check is lenient - it accepts conversations, stories, narratives, and any content type as long as it's fluent English. Only filter out clearly broken or unreadable text.

For each caption, output "yes" if it is fluent English, or "no" if it is not.

Captions:
{chr(10).join([f"{i+1}. {text[:1000]}" for i, text in enumerate(texts_batch)])}

Output format: yes/no for each (one per line, only yes or no)"""

        try:
            response = self._call_llm(prompt)
            results = []
            for line in response.split('\n'):
                line = line.strip().lower()
                if line.startswith('yes'):
                    results.append(True)
                elif line.startswith('no'):
                    results.append(False)
                else:
                    # Default to False if unclear
                    results.append(False)
            
            # Ensure we have results for all items
            while len(results) < len(texts_batch):
                results.append(False)
            
            return results[:len(texts_batch)]
        except Exception as e:
            print(f"⚠️  Error in basic_english_check: {e}")
            # Default to all False on error
            return [False] * len(texts_batch)
    



def load_how2sign_examples(how2sign_file: Path, num_examples: int = 5) -> List[str]:
    """
    Load sample How2Sign video texts for few-shot examples.
    
    Args:
        how2sign_file: Path to how2sign_video_texts.json
        num_examples: Number of examples to sample
        
    Returns:
        List of example text strings
    """
    if not how2sign_file.exists():
        print(f"⚠️  Warning: {how2sign_file} not found. Few-shot examples will be skipped.")
        return []
    
    with open(how2sign_file, 'r', encoding='utf-8') as f:
        how2sign_data = json.load(f)
    
    # Sample diverse examples (spread across the dataset)
    texts = list(how2sign_data.values())
    if len(texts) < num_examples:
        return texts
    
    # Sample evenly spaced examples
    step = len(texts) // num_examples
    examples = [texts[i * step][:500] for i in range(num_examples)]
    
    return examples


def process_batch(video_batch: List[tuple], llm_filter: LLMFilter, how2sign_examples: Optional[List[str]] = None, use_strict: bool = True) -> Dict:
    """
    Process a batch of videos through LLM filtering.
    
    Args:
        video_batch: List of (video_id, video_data) tuples
        llm_filter: LLMFilter instance
        how2sign_examples: Optional How2Sign examples (not used currently)
        use_strict: If True, use strict valid_english_check (requires instructional content).
                   If False, use lenient basic_english_check (only checks fluency).
        
    Returns:
        Dictionary with filtering results
    """
    video_ids = [vid for vid, _ in video_batch]
    # Use caption_only instead of text for more accurate content matching
    captions = [data.get('caption_only', '') or data.get('text', '') for _, data in video_batch]
    
    # Step 1: English check (strict or lenient)
    if use_strict:
        english_valid = llm_filter.valid_english_check(captions)  # Strict: requires instructional content
    else:
        english_valid = llm_filter.basic_english_check(captions)  # Lenient: only checks fluency
    
    # Step 2: How2Sign domain matching (COMMENTED OUT - only checking fluent English now)
    # valid_captions = [caption for caption, valid in zip(captions, english_valid) if valid]
    # valid_indices = [i for i, valid in enumerate(english_valid) if valid]
    # 
    # if valid_captions:
    #     domain_matches = llm_filter.how2sign_domain_match(valid_captions, how2sign_examples)
    # else:
    #     domain_matches = []
    
    # Combine results (simplified - only checking fluent English)
    results = {}
    for i, (video_id, video_data) in enumerate(video_batch):
        if english_valid[i]:
            results[video_id] = {
                'keep': True,
                'reason': 'Valid fluent English (instructional)' if use_strict else 'Valid fluent English',
                'english_valid': True,
                'domain_match': None  # Not checking domain anymore
            }
        else:
            results[video_id] = {
                'keep': False,
                'reason': 'Not valid instructional English for training' if use_strict else 'Not valid fluent English',
                'english_valid': False,
                'domain_match': None  # Not checking domain anymore
            }
    
    return results


def main():
    # Configuration
    input_file = Path("/home/mh2803/projects/sign_language_llm/data_aggregation_output/youtube_video_texts_filtered_stage1.json")
    output_dir = Path("/home/mh2803/projects/sign_language_llm/data_aggregation_output")
    output_file = output_dir / "youtube_video_texts_filtered_stage2.json"
    filtered_out_file = output_dir / "youtube_video_texts_filtered_out_stage2.json"
    stats_file = output_dir / "stage2_filtering_stats.json"
    how2sign_file = output_dir / "how2sign_video_texts.json"
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # or "claude-3-haiku-20240307"
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
    MAX_VIDEOS = int(os.getenv("MAX_VIDEOS", "0"))  # 0 = process all, N = process first N videos (for testing)
    # USE_STRICT = os.getenv("USE_STRICT", "true").lower() == "true"  # "true" = strict (requires instructional), "false" = lenient (only fluency)
    USE_STRICT=True
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        print("   Run Stage 1 first: python stage1_rule_based_filter.py")
        sys.exit(1)
    
    print("=" * 60)
    print("Stage 2: LLM Fluent English Check (Training Data Quality)")
    print("=" * 60)
    print(f"📁 Input: {input_file}")
    print(f"📁 Output (passed): {output_file}")
    print(f"📁 Output (filtered out): {filtered_out_file}")
    print(f"🤖 LLM Provider: {LLM_PROVIDER}")
    print(f"🤖 LLM Model: {LLM_MODEL}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔍 Filter Mode: {'STRICT (instructional content required)' if USE_STRICT else 'LENIENT (fluency only)'}")
    if MAX_VIDEOS > 0:
        print(f"🧪 Test Mode: Processing first {MAX_VIDEOS} videos only")
    print("=" * 60)
    
    # Load input videos
    print(f"\n📖 Loading videos from Stage 1...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_videos = json.load(f)
    
    # Limit to first N videos for testing if MAX_VIDEOS is set
    if MAX_VIDEOS > 0:
        video_items = list(all_videos.items())[:MAX_VIDEOS]
        videos = dict(video_items)
        print(f"   ⚠️  TEST MODE: Processing only first {MAX_VIDEOS} videos")
    else:
        videos = all_videos
    
    print(f"   Total videos to process: {len(videos)}")
    
    # Load How2Sign examples for few-shot learning (COMMENTED OUT - not using domain matching)
    # print(f"\n📚 Loading How2Sign examples...")
    # how2sign_examples = load_how2sign_examples(how2sign_file, num_examples=5)
    # if how2sign_examples:
    #     print(f"   Loaded {len(how2sign_examples)} examples")
    # else:
    #     print("   ⚠️  No examples loaded (will use zero-shot)")
    how2sign_examples = None  # Not using domain matching
    
    # Initialize LLM filter
    print(f"\n🤖 Initializing LLM filter...")
    try:
        llm_filter = LLMFilter(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            batch_size=BATCH_SIZE
        )
        print(f"   ✅ LLM filter initialized")
    except Exception as e:
        print(f"   ❌ Error initializing LLM filter: {e}")
        sys.exit(1)
    
    # Process videos in batches
    print(f"\n🔍 Processing videos through LLM filter...")
    video_list = list(videos.items())
    filtered_videos = {}
    filtered_out = {}
    stats = {
        'total': len(videos),
        'passed': 0,
        'filtered': 0,
        'reasons': {},
        'english_invalid': 0,
        'domain_mismatch': 0
    }
    
    num_batches = (len(video_list) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(0, len(video_list), BATCH_SIZE), desc="Processing batches", total=num_batches):
        batch = video_list[batch_idx:batch_idx + BATCH_SIZE]
        
        try:
            batch_results = process_batch(batch, llm_filter, how2sign_examples, use_strict=USE_STRICT)
            
            for video_id, result in batch_results.items():
                if result['keep']:
                    filtered_videos[video_id] = videos[video_id]
                    filtered_videos[video_id]['filtering_stages'] = {
                        'stage1': 'passed',
                        'stage2_english': 'passed' if result['english_valid'] else 'failed',
                        # 'stage2_domain': 'passed' if result['domain_match'] else 'failed'  # Commented out - not checking domain
                    }
                    stats['passed'] += 1
                else:
                    filtered_out[video_id] = {
                        'reason': result['reason'],
                        'data': videos[video_id],
                        'english_valid': result['english_valid'],
                        'domain_match': result['domain_match']
                    }
                    stats['filtered'] += 1
                    stats['reasons'][result['reason']] = stats['reasons'].get(result['reason'], 0) + 1
                    
                    if not result['english_valid']:
                        stats['english_invalid'] += 1
                    # elif not result['domain_match']:  # Commented out - not checking domain
                    #     stats['domain_mismatch'] += 1
            
            # Rate limiting: small delay between batches
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n⚠️  Error processing batch {batch_idx // BATCH_SIZE + 1}: {e}")
            # Mark all in batch as filtered
            for video_id, video_data in batch:
                filtered_out[video_id] = {
                    'reason': f'Processing error: {str(e)}',
                    'data': video_data
                }
                stats['filtered'] += 1
    
    # Save results
    print(f"\n💾 Saving filtered videos (passed)...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_videos, f, indent=2, ensure_ascii=False)
    
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
    print("✅ Stage 2 Filtering Complete!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   Total videos: {stats['total']:,}")
    print(f"   ✅ Passed (Fluent English): {stats['passed']:,} ({stats['keep_rate']:.1f}%)")
    print(f"   ❌ Filtered: {stats['filtered']:,} ({stats['filter_rate']:.1f}%)")
    print(f"\n📋 Filter reasons:")
    for reason, count in sorted(stats['reasons'].items(), key=lambda x: x[1], reverse=True):
        print(f"   - {reason}: {count:,} ({count/stats['filtered']*100:.1f}%)")
    print(f"\n📊 Breakdown:")
    print(f"   - Not valid fluent English: {stats['english_invalid']:,}")
    # print(f"   - Domain mismatch: {stats['domain_mismatch']:,}")  # Commented out - not checking domain
    print("=" * 60)
    print(f"\n📁 Output files:")
    print(f"   - Passed videos: {output_file}")
    print(f"   - Filtered out videos: {filtered_out_file}")
    print(f"   - Statistics: {stats_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()

