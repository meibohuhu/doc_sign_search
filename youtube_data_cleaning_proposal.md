# YouTube ASL Data Cleaning Proposal
## Caption-based Filtering Pipeline for How2Sign Domain Alignment

---

## Overview

**Goal**: Filter 9K YouTube ASL videos to match How2Sign's instructional video domain, reducing noise by 50-70% before video processing.

**Strategy**: Multi-stage caption-level filtering using rule-based filters, LLM classification, and semantic embedding similarity.

---

## PART 1: Data Aggregation (Video-Level)

### 1.1 How2Sign Video Text Aggregation

**Input**: `/local1/mhu/sign_language_llm/how2sign/video/train_videos/how2sign_realigned_train.csv`
- ~2,192 unique videos
- ~31K sentence segments

**Process**:
```python
# Group sentences by VIDEO_ID
how2sign_videos = {}
for row in how2sign_data:
    video_id = row['VIDEO_ID']
    sentence = row['SENTENCE']
    if video_id not in how2sign_videos:
        how2sign_videos[video_id] = []
    how2sign_videos[video_id].append(sentence)

# Combine into full video text
how2sign_full_texts = {
    video_id: " ".join(sentences) 
    for video_id, sentences in how2sign_videos.items()
}
```

**Output**: Dictionary mapping `video_id → full_text` (2,192 videos)

---

### 1.2 YouTube Video Text Aggregation

**Input**: `/local1/mhu/sign_language_llm/youtube-asl_metadata.csv`
- ~43K videos
- Captions stored as JSON in `caption_timestamps` column

**Process**:
```python
# Parse and combine captions
youtube_videos = {}
for row in youtube_data:
    video_id = row['video_id']
    
    # Parse caption_timestamps JSON
    caption_json = json.loads(row['caption_timestamps'])
    caption_text = " ".join([seg['text'] for seg in caption_json])
    
    # Combine: title + description + captions (Suggestion 1)
    full_text = f"{row['title']} {row['description']} {caption_text}"
    
    youtube_videos[video_id] = {
        'text': full_text,
        'title': row['title'],
        'description': row['description'],
        'caption_only': caption_text,
        'metadata': {...}  # channel, duration, etc.
    }
```

**Output**: Dictionary mapping `video_id → {text, title, description, ...}` (~43K videos)

---

## PART 2: How2Sign Topic Classification

### 2.1 Topic Categories (10 How2Sign Topics)

```
1. Cars and Other Vehicle
2. Games
3. Arts and Entertainment
4. Personal Care and Style
5. Food and Drinks
6. Education and Communication
7. Home and Garden
8. Pets and Animals
9. Hobbies and Crafts
10. Sports and Fitness
```

### 2.2 Topic Classification Method

**Option A: LLM-based Classification (Recommended)**
```python
# Prompt for each How2Sign video
prompt = f"""
Classify this instructional video content into one or more of these 10 topics:
1. Cars and Other Vehicle
2. Games
3. Arts and Entertainment
4. Personal Care and Style
5. Food and Drinks
6. Education and Communication
7. Home and Garden
8. Pets and Animals
9. Hobbies and Crafts
10. Sports and Fitness

Video content: {video_text}

Output format: List topic numbers (e.g., [5, 7] for Food and Home topics)
"""

# Batch process for efficiency (20-50 videos per batch)
```

**Option B: Keyword-based Classification (Faster, less accurate)**
```python
topic_keywords = {
    "Food and Drinks": ["cook", "recipe", "food", "ingredient", "kitchen", ...],
    "Home and Garden": ["home", "garden", "plant", "decorate", "furniture", ...],
    # ... define keywords for each topic
}
```

**Output**: Dictionary mapping `video_id → [topic_list]` (multi-label allowed)


---

## PART 4: Multi-Stage Filtering Pipeline (Suggestion 2)

### Stage 1: Quick Rule-Based Filtering (Filter 30-40%)

**Purpose**: Remove obviously noisy content quickly and cheaply.

```python
def rule_based_filter(video_data):
    text = video_data['text']
    caption = video_data['caption_only']
    
    # 1. Length checks
    if len(caption.split()) < 20:  # Too short
        return False, "Too short"
    if len(text) > 5000:  # Too long (might be concatenated noise)
        return False, "Too long"
    
    # 2. Language detection
    if not is_english(text):
        return False, "Not English"
    
    # 3. Character quality checks
    special_char_ratio = count_special_chars(text) / len(text)
    if special_char_ratio > 0.3:  # Too many special characters
        return False, "Too many special chars"
    
    # 4. Keyword blacklist (music, news, etc.)
    blacklist_keywords = [
        "music video", "mv", "song", "news", "breaking news",
        "commercial", "advertisement", "trailer", "movie"
    ]
    text_lower = text.lower()
    if any(kw in text_lower for kw in blacklist_keywords):
        return False, "Blacklisted keyword"
    
    # 5. Repetition check
    if has_excessive_repetition(text):
        return False, "Excessive repetition"
    
    return True, "Passed"
```

**Expected**: Filter out 30-40% of videos

---

### Stage 2: LLM Semantic Classification (Filter 30-50%)

**Purpose**: Use LLM to filter invalid English and irrelevant content, directly matching How2Sign domain.

#### Task 1: Valid English Check
```python
def llm_valid_english_check(texts_batch):
    prompt = f"""
    For each caption, determine if it is valid, understandable English.
    Output only "yes" or "no" for each caption.
    
    Captions:
    {chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts_batch)])}
    
    Output format: yes/no for each (one per line)
    """
    
    results = llm.generate(prompt)
    return [r.strip().lower() == "yes" for r in results.split('\n')]
```

#### Task 2: How2Sign Domain Matching (Enhanced with Few-Shot Examples)
```python
def llm_how2sign_domain_match(texts_batch, how2sign_examples=None):
    """
    Classify YouTube videos based on similarity to How2Sign instructional videos.
    Uses few-shot examples from How2Sign dataset for better accuracy.
    """
    # Load How2Sign examples if not provided
    if how2sign_examples is None:
        with open('how2sign_video_texts.json', 'r') as f:
            how2sign_data = json.load(f)
        # Sample diverse examples (one per topic category)
        how2sign_examples = [
            list(how2sign_data.values())[i][:300]  # First 300 chars
            for i in [0, 100, 500, 1000, 1500]  # Diverse samples
        ]
    
    # Build few-shot examples section
    examples_section = "\n".join([
        f"Example {i+1} (How2Sign): {ex}..."
        for i, ex in enumerate(how2sign_examples)
    ])
    
    prompt = f"""
You are classifying YouTube ASL video captions to determine if they match the How2Sign instructional video domain.

How2Sign videos are instructional/educational content that fall into these 10 specific topic categories:

1. Cars and Other Vehicle - Tutorials about vehicles, driving, maintenance, repairs
2. Games - Game rules, strategies, how to play games
3. Arts and Entertainment - Art techniques, creative tutorials, entertainment skills
4. Personal Care and Style - Beauty, grooming, fashion, personal hygiene tutorials
5. Food and Drinks - Cooking recipes, food preparation, drink making
6. Education and Communication - Learning skills, communication techniques, educational content
7. Home and Garden - Home improvement, gardening, DIY projects, interior design
8. Pets and Animals - Pet care, animal training, veterinary advice
9. Hobbies and Crafts - Craft tutorials, hobby instructions, creative projects
10. Sports and Fitness - Sports techniques, fitness exercises, athletic training

How2Sign videos are characterized by:
- Tutorials and how-to guides with step-by-step instructions
- Educational content teaching skills, techniques, or procedures
- Professional demonstrations and instructional content
- Clear instructional structure (not just entertainment or personal vlogs)

Examples of How2Sign content:
{examples_section}

For each YouTube video caption below, determine if it matches ANY of the 10 How2Sign topic categories above AND is instructional/educational in nature.

Classification:
- "KEEP" = The video matches one or more How2Sign categories AND is instructional/tutorial content
- "FILTER" = The video does NOT match any How2Sign category OR is not instructional (e.g., entertainment, music, news, personal vlogs, stories without instruction)

YouTube Video Captions:
{chr(10).join([f"{i+1}. {text[:400]}..." for i, text in enumerate(texts_batch)])}

Output format: KEEP or FILTER for each video (one per line, only the word)
"""
    
    results = llm.generate(prompt)
    decisions = []
    for line in results.split('\n'):
        line = line.strip().upper()
        if 'KEEP' in line:
            decisions.append(True)
        elif 'FILTER' in line:
            decisions.append(False)
        else:
            # Default to filter if unclear
            decisions.append(False)
    
    return decisions
```

**Alternative: Simpler Prompt (Faster, Less Accurate)**
```python
def llm_how2sign_simple_match(texts_batch):
    """
    Simpler version without few-shot examples for faster processing.
    """
    prompt = f"""
Classify each YouTube video caption as matching or not matching How2Sign instructional video domain.

How2Sign videos are instructional/educational content in these 10 categories:
1. Cars and Other Vehicle
2. Games
3. Arts and Entertainment
4. Personal Care and Style
5. Food and Drinks
6. Education and Communication
7. Home and Garden
8. Pets and Animals
9. Hobbies and Crafts
10. Sports and Fitness

The video must be instructional/tutorial content (how-to guides, step-by-step instructions, educational demonstrations).

For each caption, output:
- "KEEP" if it matches one or more How2Sign categories AND is instructional/tutorial
- "FILTER" if it doesn't match any category OR is not instructional (entertainment, music, news, vlogs)

Captions:
{chr(10).join([f"{i+1}. {text[:300]}..." for i, text in enumerate(texts_batch)])}

Output: KEEP or FILTER (one per line)
"""
    results = llm.generate(prompt)
    return [line.strip().upper().startswith('KEEP') for line in results.split('\n') if line.strip()]
```

**Processing**: Batch process (20-50 videos per batch) to reduce API calls

**Expected**: Filter out additional 30-50% of videos (more aggressive than original Stage 2+3 combined)

---

### Stage 3: Topic Similarity Filtering (OPTIONAL - Skip for Fast Processing)

**Note**: This stage is skipped for fast processing. Stage 2's enhanced LLM classification replaces this.

**Alternative**: If you want topic-level filtering without embeddings, you can add a topic classification step in Stage 2:
```python
def llm_topic_classification(texts_batch):
    """
    Optional: Classify videos into How2Sign topics for better organization.
    """
    prompt = f"""
Classify each video into one or more How2Sign topics:
1. Cars and Other Vehicle
2. Games
3. Arts and Entertainment
4. Personal Care and Style
5. Food and Drinks
6. Education and Communication
7. Home and Garden
8. Pets and Animals
9. Hobbies and Crafts
10. Sports and Fitness

For each caption, output topic numbers (e.g., "5,7" for Food and Home).

Captions:
{chr(10).join([f"{i+1}. {text[:300]}..." for i, text in enumerate(texts_batch)])}

Output: Topic numbers for each (one per line)
"""
    # Implementation similar to above
```

---

### Stage 4: Final Quality Check

**Purpose**: Remove duplicates and check final quality metrics.

```python
def final_quality_check(filtered_videos):
    # 1. Remove near-duplicates (similarity > 0.95)
    unique_videos = remove_duplicates(filtered_videos, threshold=0.95)
    
    # 2. Check length distribution
    lengths = [len(v['text'].split()) for v in unique_videos]
    median_length = np.median(lengths)
    
    # Remove outliers (too short or too long)
    filtered = [
        v for v in unique_videos 
        if 20 <= len(v['text'].split()) <= 500 and
           abs(len(v['text'].split()) - median_length) < 3 * np.std(lengths)
    ]
    
    return filtered
```

---

## PART 5: Implementation Pipeline

### 5.1 Complete Pipeline Flow (Fast Version - No Embeddings)

```
YouTube Videos (43K)
    ↓
[Stage 1] Rule-Based Filtering
    → Keep: ~26-30K videos (30-40% filtered)
    ↓
[Stage 2] LLM How2Sign Domain Matching (Enhanced)
    → Keep: ~13-18K videos (30-50% filtered)
    ↓
[Stage 3] Final Quality Check
    → Final: ~12-16K videos (50-70% total filtered)
    ↓
Output: Filtered video list with metadata
```


---

## PART 6: Implementation Details

### 6.1 Required Libraries

```python
# Core
import pandas as pd
import numpy as np
import json
from collections import defaultdict

# LLM (choose one)
# Option 1: OpenAI API
import openai

# Option 2: Local LLM (e.g., Qwen, InternVL)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Option 3: Anthropic Claude API
import anthropic

# Language detection
from langdetect import detect

# Utilities
import re
from tqdm import tqdm

# NOTE: Embedding libraries NOT needed for fast version
# from sentence_transformers import SentenceTransformer  # Skip this
# from sklearn.metrics.pairwise import cosine_similarity  # Skip this
```

### 6.2 Configuration

```python
CONFIG = {
    # Embedding model
    "embedding_model": "BAAI/bge-large-en-v1.5",  # or "hkunlp/instructor-large"
    
    # LLM settings
    "llm_provider": "openai",  # or "local"
    "llm_model": "gpt-4o-mini",  # or local model path
    "batch_size_llm": 20,
    
    # Filtering thresholds
    "min_words": 20,
    "max_words": 500,
    "min_similarity": 0.35,  # Base threshold
    "topic_thresholds": {...},  # Topic-specific
    
    # Processing
    "num_workers": 4,
    "chunk_size": 1000,
}
```

### 6.3 Performance Estimates

**Fast Version (No Embeddings)**:
- **Stage 1 (Rule-based)**: ~1-2 hours for 43K videos
- **Stage 2 (LLM Enhanced)**: ~6-12 hours (depends on API rate limits, batch size)
- **Stage 3 (Final check)**: ~30 minutes

**Total**: ~7-14 hours (similar to original, but simpler implementation)

**Original Version (With Embeddings)**:
- **Stage 1 (Rule-based)**: ~1-2 hours
- **Stage 2 (LLM)**: ~5-10 hours
- **Stage 3 (Embedding)**: ~2-3 hours
- **Stage 4 (Final check)**: ~30 minutes

**Total**: ~8-15 hours

---

## PART 7: Validation & Tuning

### 7.1 Small-Scale Testing

1. **Sample 1000 videos** from YouTube dataset
2. Run full pipeline
3. **Manual review** of filtered results:
   - Check false positives (kept but shouldn't)
   - Check false negatives (filtered but should keep)
4. **Adjust thresholds** based on review

### 7.2 Metrics to Track

- **Filtering rate** at each stage
- **Topic distribution** of kept videos
- **Average similarity scores** per topic
- **Processing time** per stage

### 7.3 Iterative Refinement

- Start with conservative thresholds (keep more)
- Gradually tighten based on quality assessment
- Monitor topic balance (ensure all 10 topics represented)

---

## Summary

This proposal combines:
1. ✅ **Video-level aggregation** (How2Sign + YouTube)
2. ✅ **Multi-stage filtering** (Rule → LLM → Final Check)
3. ✅ **LLM-based domain matching** (replaces embedding similarity for speed)
4. ✅ **Rich text combination** (title + description + captions)
5. ✅ **Quality checks** at each stage

**Expected Result**: Filter 50-70% of noisy YouTube videos while maintaining domain consistency with How2Sign.

---

## PART 8: Quick Start Guide (Fast Version)

### 8.1 Prerequisites

```bash
# Install required packages
pip install openai anthropic langdetect tqdm pandas

# Set API keys
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"
```

### 8.2 Run Pipeline

```bash
cd /home/mh2803/projects/sign_language_llm/data_aggregation_output

# Step 1: Data aggregation (if not done)
python part1_data_aggregation.py

# Step 2: Stage 1 - Rule-based filtering
python stage1_rule_based_filter.py

# Step 3: Stage 2 - LLM How2Sign domain matching
# Using OpenAI (default)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export BATCH_SIZE=20
python stage2_llm_how2sign_filter.py

# OR using Anthropic Claude
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-haiku-20240307
python stage2_llm_how2sign_filter.py
```

### 8.3 Output Files

After running Stage 2, you'll get:
- `youtube_video_texts_filtered_stage2.json` - Videos that passed (KEEP)
- `youtube_video_texts_filtered_out_stage2.json` - Videos filtered out with reasons
- `stage2_filtering_stats.json` - Statistics and breakdown

### 8.4 Cost Estimation

For ~30K videos (after Stage 1) with batch size 20:
- **OpenAI GPT-4o-mini**: ~$5-15 (depends on text length)
- **Anthropic Claude Haiku**: ~$3-10 (depends on text length)

**Tip**: Start with a small sample (100-500 videos) to test and estimate costs.

### 8.5 Troubleshooting

**Issue**: API rate limits
- **Solution**: Reduce `BATCH_SIZE` (e.g., 10) or add longer delays between batches

**Issue**: High costs
- **Solution**: Use simpler model (gpt-4o-mini or claude-haiku), increase batch size, or filter more aggressively in Stage 1

**Issue**: Inconsistent results
- **Solution**: Check API responses, adjust temperature (currently 0.1), or use few-shot examples from How2Sign

