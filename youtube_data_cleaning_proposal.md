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

## PART 3: Topic Centroid Calculation

### 3.1 Group Videos by Topic

```python
# Group How2Sign videos by topic
topic_videos = {topic: [] for topic in topics}
for video_id, text in how2sign_full_texts.items():
    video_topics = classify_video_topics(text)  # From Part 2
    for topic in video_topics:
        topic_videos[topic].append(text)
```

### 3.2 Compute Embedding Centroids

```python
# Load embedding model (e.g., BGE-large-en-v1.5 or Instructor-large)
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Compute centroid for each topic
topic_centroids = {}
for topic, texts in topic_videos.items():
    if len(texts) < 5:  # Skip topics with too few videos
        continue
    
    # Encode all videos in this topic
    embeddings = embed_model.encode(texts, batch_size=32)
    
    # Compute mean embedding (centroid)
    centroid = np.mean(embeddings, axis=0)
    topic_centroids[topic] = centroid
    
    print(f"Topic: {topic}, Videos: {len(texts)}, Centroid shape: {centroid.shape}")
```

**Output**: Dictionary mapping `topic → centroid_embedding` (10 centroids)

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

### Stage 2: LLM Semantic Classification (Filter 20-30%)

**Purpose**: Use LLM to filter invalid English and irrelevant content.

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

#### Task 2: Relevance-to-How2Sign Classification
```python
def llm_relevance_classification(texts_batch):
    prompt = f"""
    Classify each caption based on relevance to instructional/educational content 
    similar to How2Sign (tutorials, how-to guides, instructional videos).
    
    Categories:
    1 = Highly relevant (instructional, tutorial, how-to)
    2 = Possibly relevant (educational, informative)
    3 = Irrelevant (entertainment, music, news, personal vlog)
    
    Captions:
    {chr(10).join([f"{i+1}. {text[:200]}..." for i, text in enumerate(texts_batch)])}
    
    Output format: Category number for each (one per line)
    """
    
    results = llm.generate(prompt)
    categories = [int(r.strip()) for r in results.split('\n') if r.strip().isdigit()]
    
    # Keep only category 1 and some category 2
    return [cat in [1, 2] for cat in categories]
```

**Processing**: Batch process (20-50 videos per batch) to reduce API calls

**Expected**: Filter out additional 20-30% of videos

---

### Stage 3: Topic Similarity Filtering (Filter 10-20%)

**Purpose**: Match YouTube videos to How2Sign topics using embedding similarity.

```python
def topic_similarity_filter(youtube_video_text, topic_centroids, threshold=0.35):
    """
    Compute similarity between YouTube video and How2Sign topic centroids.
    
    Args:
        youtube_video_text: Full text of YouTube video
        topic_centroids: Dict of topic → centroid embeddings
        threshold: Minimum similarity to keep (can be topic-specific)
    
    Returns:
        (keep: bool, best_topic: str, max_similarity: float)
    """
    # Encode YouTube video
    youtube_embedding = embed_model.encode([youtube_video_text])[0]
    
    # Compute similarity with all topic centroids
    similarities = {}
    for topic, centroid in topic_centroids.items():
        similarity = cosine_similarity(
            youtube_embedding.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0][0]
        similarities[topic] = similarity
    
    # Find best matching topic
    best_topic = max(similarities, key=similarities.get)
    max_similarity = similarities[best_topic]
    
    # Topic-specific thresholds (Suggestion 3)
    topic_thresholds = {
        "Food and Drinks": 0.35,      # Common topic
        "Home and Garden": 0.35,
        "Personal Care and Style": 0.35,
        "Education and Communication": 0.35,
        "Sports and Fitness": 0.35,
        "Arts and Entertainment": 0.30,  # Broader topic
        "Games": 0.30,
        "Cars and Other Vehicle": 0.30,
        "Pets and Animals": 0.30,      # Less common
        "Hobbies and Crafts": 0.30,
    }
    
    threshold = topic_thresholds.get(best_topic, 0.35)
    
    keep = max_similarity >= threshold
    
    return keep, best_topic, max_similarity, similarities
```

**Expected**: Filter out additional 10-20% of videos

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

### 5.1 Complete Pipeline Flow

```
YouTube Videos (43K)
    ↓
[Stage 1] Rule-Based Filtering
    → Keep: ~26-30K videos (30-40% filtered)
    ↓
[Stage 2] LLM Semantic Classification
    → Keep: ~18-21K videos (20-30% filtered)
    ↓
[Stage 3] Topic Similarity Filtering
    → Keep: ~14-18K videos (10-20% filtered)
    ↓
[Stage 4] Final Quality Check
    → Final: ~12-16K videos (50-70% total filtered)
    ↓
Output: Filtered video list with metadata
```

### 5.2 Output Format

```json
{
    "video_id": "abc123",
    "title": "...",
    "matched_topic": "Food and Drinks",
    "similarity_score": 0.42,
    "all_topic_similarities": {
        "Food and Drinks": 0.42,
        "Home and Garden": 0.38,
        ...
    },
    "filtering_stages": {
        "rule_based": "passed",
        "llm_english": "passed",
        "llm_relevance": "category_1",
        "topic_similarity": "passed"
    }
}
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

# Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LLM (choose one)
# Option 1: OpenAI API
import openai

# Option 2: Local LLM (e.g., Qwen, InternVL)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Language detection
from langdetect import detect

# Utilities
import re
from tqdm import tqdm
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

- **Stage 1 (Rule-based)**: ~1-2 hours for 43K videos
- **Stage 2 (LLM)**: ~5-10 hours (depends on API rate limits)
- **Stage 3 (Embedding)**: ~2-3 hours
- **Stage 4 (Final check)**: ~30 minutes

**Total**: ~8-15 hours for full pipeline

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
2. ✅ **Multi-stage filtering** (Rule → LLM → Embedding)
3. ✅ **Topic-based similarity** (10 topic centroids)
4. ✅ **Rich text combination** (title + description + captions)
5. ✅ **Quality checks** at each stage

**Expected Result**: Filter 50-70% of noisy YouTube videos while maintaining domain consistency with How2Sign.

