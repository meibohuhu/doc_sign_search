#!/usr/bin/env python3
"""
Analyze training results to diagnose potential training issues.

This script examines the model outputs vs ground truth to identify
systematic problems that might indicate training issues.
"""

import json
import sys
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np


def analyze_semantic_similarity(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Analyze if model outputs are semantically related to ground truth."""
    results = {
        "exact_matches": 0,
        "contains_keywords": 0,
        "length_difference_stats": [],
        "word_overlap_ratio": [],
        "completely_different": 0,
        "examples": {
            "good_matches": [],
            "partial_matches": [],
            "poor_matches": []
        }
    }
    
    for pred, gt in zip(predictions, ground_truths):
        pred_words = set(pred.lower().split())
        gt_words = set(gt.lower().split())
        
        # Exact match
        if pred.strip().lower() == gt.strip().lower():
            results["exact_matches"] += 1
            if len(results["examples"]["good_matches"]) < 3:
                results["examples"]["good_matches"].append((pred, gt))
        
        # Word overlap
        overlap = len(pred_words & gt_words)
        total_unique = len(pred_words | gt_words)
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0.0
        results["word_overlap_ratio"].append(overlap_ratio)
        
        # Check if prediction contains key words from ground truth
        if len(gt_words) > 0:
            key_phrase_found = any(
                word in pred.lower() for word in gt_words 
                if len(word) > 4  # Only check meaningful words
            )
            if key_phrase_found:
                results["contains_keywords"] += 1
        
        # Length difference
        length_diff = abs(len(pred.split()) - len(gt.split()))
        results["length_difference_stats"].append(length_diff)
        
        # Categorize examples
        if overlap_ratio > 0.3:
            if len(results["examples"]["partial_matches"]) < 3:
                results["examples"]["partial_matches"].append((pred, gt, overlap_ratio))
        elif overlap_ratio < 0.1:
            results["completely_different"] += 1
            if len(results["examples"]["poor_matches"]) < 5:
                results["examples"]["poor_matches"].append((pred, gt, overlap_ratio))
    
    # Calculate statistics
    results["avg_word_overlap"] = np.mean(results["word_overlap_ratio"])
    results["median_word_overlap"] = np.median(results["word_overlap_ratio"])
    results["avg_length_difference"] = np.mean(results["length_difference_stats"])
    results["total_samples"] = len(predictions)
    
    return results


def analyze_grammatical_quality(predictions: List[str]) -> Dict:
    """Check if predictions are at least grammatically correct English."""
    # Simple heuristic: check for proper sentence structure
    results = {
        "ends_with_punctuation": 0,
        "starts_with_capital": 0,
        "reasonable_length": 0,  # 5-50 words
        "contains_common_words": 0
    }
    
    common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those"}
    
    for pred in predictions:
        pred_lower = pred.lower()
        pred_words = pred.split()
        
        if pred.strip().endswith(('.', '!', '?')):
            results["ends_with_punctuation"] += 1
        
        if pred.strip() and pred.strip()[0].isupper():
            results["starts_with_capital"] += 1
        
        if 5 <= len(pred_words) <= 50:
            results["reasonable_length"] += 1
        
        if any(word.lower() in common_words for word in pred_words):
            results["contains_common_words"] += 1
    
    total = len(predictions)
    for key in ["ends_with_punctuation", "starts_with_capital", "reasonable_length", "contains_common_words"]:
        results[f"{key}_pct"] = results[key] / total if total > 0 else 0.0
    
    return results


def analyze_topic_consistency(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Check if model is generating text about similar topics to ground truth."""
    # Extract topic words (nouns, main verbs)
    # Simple approach: look for common domain-specific words
    
    pred_topics = Counter()
    gt_topics = Counter()
    
    # Common topic indicators
    topic_keywords = {}
    
    for pred, gt in zip(predictions, ground_truths):
        pred_lower = pred.lower()
        gt_lower = gt.lower()
        
        # Check for various topic categories
        if any(word in pred_lower for word in ["art", "painting", "artist", "design", "color"]):
            pred_topics["art"] += 1
        if any(word in gt_lower for word in ["art", "painting", "artist", "design", "color"]):
            gt_topics["art"] += 1
        
        if any(word in pred_lower for word in ["therapist", "client", "therapy", "self esteem"]):
            pred_topics["therapy"] += 1
        if any(word in gt_lower for word in ["therapist", "client", "therapy", "self esteem"]):
            gt_topics["therapy"] += 1
        
        if any(word in pred_lower for word in ["massage", "thumb", "leg", "palming"]):
            pred_topics["massage"] += 1
        if any(word in gt_lower for word in ["massage", "thumb", "leg", "palming"]):
            gt_topics["massage"] += 1
    
    return {
        "prediction_topics": dict(pred_topics),
        "ground_truth_topics": dict(gt_topics),
        "topic_overlap": len(set(pred_topics.keys()) & set(gt_topics.keys()))
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_issue.py <results_json_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    print(f"Analyzing results from: {results_file}")
    print("=" * 80)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    predictions = [item["model_output"] for item in data]
    ground_truths = [item["ground_truth"] for item in data]
    
    print(f"\nTotal samples: {len(predictions)}")
    print("\n" + "=" * 80)
    print("1. SEMANTIC SIMILARITY ANALYSIS")
    print("=" * 80)
    
    semantic_results = analyze_semantic_similarity(predictions, ground_truths)
    print(f"Exact matches: {semantic_results['exact_matches']} ({semantic_results['exact_matches']/len(predictions)*100:.2f}%)")
    print(f"Contains keywords from GT: {semantic_results['contains_keywords']} ({semantic_results['contains_keywords']/len(predictions)*100:.2f}%)")
    print(f"Average word overlap ratio: {semantic_results['avg_word_overlap']:.3f}")
    print(f"Median word overlap ratio: {semantic_results['median_word_overlap']:.3f}")
    print(f"Average length difference: {semantic_results['avg_length_difference']:.1f} words")
    print(f"Completely different (overlap < 0.1): {semantic_results['completely_different']} ({semantic_results['completely_different']/len(predictions)*100:.2f}%)")
    
    print("\n--- Poor Match Examples (High Overlap) ---")
    for pred, gt, overlap in semantic_results['examples']['poor_matches'][:3]:
        print(f"Overlap: {overlap:.2f}")
        print(f"GT: {gt}")
        print(f"Pred: {pred}")
        print()
    
    print("\n" + "=" * 80)
    print("2. GRAMMATICAL QUALITY ANALYSIS")
    print("=" * 80)
    
    grammar_results = analyze_grammatical_quality(predictions)
    print(f"Ends with punctuation: {grammar_results['ends_with_punctuation_pct']*100:.1f}%")
    print(f"Starts with capital: {grammar_results['starts_with_capital_pct']*100:.1f}%")
    print(f"Reasonable length (5-50 words): {grammar_results['reasonable_length_pct']*100:.1f}%")
    print(f"Contains common words: {grammar_results['contains_common_words_pct']*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("3. TOPIC CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    topic_results = analyze_topic_consistency(predictions, ground_truths)
    print("Prediction topics:")
    for topic, count in sorted(topic_results['prediction_topics'].items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")
    print("Ground truth topics:")
    for topic, count in sorted(topic_results['ground_truth_topics'].items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")
    print(f"Topic overlap: {topic_results['topic_overlap']}")
    
    print("\n" + "=" * 80)
    print("4. DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    # Issue detection
    issues = []
    recommendations = []
    
    if semantic_results['avg_word_overlap'] < 0.15:
        issues.append("Very low semantic similarity - model outputs are unrelated to ground truth")
        recommendations.append(
            "The model is generating text that doesn't match the sign language inputs. "
            "This suggests:\n"
            "  - Vision encoder may not be properly learning sign language features\n"
            "  - Training data might be misaligned (wrong video-text pairs)\n"
            "  - Model might be overfitting to training prompts rather than video content\n"
            "  - Loss might not be properly incentivizing correct translations"
        )
    
    if grammar_results['contains_common_words_pct'] > 0.8 and semantic_results['avg_word_overlap'] < 0.2:
        issues.append("Model generates fluent English but ignores video content")
        recommendations.append(
            "The model can generate grammatically correct text but isn't learning the "
            "visual-sign language mapping. Consider:\n"
            "  - Check if vision tower is frozen during training (should be unfrozen)\n"
            "  - Verify that video features are actually being used in forward pass\n"
            "  - Add visual attention visualization to confirm model is looking at video\n"
            "  - Reduce learning rate for vision components or use separate vision LR\n"
            "  - Add more explicit supervision on vision-text alignment"
        )
    
    if semantic_results['completely_different'] / len(predictions) > 0.7:
        issues.append("Majority of predictions are completely wrong")
        recommendations.append(
            "Over 70% of predictions have minimal word overlap with ground truth. "
            "This is a severe training issue:\n"
            "  - Verify training data format and video loading\n"
            "  - Check if correct checkpoint is being evaluated\n"
            "  - Ensure model architecture supports video inputs properly\n"
            "  - Consider that loss reduction might not correlate with translation quality"
        )
    
    if semantic_results['avg_length_difference'] > 10:
        issues.append("Large length mismatch between predictions and ground truth")
        recommendations.append(
            "Predictions have very different lengths than ground truth. This might indicate:\n"
            "  - Model is not learning proper sequence length control\n"
            "  - Training uses different length distributions than evaluation\n"
            "  - EOS token training might be insufficient"
        )
    
    if not issues:
        issues.append("No obvious issues detected (or analysis needs refinement)")
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. {issue}")
    
    print("\n--- Recommendations ---")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")
    
    print("\n" + "=" * 80)
    print("GENERAL TRAINING ADVICE")
    print("=" * 80)
    print("""
Based on BLEU4=6.12% and loss=1.4, here are likely issues:

1. **Loss-Task Mismatch**: 
   - Loss of 1.4 is reasonable for language modeling but doesn't ensure translation quality
   - Cross-entropy loss optimizes token prediction, not semantic correctness
   - Consider adding BLEU/ROUGE as validation metric during training

2. **Vision-Text Alignment**:
   - Model might be learning to generate text without properly attending to video
   - Check attention weights to ensure model focuses on sign language regions
   - Consider freezing/unfreezing strategy for vision encoder

3. **Training Data Issues**:
   - Verify video-text alignment in training set
   - Check for data leakage or corrupted pairs
   - Ensure sufficient diversity in training examples

4. **Training Strategy**:
   - Try curriculum learning (start with shorter sequences)
   - Use teacher forcing correctly (not just at training time)
   - Consider fine-tuning vision encoder separately first
   - Add contrastive learning to align visual and text features

5. **Evaluation**:
   - Low BLEU on training subset is concerning - suggests underfitting or misalignment
   - Check if evaluation is using the same preprocessing as training
   - Consider whether 5000 steps is sufficient for convergence
    """)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

