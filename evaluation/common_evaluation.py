#!/usr/bin/env python3
"""
Common Video Evaluation Module for LLaVA-NeXT Metrics

This module provides standardized evaluation functions that can be used
across different model metrics scripts (InternVL, Gemini, LLaVA-NeXT, etc.)
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Optional sklearn import - fallback if not available
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some metrics may be limited.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def normalize_text(text: str) -> str:
    """Normalize text for evaluation by removing extra whitespace and converting to lowercase."""
    return re.sub(r'\s+', ' ', text.strip().lower())

def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return normalize_text(text).split()

def exact_match_score(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate exact match score."""
    matches = 0
    for pred, gt in zip(predictions, ground_truths):
        if normalize_text(pred) == normalize_text(gt):
            matches += 1
    return matches / len(predictions) if predictions else 0.0

def calculate_bleu_scores(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Calculate BLEU scores (1-4)."""
    smoothing = SmoothingFunction().method1
    
    bleu_scores = {
        'bleu1': 0.0,
        'bleu2': 0.0,
        'bleu3': 0.0,
        'bleu4': 0.0
    }
    
    total_bleu_1 = 0.0
    total_bleu_2 = 0.0
    total_bleu_3 = 0.0
    total_bleu_4 = 0.0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = tokenize(pred)
        gt_tokens = tokenize(gt)
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            continue
            
        # Calculate BLEU scores
        bleu_1 = sentence_bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        total_bleu_1 += bleu_1
        total_bleu_2 += bleu_2
        total_bleu_3 += bleu_3
        total_bleu_4 += bleu_4
    
    n_samples = len(predictions)
    if n_samples > 0:
        bleu_scores['bleu1'] = total_bleu_1 / n_samples
        bleu_scores['bleu2'] = total_bleu_2 / n_samples
        bleu_scores['bleu3'] = total_bleu_3 / n_samples
        bleu_scores['bleu4'] = total_bleu_4 / n_samples
    
    return bleu_scores

def calculate_f1_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Calculate F1, precision, and recall scores."""
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = set(tokenize(pred))
        gt_tokens = set(tokenize(gt))
        
        if len(gt_tokens) == 0:
            if len(pred_tokens) == 0:
                f1_scores.append(1.0)
                precision_scores.append(1.0)
                recall_scores.append(1.0)
            else:
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
        else:
            if len(pred_tokens) == 0:
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
            else:
                intersection = len(pred_tokens & gt_tokens)
                precision = intersection / len(pred_tokens)
                recall = intersection / len(gt_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                f1_scores.append(f1)
                precision_scores.append(precision)
                recall_scores.append(recall)
    
    return {
        'f1_score': np.mean(f1_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores)
    }

def calculate_rouge_l_f1(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate ROUGE-L F1 score."""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    rouge_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_tokens = tokenize(pred)
        gt_tokens = tokenize(gt)
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            rouge_scores.append(1.0)
        elif len(pred_tokens) == 0 or len(gt_tokens) == 0:
            rouge_scores.append(0.0)
        else:
            lcs_len = lcs_length(pred_tokens, gt_tokens)
            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            rouge_scores.append(f1)
    
    return np.mean(rouge_scores) if rouge_scores else 0.0

def comprehensive_evaluation(references: List[str], predictions: List[str]) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation using multiple metrics.
    
    Args:
        references: List of ground truth answers
        predictions: List of model predictions
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    if not references or not predictions:
        return {
            'exact_match': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'bleu_scores': {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0},
            'rouge_l_f1': 0.0,
            'total_samples': 0
        }
    
    # Calculate all metrics
    exact_match = exact_match_score(predictions, references)
    f1_metrics = calculate_f1_metrics(predictions, references)
    bleu_scores = calculate_bleu_scores(predictions, references)
    rouge_l_f1 = calculate_rouge_l_f1(predictions, references)
    
    return {
        'exact_match': exact_match,
        'f1_score': f1_metrics['f1_score'],
        'precision': f1_metrics['precision'],
        'recall': f1_metrics['recall'],
        'bleu_scores': bleu_scores,
        'rouge_l_f1': rouge_l_f1,
        'total_samples': len(references)
    }

def print_evaluation_results(results: Dict[str, Any], model_name: str = "Model"):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {model_name}")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Exact Match Score: {results['exact_match']:.4f}")
    print(f"  F1 Score:         {results['f1_score']:.4f}")
    print(f"  Precision:        {results['precision']:.4f}")
    print(f"  Recall:           {results['recall']:.4f}")
    print(f"  ROUGE-L F1:       {results['rouge_l_f1']:.4f}")
    
    print(f"\n📈 BLEU Scores:")
    for k, v in results['bleu_scores'].items():
        print(f"  {k.upper()}: {v:.4f}")
    
    print(f"\n📋 Summary:")
    print(f"  Total Samples:    {results['total_samples']}")
    
    print("="*60)

def save_evaluation_results(results: Dict[str, Any], output_file: str, model_name: str = "Model"):
    """Save evaluation results to JSON file."""
    eval_results = {
        "model_name": model_name,
        "evaluation_metrics": results,
        "summary": {
            "exact_match": results['exact_match'],
            "f1_score": results['f1_score'],
            "precision": results['precision'],
            "recall": results['recall'],
            "rouge_l_f1": results['rouge_l_f1'],
            "bleu_1": results['bleu_scores']['bleu1'],
            "bleu_2": results['bleu_scores']['bleu2'],
            "bleu_3": results['bleu_scores']['bleu3'],
            "bleu_4": results['bleu_scores']['bleu4'],
            "total_samples": results['total_samples']
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Evaluation metrics saved to: {output_file}")

# Legacy compatibility function for existing SLRT_metrics
def translation_performance(references: List[str], predictions: List[str]) -> Tuple[Dict[str, float], float]:
    """
    Legacy compatibility function that returns BLEU scores and ROUGE-L F1
    in the format expected by existing SLRT_metrics usage.
    """
    results = comprehensive_evaluation(references, predictions)
    return results['bleu_scores'], results['rouge_l_f1']
