#!/usr/bin/env python3
"""
SSVP-SLT Style Evaluation Module for LLaVA-NeXT Metrics

This module provides evaluation functions using the exact same implementation
as ssvp_slt/translation/engine_translation.py for consistency:

- BLEU via SacreBLEU (Papineni et al., 2002; Post, 2018)
- ROUGE-L via HuggingFace evaluate (Lin, 2004)
- BLEURT-20 via HuggingFace evaluate (Sellam et al., 2020)
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
import inspect

# Import evaluation metrics using the same approach as SSVP-SLT
try:
    import evaluate as hf_evaluate
    HF_EVALUATE_AVAILABLE = True
except ImportError:
    HF_EVALUATE_AVAILABLE = False
    print("Warning: evaluate (HuggingFace) not available. Some metrics will be skipped.")

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. BLEU metrics will be skipped.")

# Load metrics the same way as in engine_translation.py
if HF_EVALUATE_AVAILABLE:
    try:
        rouge_metric = hf_evaluate.load("rouge")
        ROUGE_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not load ROUGE metric: {e}")
        ROUGE_AVAILABLE = False
    
    try:
        bleurt_metric = hf_evaluate.load("bleurt", module_type="metric", config_name="BLEURT-20")
        BLEURT_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not load BLEURT-20 metric: {e}")
        BLEURT_AVAILABLE = False
else:
    ROUGE_AVAILABLE = False
    BLEURT_AVAILABLE = False

def normalize_text(text: str) -> str:
    """Normalize text for evaluation by removing extra whitespace and converting to lowercase."""
    import re
    return re.sub(r'\s+', ' ', text.strip().lower())

def exact_match_score(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate exact match score."""
    matches = 0
    for pred, gt in zip(predictions, ground_truths):
        if normalize_text(pred) == normalize_text(gt):
            matches += 1
    return matches / len(predictions) if predictions else 0.0

def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return normalize_text(text).split()

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

def calculate_bleu_scores_ssvp(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU scores using the exact same approach as SSVP-SLT.
    Replicates the compute_bleu function from utils_translation.py.
    """
    if not SACREBLEU_AVAILABLE:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
    
    if not predictions or not ground_truths:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
    
    try:
        # Use the same approach as in engine_translation.py line 261
        # This collects the statistics like in the original code
        bleu = sacrebleu.corpus_bleu(predictions, [ground_truths])
        
        # Extract the counts and totals like in utils_translation.py compute_bleu function
        # This replicates the exact logic from lines 265-272 in utils_translation.py
        import inspect
        
        try:
            from sacrebleu.metrics import BLEU
            comp_bleu = BLEU.compute_bleu
        except ImportError:
            # compatibility API for sacrebleu 1.x
            comp_bleu = sacrebleu.compute_bleu

        fn_sig = inspect.getfullargspec(comp_bleu)[0]
        if "smooth_method" in fn_sig:
            smooth = {"smooth_method": "exp"}
        else:
            smooth = {"smooth": "exp"}
        
        # Calculate BLEU scores for different orders using the exact same method
        # as in utils_translation.py lines 265-272
        bleu_scores = {}
        for order in range(1, 5):  # BLEU-1 to BLEU-4
            bleu_result = comp_bleu(
                correct=np.array([int(bleu.counts[i]) for i in range(order)]),
                total=np.array([int(bleu.totals[i]) for i in range(order)]),
                sys_len=int(bleu.sys_len),
                ref_len=int(bleu.ref_len),
                max_ngram_order=order,
                **smooth,
            )
            bleu_scores[f'bleu{order}'] = bleu_result.score / 100.0  # Convert to 0-1 scale
        
        return bleu_scores
        
    except Exception as e:
        print(f"Error calculating BLEU scores: {e}")
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}

def calculate_rouge_l_ssvp(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate ROUGE-L score using the exact same approach as SSVP-SLT.
    Uses HuggingFace evaluate rouge metric (line 293 in engine_translation.py).
    """
    if not ROUGE_AVAILABLE:
        return 0.0
    
    if not predictions or not ground_truths:
        return 0.0
    
    try:
        # Use the same approach as in engine_translation.py line 293
        rouge_results = rouge_metric.compute(predictions=predictions, references=ground_truths)
        return rouge_results['rougeL']
        
    except Exception as e:
        print(f"Error calculating ROUGE-L score: {e}")
        return 0.0

def calculate_bleurt_score_ssvp(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate BLEURT score using the exact same approach as SSVP-SLT.
    Uses HuggingFace evaluate bleurt metric with BLEURT-20 config (lines 349, 355-358).
    """
    if not BLEURT_AVAILABLE:
        print("BLEURT-20 not available, skipping BLEURT evaluation")
        return 0.0
    
    if not predictions or not ground_truths:
        return 0.0
    
    try:
        # Use the same approach as in engine_translation.py lines 355-358
        bleurt_results = bleurt_metric.compute(
            predictions=predictions, references=ground_truths
        )
        # Return mean score, same as line 358
        return np.array(bleurt_results["scores"]).mean()
        
    except Exception as e:
        print(f"Error calculating BLEURT scores: {e}")
        return 0.0

def comprehensive_evaluation_ssvp(references: List[str], predictions: List[str]) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation using the exact same metrics as SSVP-SLT.
    
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
            'bleurt_score': 0.0,
            'total_samples': 0
        }
    
    # Calculate all metrics using SSVP-SLT approach
    exact_match = exact_match_score(predictions, references)
    f1_metrics = calculate_f1_metrics(predictions, references)
    bleu_scores = calculate_bleu_scores_ssvp(predictions, references)
    rouge_l_f1 = calculate_rouge_l_ssvp(predictions, references)
    bleurt_score = calculate_bleurt_score_ssvp(predictions, references)
    
    return {
        'exact_match': exact_match,
        'f1_score': f1_metrics['f1_score'],
        'precision': f1_metrics['precision'],
        'recall': f1_metrics['recall'],
        'bleu_scores': bleu_scores,
        'rouge_l_f1': rouge_l_f1,
        'bleurt_score': bleurt_score,
        'total_samples': len(references)
    }

def print_evaluation_results_ssvp(results: Dict[str, Any], model_name: str = "Model"):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {model_name} (SSVP-SLT Style)")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Exact Match Score: {results['exact_match']:.4f}")
    print(f"  F1 Score:         {results['f1_score']:.4f}")
    print(f"  Precision:        {results['precision']:.4f}")
    print(f"  Recall:           {results['recall']:.4f}")
    print(f"  ROUGE-L F1:       {results['rouge_l_f1']:.4f}")
    print(f"  BLEURT-20 Score:  {results.get('bleurt_score', 0.0):.4f}")
    
    print(f"\n📈 BLEU Scores (SacreBLEU - SSVP-SLT Style):")
    for k, v in results['bleu_scores'].items():
        print(f"  {k.upper()}: {v:.4f}")
    
    print(f"\n📋 Summary:")
    print(f"  Total Samples:    {results['total_samples']}")
    
    print("="*60)

def save_evaluation_results_ssvp(results: Dict[str, Any], output_file: str, model_name: str = "Model"):
    """Save evaluation results to JSON file."""
    eval_results = {
        "model_name": model_name,
        "evaluation_style": "SSVP-SLT",
        "evaluation_metrics": results,
        "summary": {
            "exact_match": results['exact_match'],
            "f1_score": results['f1_score'],
            "precision": results['precision'],
            "recall": results['recall'],
            "rouge_l_f1": results['rouge_l_f1'],
            "bleurt_score": results.get('bleurt_score', 0.0),
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

# Convenience functions that match the original interface
def comprehensive_evaluation(references: List[str], predictions: List[str]) -> Dict[str, Any]:
    """Convenience function that uses SSVP-SLT style evaluation."""
    return comprehensive_evaluation_ssvp(references, predictions)

def print_evaluation_results(results: Dict[str, Any], model_name: str = "Model"):
    """Convenience function that uses SSVP-SLT style printing."""
    return print_evaluation_results_ssvp(results, model_name)

def save_evaluation_results(results: Dict[str, Any], output_file: str, model_name: str = "Model"):
    """Convenience function that uses SSVP-SLT style saving."""
    return save_evaluation_results_ssvp(results, output_file, model_name)
