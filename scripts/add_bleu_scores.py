#!/usr/bin/env python3
"""
Add BLEU scores to each entry in the results JSON file.
"""

import json
import sys
import argparse
from typing import List, Dict, Any
from collections import Counter
import math


def calculate_individual_bleu(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate BLEU scores (1-4) for a single prediction-reference pair.
    
    Args:
        prediction: The model output
        reference: The ground truth
        
    Returns:
        Dictionary with bleu1, bleu2, bleu3, bleu4 scores
    """
    try:
        # Tokenize the texts
        pred_tokens = prediction.strip().split()
        ref_tokens = reference.strip().split()
        
        # Handle empty cases
        if not pred_tokens or not ref_tokens:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
        
        bleu_scores = {}
        max_order = 4
        
        # For each BLEU order (1 to 4)
        for order in range(1, max_order + 1):
            # Generate n-grams for prediction
            pred_ngrams = []
            for i in range(len(pred_tokens) - order + 1):
                ngram = tuple(pred_tokens[i:i+order])
                pred_ngrams.append(ngram)
            
            # Generate n-grams for reference
            ref_ngrams = []
            for i in range(len(ref_tokens) - order + 1):
                ngram = tuple(ref_tokens[i:i+order])
                ref_ngrams.append(ngram)
            
            # Count n-gram occurrences in prediction and reference
            pred_counts = Counter(pred_ngrams)
            ref_counts = Counter(ref_ngrams)
            
            # Calculate clipped matches (min of pred_count and ref_count for each ngram)
            matches = 0
            for ngram, pred_count in pred_counts.items():
                matches += min(pred_count, ref_counts.get(ngram, 0))
            
            # Precision = matches / total prediction n-grams
            total_pred_ngrams = len(pred_ngrams)
            precision = matches / total_pred_ngrams if total_pred_ngrams > 0 else 0.0
            
            # Calculate accumulated precision for BLEU-n (geometric mean of all n-gram precisions)
            if order == 1:
                bleu = precision
            else:
                # Need to calculate geometric mean of all precisions up to order
                precisions = []
                for n in range(1, order + 1):
                    # Generate n-grams for order n
                    pred_ngrams_n = []
                    for i in range(len(pred_tokens) - n + 1):
                        ngram = tuple(pred_tokens[i:i+n])
                        pred_ngrams_n.append(ngram)
                    
                    ref_ngrams_n = []
                    for i in range(len(ref_tokens) - n + 1):
                        ngram = tuple(ref_tokens[i:i+n])
                        ref_ngrams_n.append(ngram)
                    
                    pred_counts_n = Counter(pred_ngrams_n)
                    ref_counts_n = Counter(ref_ngrams_n)
                    
                    matches_n = 0
                    for ngram, pred_count in pred_counts_n.items():
                        matches_n += min(pred_count, ref_counts_n.get(ngram, 0))
                    
                    total_pred_ngrams_n = len(pred_ngrams_n)
                    precision_n = matches_n / total_pred_ngrams_n if total_pred_ngrams_n > 0 else 0.0
                    precisions.append(precision_n)
                
                # Filter out impossible precisions (when sentence is too short for that n-gram order)
                # and calculate geometric mean using only available precisions
                valid_precisions = [p for p in precisions if p >= 0]
                
                if valid_precisions and len([p for p in valid_precisions if p > 0]) > 0:
                    # Calculate geometric mean of valid precisions
                    non_zero_precisions = [p for p in valid_precisions if p > 0]
                    if non_zero_precisions:
                        bleu = math.exp(sum(math.log(p) for p in non_zero_precisions) / len(valid_precisions))
                    else:
                        bleu = 0.0
                else:
                    bleu = 0.0
            
            # Apply brevity penalty: min(1, exp(1 - ref_len / pred_len))
            if len(ref_tokens) > 0:
                brevity_penalty = math.exp(1 - len(pred_tokens) / len(ref_tokens))
                brevity_penalty = min(1.0, brevity_penalty)
            else:
                brevity_penalty = 0.0
            
            bleu *= brevity_penalty
            bleu_scores[f'bleu{order}'] = bleu
        
        return bleu_scores
        
    except Exception as e:
        print(f"Error calculating BLEU for prediction '{prediction[:50] if len(prediction) > 50 else prediction}...': {e}")
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}


def add_bleu_scores_to_results(input_file: str, output_file: str = None, overwrite: bool = False):
    """
    Read results JSON file, calculate BLEU scores for each entry, and save.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional, defaults to input_file)
        overwrite: If True, overwrite the input file
    """
    # Read the input file
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} entries to process.")
    
    # Calculate BLEU scores for each entry
    for i, entry in enumerate(data):
        if i % 100 == 0:
            print(f"Processing entry {i+1}/{len(data)}...")
        
        model_output = entry.get('model_output', '')
        ground_truth = entry.get('ground_truth', '')
        
        # Calculate BLEU scores
        bleu_scores = calculate_individual_bleu(model_output, ground_truth)
        
        # Add to entry
        entry['bleu_scores'] = bleu_scores
    
    # Determine output file
    if output_file is None or overwrite:
        output_file = input_file
    elif output_file is None:
        # Create output filename from input
        base = input_file.rsplit('.', 1)[0]
        output_file = f"{base}_with_bleu.json"
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Added BLEU scores to {len(data)} entries.")
    
    # Print summary statistics
    avg_bleu1 = sum(e.get('bleu_scores', {}).get('bleu1', 0.0) for e in data) / len(data)
    avg_bleu2 = sum(e.get('bleu_scores', {}).get('bleu2', 0.0) for e in data) / len(data)
    avg_bleu3 = sum(e.get('bleu_scores', {}).get('bleu3', 0.0) for e in data) / len(data)
    avg_bleu4 = sum(e.get('bleu_scores', {}).get('bleu4', 0.0) for e in data) / len(data)
    
    print("\nAverage BLEU scores:")
    print(f"  BLEU-1: {avg_bleu1:.4f}")
    print(f"  BLEU-2: {avg_bleu2:.4f}")
    print(f"  BLEU-3: {avg_bleu3:.4f}")
    print(f"  BLEU-4: {avg_bleu4:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Add BLEU scores to model evaluation results')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to output JSON file (default: input_file)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite input file instead of creating new one')
    
    args = parser.parse_args()
    
    add_bleu_scores_to_results(
        input_file=args.input_file,
        output_file=args.output,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    main()

