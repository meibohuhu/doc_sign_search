#!/usr/bin/env python3
"""
Add BLEU scores to each entry in the results JSON file using SSVP method (sacrebleu).
"""

import json
import sys
import argparse
from typing import List, Dict, Any
import numpy as np
import inspect

# Import sacrebleu for SSVP-style BLEU calculation
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Error: sacrebleu is not available. Please install it with: pip install sacrebleu")
    sys.exit(1)


def calculate_individual_bleu_ssvp(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate BLEU scores (1-4) for a single prediction-reference pair using SSVP method.
    
    Args:
        prediction: The model output
        reference: The ground truth
        
    Returns:
        Dictionary with bleu1, bleu2, bleu3, bleu4 scores
    """
    try:
        # Tokenize the texts (sacrebleu expects word-tokenized strings)
        pred_tokens = prediction.strip().split()
        ref_tokens = reference.strip().split()
        
        # Handle empty cases
        if not pred_tokens or not ref_tokens:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
        
        # Calculate corpus BLEU for individual pair
        # sacrebleu expects lists, so we wrap our single strings in lists
        # Use -lc --tokenize 13a configuration for consistent evaluation
        bleu = sacrebleu.corpus_bleu(
            [prediction], 
            [[reference]],
            lowercase=True,      # -lc flag: lowercase the data
            tokenize='13a'       # --tokenize 13a: use 13a tokenization
        )
        
        # Get the compute_bleu function
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
        
        # Calculate BLEU scores for different orders
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
            # Convert to 0-1 scale (sacrebleu returns 0-100)
            bleu_scores[f'bleu{order}'] = bleu_result.score / 100.0
        
        return bleu_scores
        
    except Exception as e:
        print(f"Error calculating BLEU for prediction '{prediction[:50] if len(prediction) > 50 else prediction}...': {e}")
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}


def add_bleu_scores_to_results(input_file: str, output_file: str = None, overwrite: bool = False):
    """
    Read results JSON file, calculate BLEU scores for each entry using SSVP method, and save.
    
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
        if (i + 1) % 100 == 0:
            print(f"Processing entry {i+1}/{len(data)}...")
        
        model_output = entry.get('model_output', '')
        ground_truth = entry.get('ground_truth', '')
        
        # Calculate BLEU scores using SSVP method
        bleu_scores = calculate_individual_bleu_ssvp(model_output, ground_truth)
        
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
    
    print("\nAverage BLEU scores (SSVP-SLT method):")
    print(f"  BLEU-1: {avg_bleu1:.4f}")
    print(f"  BLEU-2: {avg_bleu2:.4f}")
    print(f"  BLEU-3: {avg_bleu3:.4f}")
    print(f"  BLEU-4: {avg_bleu4:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Add BLEU scores to model evaluation results using SSVP method')
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


