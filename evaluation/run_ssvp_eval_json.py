#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

from ssvp_evaluation import comprehensive_evaluation_ssvp, print_evaluation_results_ssvp, save_evaluation_results_ssvp


def load_predictions_and_refs(json_path: Path) -> (List[str], List[str]):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expect a list of objects each having 'model_output' and 'ground_truth'
    preds = []
    refs = []
    for item in data:
        preds.append(item.get('model_output', ''))
        refs.append(item.get('ground_truth', ''))
    return preds, refs


def main():
    parser = argparse.ArgumentParser(description='Run SSVP evaluation on a JSON results file.')
    parser.add_argument('--input', '-i', required=True, help='Path to JSON file with fields: model_output, ground_truth')
    parser.add_argument('--output', '-o', help='Optional path to save evaluation metrics JSON')
    parser.add_argument('--model-name', default='Model', help='Name to show in the printed report')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    preds, refs = load_predictions_and_refs(input_path)
    results = comprehensive_evaluation_ssvp(refs, preds)
    print_evaluation_results_ssvp(results, args.model_name)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_evaluation_results_ssvp(results, str(out_path), args.model_name)


if __name__ == '__main__':
    main()
