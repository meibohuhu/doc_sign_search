#!/usr/bin/env python3
"""
Script to shuffle the order of objects in a JSON array file.
"""
import json
import random
import sys

def shuffle_json_file(input_file, output_file=None, seed=None):
    """
    Shuffle the order of objects in a JSON array file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input)
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"Loading JSON file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} objects")
    print(f"First object ID: {data[0]['id']}")
    print(f"Last object ID: {data[-1]['id']}")
    
    print("Shuffling objects...")
    random.shuffle(data)
    
    print(f"After shuffle - First object ID: {data[0]['id']}")
    print(f"After shuffle - Last object ID: {data[-1]['id']}")
    
    output_path = output_file if output_file else input_file
    print(f"Saving shuffled data to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    input_file = "/home/mh2803/projects/sign_language_llm/script_adobe/1125/merged_train.json"
    
    # You can specify an output file or it will overwrite the input
    output_file = None  # Set to a path if you want to keep original
    
    # Optional: set a seed for reproducibility
    seed = None  # Set to a number if you want reproducible shuffling
    
    shuffle_json_file(input_file, output_file, seed)

