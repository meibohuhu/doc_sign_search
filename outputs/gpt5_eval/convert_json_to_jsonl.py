#!/usr/bin/env python3
"""
Convert JSON array file to JSONL format.
Each line contains one JSON object in compact format (no indentation).
"""

import json
import sys
import argparse
from pathlib import Path


def convert_json_to_jsonl(input_file: str, output_file: str):
    """
    Convert JSON array file to JSONL format.
    
    Args:
        input_file: Path to input JSON file (array format)
        output_file: Path to output JSONL file (one JSON object per line)
    """
    # Read input JSON file
    print(f"📖 Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Found {len(data)} entries")
    
    # Write output JSONL file
    print(f"\n💾 Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            # Write each entry as a compact JSON line (no indentation)
            json_line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
            f.write(json_line + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Input entries: {len(data)}")
    print(f"Output lines: {len(data)}")
    print(f"\n✅ Converted file saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON array file to JSONL format"
    )
    parser.add_argument("input_file", type=str,
                       help="Input JSON file (array format)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSONL file path (default: replaces .json with .jsonl)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"❌ Error: {args.input_file} is not a valid file")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        output_file = str(input_path.parent / f"{input_path.stem}{input_path.suffix.replace('.json', '.jsonl')}")
    else:
        output_file = args.output
    
    convert_json_to_jsonl(args.input_file, output_file)


if __name__ == "__main__":
    main()

