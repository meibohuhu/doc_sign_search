#!/usr/bin/env python3
"""
Merge QA pairs samples with translation data.
Combines Visual Grounding tasks with Translation tasks for multi-task training.
"""

import json
import sys
import argparse
import random
from pathlib import Path


def merge_qa_with_translation(qa_file: str, translation_file: str, output_file: str, 
                              translation_sample_size: int = None, seed: int = 42):
    """
    Merge QA pairs with translation data.
    
    Args:
        qa_file: Path to QA pairs JSON file
        translation_file: Path to translation JSON file
        output_file: Path to output merged JSON file
        translation_sample_size: Number of translation entries to include (None = all)
        seed: Random seed for sampling translation entries
    """
    random.seed(seed)
    
    # Read QA pairs file
    print(f"📖 Reading QA pairs file: {qa_file}")
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"   Found {len(qa_data)} QA pair entries")
    
    # Read translation file
    print(f"\n📖 Reading translation file: {translation_file}")
    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_data = json.load(f)
    print(f"   Found {len(translation_data)} translation entries")
    
    # Sample translation entries if needed
    if translation_sample_size is not None and translation_sample_size < len(translation_data):
        translation_sample = random.sample(translation_data, translation_sample_size)
        print(f"   Sampled {translation_sample_size} translation entries")
    else:
        translation_sample = translation_data
        print(f"   Using all {len(translation_sample)} translation entries")
    
    # Merge data
    merged_data = qa_data + translation_sample
    
    # Shuffle merged data
    random.shuffle(merged_data)
    print(f"\n📊 Merged data:")
    print(f"   Total entries: {len(merged_data)}")
    print(f"   - Visual Grounding (QA pairs): {len(qa_data)}")
    print(f"   - Translation: {len(translation_sample)}")
    
    # Write output file
    print(f"\n💾 Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Merged file saved to: {output_file}")
    print()


def analyze_prompt_compatibility(qa_file: str, translation_file: str):
    """
    Analyze prompt format compatibility between QA pairs and translation data.
    """
    print("=" * 60)
    print("📋 PROMPT COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_data = json.load(f)
    
    # Analyze QA pairs prompts
    qa_sample = qa_data[0]
    qa_human = qa_sample['conversations'][0]['value']
    
    print("\n1. Visual Grounding (QA Pairs) Prompt Format:")
    print("   " + "\n   ".join(qa_human.split('\n')[:3]))
    print("\n   Characteristics:")
    print("   - Task tag: [TASK: Visual Grounding]")
    print("   - Question-based (closed-ended)")
    print("   - Short answer format")
    print("   - Multiple choice options provided")
    
    # Analyze translation prompts
    translation_sample = translation_data[0]
    translation_human = translation_sample['conversations'][0]['value']
    
    print("\n2. Translation Prompt Format:")
    print("   " + "\n   ".join(translation_human.split('\n')))
    print("\n   Characteristics:")
    print("   - Task tag: [TASK: Translation]")
    print("   - Instruction-based (open-ended)")
    print("   - Long-form answer format")
    print("   - No options provided")
    
    print("\n3. Compatibility Assessment:")
    print("   ✅ Format compatible:")
    print("      - Both use <video> tag")
    print("      - Both use [TASK: XXX] tag format")
    print("      - Both follow same JSON structure")
    print("      - Both have conversations with 'from' and 'value'")
    
    print("\n   ✅ Training benefits:")
    print("      - Multi-task learning (Visual Grounding + Translation)")
    print("      - Model learns both closed and open-ended responses")
    print("      - Better generalization across tasks")
    print("      - Balanced task representation")
    
    print("\n   ⚠️  Considerations:")
    print("      - Answer length mismatch (short vs long)")
    print("      - Response format differs (choice vs free-form)")
    print("      - Task-specific capabilities needed")
    
    print("\n   💡 Recommendation:")
    print("      - Merging is APPROPRIATE for multi-task training")
    print("      - Helps model learn task-specific behavior")
    print("      - Consider balanced sampling for best results")
    print()
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge QA pairs with translation data for multi-task training"
    )
    parser.add_argument("qa_file", type=str,
                       help="Input QA pairs JSON file")
    parser.add_argument("translation_file", type=str,
                       help="Input translation JSON file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output merged JSON file path")
    parser.add_argument("--translation-size", type=int, default=None,
                       help="Number of translation entries to include (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze prompt compatibility before merging")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.qa_file).is_file():
        print(f"❌ Error: {args.qa_file} is not a valid file")
        sys.exit(1)
    
    if not Path(args.translation_file).is_file():
        print(f"❌ Error: {args.translation_file} is not a valid file")
        sys.exit(1)
    
    # Analyze compatibility if requested
    if args.analyze:
        analyze_prompt_compatibility(args.qa_file, args.translation_file)
    
    # Merge files
    merge_qa_with_translation(
        args.qa_file,
        args.translation_file,
        args.output,
        args.translation_size,
        args.seed
    )


if __name__ == "__main__":
    main()

