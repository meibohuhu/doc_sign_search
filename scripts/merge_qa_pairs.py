#!/usr/bin/env python3
"""
Merge QA pairs from segmented_train_val_combined_4000.jsonl into segmented_train_val_combined.jsonl
- Find matching videos by video field
- Replace prompt (human value) from 4000 file
- Keep answer (gpt value) from original file
- Output to new jsonl file
"""

import json
from pathlib import Path
from typing import Dict, List

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def create_video_index(data: List[Dict]) -> Dict[str, Dict]:
    """Create index mapping video name to entry"""
    index = {}
    for entry in data:
        video = entry.get('video', '')
        if video:
            index[video] = entry
    return index

def merge_qa_pairs(
    source_file: str,
    target_file: str,
    output_file: str
):
    """
    Merge QA pairs from source_file into target_file
    
    Args:
        source_file: Path to segmented_train_val_combined_4000.jsonl (has new prompts)
        target_file: Path to segmented_train_val_combined.jsonl (has original data)
        output_file: Path to output jsonl file
    """
    print(f"Loading source file: {source_file}")
    source_data = load_jsonl(source_file)
    print(f"  Loaded {len(source_data)} entries")
    
    print(f"\nLoading target file: {target_file}")
    target_data = load_jsonl(target_file)
    print(f"  Loaded {len(target_data)} entries")
    
    # Create index of target data by video name
    print("\nCreating video index from target file...")
    target_index = create_video_index(target_data)
    print(f"  Indexed {len(target_index)} videos")
    
    # Create index of source data by video name
    print("\nCreating video index from source file...")
    source_index = create_video_index(source_data)
    print(f"  Indexed {len(source_index)} videos")
    
    # Process target data and replace prompts where matches exist
    print("\nProcessing entries...")
    merged_data = []
    replaced_count = 0
    not_found_count = 0
    
    for entry in target_data:
        video = entry.get('video', '')
        
        if video in source_index:
            # Found matching video in source file
            source_entry = source_index[video]
            
            # Get new prompt from source entry
            source_conversations = source_entry.get('conversations', [])
            new_prompt = None
            for conv in source_conversations:
                if conv.get('from') == 'human':
                    new_prompt = conv.get('value', '')
                    break
            
            if new_prompt:
                # Get original prompt from target entry
                original_conversations = entry.get('conversations', [])
                original_prompt = None
                for conv in original_conversations:
                    if conv.get('from') == 'human':
                        original_prompt = conv.get('value', '')
                        break
                
                # Replace the translation instruction part in new_prompt with original prompt
                # Find the position of "\nTranslate the American Sign Language in this video to English."
                translation_instruction = "\nTranslate the American Sign Language in this video to English."
                
                if translation_instruction in new_prompt:
                    # Split at the translation instruction
                    qa_part = new_prompt.split(translation_instruction)[0]
                    
                    # Extract the instruction part from original prompt (everything after "<video>\n")
                    if original_prompt and original_prompt.startswith("<video>\n"):
                        original_instruction = original_prompt[len("<video>\n"):]
                    else:
                        original_instruction = original_prompt if original_prompt else ""
                    
                    # Combine: QA part + original instruction
                    combined_prompt = qa_part + "\n" + original_instruction
                else:
                    # If translation instruction not found, use new_prompt as is
                    combined_prompt = new_prompt
                
                # Create new entry with combined prompt but keep original answer
                new_entry = entry.copy()
                new_conversations = []
                
                # Add combined prompt
                new_conversations.append({
                    "from": "human",
                    "value": combined_prompt
                })
                
                # Keep original answer
                for conv in original_conversations:
                    if conv.get('from') == 'gpt':
                        new_conversations.append(conv)
                        break
                
                new_entry['conversations'] = new_conversations
                merged_data.append(new_entry)
                replaced_count += 1
            else:
                # No human prompt found in source, keep original
                merged_data.append(entry)
        else:
            # No match found, keep original entry
            merged_data.append(entry)
            not_found_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  Total entries: {len(merged_data)}")
    print(f"  Replaced prompts: {replaced_count}")
    print(f"  Kept original (no match): {not_found_count}")
    
    # Write output file
    print(f"\nWriting output to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully wrote {len(merged_data)} entries to {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge QA pairs: replace prompts from source file, keep answers from target file'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source JSONL file with new prompts (segmented_train_val_combined_4000.jsonl)'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Target JSONL file with original data (segmented_train_val_combined.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    
    args = parser.parse_args()
    
    merge_qa_pairs(args.source, args.target, args.output)

if __name__ == '__main__':
    main()

