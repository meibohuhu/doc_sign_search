import json
import random

jsonl_file = "/home/mh2803/projects/sign_language_llm/InternVL/data/how2sign/combined_datasets.jsonl"

print(f"Reading {jsonl_file}...")
entries = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

print(f"Loaded {len(entries)} entries")

# Shuffle the entries
print("Shuffling entries...")
random.shuffle(entries)

# Write back to file
print(f"Writing shuffled data back to {jsonl_file}...")
with open(jsonl_file, 'w', encoding='utf-8') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Done! Shuffled {len(entries)} entries")

