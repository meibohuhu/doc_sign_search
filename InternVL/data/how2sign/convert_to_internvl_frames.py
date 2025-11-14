#!/usr/bin/env python3
"""
Rewrite How2Sign JSONL annotations so that every sample aligns with the
InternVL2.5 conversation template and produces deterministic token counts.

Usage:
    python convert_to_internvl_frames.py \
        --src train_how2sign_under10s_internvl.jsonl \
        --dst train_how2sign_under10s_internvl_fixed.jsonl \
        --frame-count 96

The script will:
  * insert an explicit system prompt;
  * expand the `<video>` placeholder into `Frame-{i}: <image>` lines;
  * strip trailing whitespace in all messages;
  * optionally compute and store the `length` field (disabled by default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "You're ASL Task Assistant."


def build_frame_block(frame_count: int) -> str:
    return "\n".join(f"Frame-{i + 1}: <image>" for i in range(frame_count))


def rewite_conversations(
    record: dict,
    *,
    frame_block: str,
    system_prompt: str,
) -> dict:
    conversations = record.get("conversations", [])
    if not conversations:
        raise ValueError("Conversation list is empty")

    # Ensure system prompt is the first turn
    if conversations[0]["from"] != "system":
        conversations = [
            {"from": "system", "value": system_prompt},
            *conversations,
        ]
    else:
        conversations[0]["value"] = system_prompt.strip()

    new_conversations = []
    for turn in conversations:
        role = turn["from"]
        value = turn["value"]
        if role == "human":
            value = value.replace("<video>", frame_block)
        value = value.strip()
        new_conversations.append({"from": role, "value": value})

    record["conversations"] = new_conversations
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source JSONL file")
    parser.add_argument("--dst", type=Path, required=True, help="Destination JSONL file")
    parser.add_argument("--frame-count", type=int, default=96, help="Number of frame placeholders to insert")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to inject when missing",
    )
    parser.add_argument(
        "--compute-length",
        action="store_true",
        help="Tokenize conversations to pre-compute the `length` field (requires HF tokenizer).",
    )
    args = parser.parse_args()

    frame_block = build_frame_block(args.frame_count)

    tokenizer = None
    if args.compute_length:
        try:
            from transformers import AutoTokenizer  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers is required when --compute-length is enabled. "
                "Install it via `pip install transformers`."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-4B", trust_remote_code=True, use_fast=False)

    processed = 0
    with args.src.open("r", encoding="utf-8") as fin, args.dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record = rewite_conversations(
                record,
                frame_block=frame_block,
                system_prompt=args.system_prompt,
            )
            if tokenizer is not None:
                text = "\n".join(turn["value"] for turn in record["conversations"])
                record["length"] = tokenizer(text, return_tensors="pt", truncation=False).input_ids.size(1)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

    print(f"Processed {processed} samples → {args.dst}")


if __name__ == "__main__":
    main()

