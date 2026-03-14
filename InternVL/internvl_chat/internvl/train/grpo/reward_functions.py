"""
Rule-based reward functions for GRPO training on sign language translation.
Each function takes (completions, ground_truths, **kwargs) and returns list[float].
"""

import os
import warnings
from typing import List

import torch


def bleu_reward(completions: list, ground_truths: list, **kwargs) -> List[float]:
    """BLEU-4 reward for sign language translation quality."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoother = SmoothingFunction().method1
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        text = text.strip()
        gt = gt.strip()
        if not text:
            rewards.append(0.0)
            continue
        try:
            score = sentence_bleu(
                [gt.split()],
                text.split(),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoother,
            )
            rewards.append(float(score))
        except Exception:
            rewards.append(0.0)
    return rewards


def bleu1_reward(completions: list, ground_truths: list, **kwargs) -> List[float]:
    """BLEU-1 reward for unigram precision."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoother = SmoothingFunction().method1
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        text = text.strip()
        gt = gt.strip()
        if not text:
            rewards.append(0.0)
            continue
        try:
            score = sentence_bleu(
                [gt.split()],
                text.split(),
                weights=(1.0, 0, 0, 0),
                smoothing_function=smoother,
            )
            rewards.append(float(score))
        except Exception:
            rewards.append(0.0)
    return rewards


def rouge_reward(completions: list, ground_truths: list, **kwargs) -> List[float]:
    """ROUGE-L F1 reward using Longest Common Subsequence."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        text = text.strip()
        gt = gt.strip()
        if not text:
            rewards.append(0.0)
            continue
        try:
            score = scorer.score(gt, text)
            rewards.append(float(score['rougeL'].fmeasure))
        except Exception:
            rewards.append(0.0)
    return rewards


def bertscore_reward(completions: list, ground_truths: list, **kwargs) -> List[float]:
    """BERTScore F1 reward for semantic similarity."""
    import bert_score

    texts = []
    refs = []
    for completion, gt in zip(completions, ground_truths):
        text = completion if isinstance(completion, str) else completion[0]["content"]
        text = text.strip() if text.strip() else "."  # bert_score needs non-empty
        texts.append(text)
        refs.append(gt.strip())

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P, R, F1 = bert_score.score(
                texts,
                refs,
                lang="en",
                model_type="bert-base-uncased",
                batch_size=len(texts),
                verbose=False,
                device="cpu",  # compute on CPU to avoid GPU memory contention
            )
        return F1.tolist()
    except Exception as e:
        print(f"[bertscore_reward] Error: {e}")
        return [0.0] * len(texts)
