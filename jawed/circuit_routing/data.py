"""Parallel English -> Hindi data for the LoRA fine-tuning baseline.

The baseline is a supervised generation task: given an English sentence, produce
its Hindi (Devanagari) translation. Loss is computed on the Hindi target only --
the prompt tokens are masked with ``IGNORE_INDEX`` so the model is scored on
translating, not on echoing the prompt. This matches the Stage-1 teacher-forced
target-probability metric, so the baseline number is on the same scale as the
tracing scores.

Two sources are supported:

* a local JSONL file, one ``{"en": ..., "hi": ...}`` object per line. This is
  where the finetuning task drops in when Satyam's split lands -- point the
  config at it and nothing else changes.
* a Hugging Face parallel corpus (default ``cfilt/iitb-english-hindi``), used
  when no local file is given. Its rows are ``{"translation": {"en", "hi"}}``.

Owner: Jawed (Stage 2 baseline).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Prompt shown to the model. The trailing space keeps the first Hindi token off
# the ``Hindi:`` label boundary, which the Stage-1 tokenizer analysis showed
# matters for Devanagari BPE fragmentation.
PROMPT_TEMPLATE = "English: {en}\nHindi: "

# HuggingFace / PyTorch convention: positions with this label are skipped by the
# cross-entropy loss.
IGNORE_INDEX = -100


@dataclass
class Pair:
    """One supervised translation example."""

    en: str
    hi: str


def _clean(text: str) -> str:
    """Collapse whitespace; parallel corpora are full of stray tabs/newlines."""
    return " ".join(str(text).strip().split())


def load_pairs_jsonl(path: str | Path, limit: Optional[int] = None) -> List[Pair]:
    """Read ``{"en", "hi"}`` objects from a JSONL file (Satyam's task format)."""
    pairs: List[Pair] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            en, hi = _clean(obj["en"]), _clean(obj["hi"])
            if en and hi:
                pairs.append(Pair(en, hi))
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


def load_pairs_hf(
    dataset_name: str,
    split: str,
    limit: Optional[int] = None,
    config_name: Optional[str] = None,
    en_key: str = "en",
    hi_key: str = "hi",
) -> List[Pair]:
    """Stream a HF parallel corpus into ``Pair``s.

    Handles both the nested ``{"translation": {...}}`` layout (IIT-B) and a flat
    ``{"en": ..., "hi": ...}`` layout. ``streaming=True`` avoids downloading the
    full multi-GB corpus when only a small ``limit`` is needed for one number.
    """
    from datasets import load_dataset

    stream = limit is not None
    ds = load_dataset(
        dataset_name,
        config_name,
        split=split,
        streaming=stream,
    )
    pairs: List[Pair] = []
    for row in ds:
        rec = row.get("translation", row) if isinstance(row, dict) else row
        try:
            en, hi = _clean(rec[en_key]), _clean(rec[hi_key])
        except (KeyError, TypeError):
            continue
        if en and hi:
            pairs.append(Pair(en, hi))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def encode_example(tokenizer, pair: Pair, max_len: int):
    """Tokenize one ``Pair`` into ``(input_ids, labels)`` with the prompt masked.

    ``labels`` is ``IGNORE_INDEX`` over the prompt span and the true token ids
    over the Hindi target (plus EOS), so the loss only sees the translation.
    """
    prompt_ids = tokenizer.encode(
        PROMPT_TEMPLATE.format(en=pair.en), add_special_tokens=False
    )
    target_ids = tokenizer.encode(pair.hi, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + target_ids)[:max_len]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + target_ids)[:max_len]
    return input_ids, labels


def collate_batch(examples, pad_token_id: int):
    """Right-pad a list of ``(input_ids, labels)`` into padded tensors.

    Padding is masked in both ``attention_mask`` (0) and ``labels``
    (``IGNORE_INDEX``) so padded positions contribute nothing to the loss.
    """
    import torch

    max_len = max(len(ids) for ids, _ in examples)
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in examples:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_splits(
    *,
    jsonl_train: Optional[str] = None,
    jsonl_eval: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_train_split: str = "train",
    hf_eval_split: str = "validation",
    train_size: Optional[int] = None,
    eval_size: Optional[int] = None,
):
    """Resolve train/eval ``Pair`` lists from whichever source is configured.

    Local JSONL takes precedence over the HF corpus so the moment Satyam's task
    file exists the baseline retrains on it with no code change.
    """
    if jsonl_train:
        train = load_pairs_jsonl(jsonl_train, limit=train_size)
        if jsonl_eval:
            eval_pairs = load_pairs_jsonl(jsonl_eval, limit=eval_size)
        else:
            # No dedicated eval file: hold out the tail of the train file.
            n_eval = eval_size or max(1, len(train) // 10)
            eval_pairs, train = train[-n_eval:], train[:-n_eval]
        return train, eval_pairs

    if not hf_dataset:
        raise ValueError(
            "no data source configured: set data.jsonl_train or data.hf_dataset"
        )
    if hf_eval_split:
        # Separate eval split (may be a different domain than train).
        train = load_pairs_hf(hf_dataset, hf_train_split, limit=train_size, config_name=hf_config)
        eval_pairs = load_pairs_hf(hf_dataset, hf_eval_split, limit=eval_size, config_name=hf_config)
        return train, eval_pairs

    # hf_eval_split is null -> carve the eval set from the tail of the train
    # split so both are the SAME distribution. This isolates "did LoRA learn the
    # task" from any train-vs-official-dev domain shift.
    n_eval = eval_size or 200
    total = (train_size + n_eval) if train_size else None
    combined = load_pairs_hf(hf_dataset, hf_train_split, limit=total, config_name=hf_config)
    if len(combined) <= n_eval:
        raise ValueError(f"not enough data ({len(combined)}) to hold out {n_eval} eval examples")
    eval_pairs, train = combined[-n_eval:], combined[:-n_eval]
    return train, eval_pairs
