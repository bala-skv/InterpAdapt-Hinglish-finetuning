"""SAIL-2017 Romanized (Hinglish) sentiment data for the LoRA baseline.

The Stage-2 task is 3-class sentiment classification on code-mixed
Hindi-English social-media text: given a Romanized (Hinglish) post, predict
``negative`` / ``neutral`` / ``positive``. We frame it as label-word generation
so a causal LM (Qwen2.5-1.5B) can be scored the same way as Stage-1 tracing:
teacher-forced log P(label | prompt), with the prompt masked out of the loss.
The reported number is classification **accuracy / macro-F1**, computed by
ranking the three label verbalizers by their per-token log-prob (argmax).

Data source: Satyam's frozen, de-duplicated Romanized Stage-2 splits on the Hub
(``satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2``,
train 10,068 / validation 1,260 / test 1,261). Each HF row is a single ``text``
string of the form ``"<post>\\t<label>"`` -- the gold label is tab-separated at
the end and posts may contain embedded newlines. We parse on the final tab and
validate the label, merging any stray continuation lines defensively.

A local JSONL source (``{"text","label"}`` per line) is also supported and takes
precedence, so a refreshed split can drop in with no code change.

Owner: Jawed (Stage 2 baseline).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# The three SAIL-2017 sentiment classes (order fixed for reproducible reporting).
LABELS: List[str] = ["negative", "neutral", "positive"]
LABEL_SET = set(LABELS)

# Verbalizer: the exact target string generated/scored for each label. The
# leading space follows GPT-style BPE, keeping the label on its own token
# boundary after the ``Sentiment:`` prompt tail.
VERBALIZER = {label: " " + label for label in LABELS}

# Instruction-style prompt. A clear instruction helps the *base* (zero-shot)
# model too, which keeps the base-vs-LoRA comparison honest.
PROMPT_TEMPLATE = (
    "Classify the sentiment of this Hindi-English code-mixed social media post "
    "as negative, neutral, or positive.\n"
    "Post: {text}\n"
    "Sentiment:"
)

# HuggingFace / PyTorch convention: positions with this label are skipped by the
# cross-entropy loss.
IGNORE_INDEX = -100


@dataclass
class Example:
    """One supervised sentiment example."""

    text: str
    label: str


def _clean(text: str) -> str:
    """Collapse internal whitespace/newlines into single spaces."""
    return " ".join(str(text).strip().split())


def _iter_records(text_rows: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """Yield ``(post, label)`` from raw ``text`` rows of the form ``post\\tlabel``.

    Robust to two on-Hub layouts: complete records per row (the common case) and
    posts whose embedded newlines were split across rows by a text loader. A row
    only *completes* a record when its final tab-separated field is a valid
    label; otherwise it is buffered as a continuation of the current post.
    """
    buf: List[str] = []
    for raw in text_rows:
        raw = str(raw).rstrip("\r\n")
        if "\t" in raw:
            pre, tail = raw.rsplit("\t", 1)
            label = tail.strip().lower()
            if label in LABEL_SET:
                buf.append(pre)
                post = _clean("\n".join(buf))
                buf = []
                if post:
                    yield post, label
                continue
        buf.append(raw)  # continuation line (no valid trailing label)
    # any trailing buffer without a label is incomplete -> dropped


def load_examples_jsonl(path: str | Path, limit: Optional[int] = None) -> List[Example]:
    """Read ``{"text","label"}`` objects from a JSONL file (local task format)."""
    out: List[Example] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = _clean(obj["text"])
            label = str(obj["label"]).strip().lower()
            if text and label in LABEL_SET:
                out.append(Example(text, label))
            if limit is not None and len(out) >= limit:
                break
    return out


def load_examples_hf(
    dataset_name: str,
    split: str,
    limit: Optional[int] = None,
    config_name: Optional[str] = None,
    text_key: str = "text",
    data_file: Optional[str] = None,
) -> List[Example]:
    """Load a SAIL split from the Hub and parse the ``post\\tlabel`` text column.

    The repo ships BOTH ``raw/`` and ``sanitized/`` files, and the auto-detected
    default config merges them under each split (~2x rows, and it re-introduces
    the 12 train->eval leaks). Always pass ``data_file`` (e.g.
    ``"sanitized/train.txt"``) so only the de-duplicated sanitized split is used.
    """
    from datasets import load_dataset

    if data_file:
        ds = load_dataset(dataset_name, data_files={split: data_file}, split=split)
    else:
        ds = load_dataset(dataset_name, config_name, split=split)
    rows = (row[text_key] for row in ds)
    out: List[Example] = []
    for post, label in _iter_records(rows):
        out.append(Example(post, label))
        if limit is not None and len(out) >= limit:
            break
    return out


def encode_example(tokenizer, ex: Example, max_len: int):
    """Tokenize one ``Example`` into ``(input_ids, labels)`` with the prompt masked.

    ``labels`` is ``IGNORE_INDEX`` over the prompt span and the true token ids
    over the label verbalizer (plus EOS), so the loss only sees the label word.
    """
    prompt_ids = tokenizer.encode(
        PROMPT_TEMPLATE.format(text=ex.text), add_special_tokens=False
    )
    target_ids = tokenizer.encode(VERBALIZER[ex.label], add_special_tokens=False)
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
    hf_train_file: Optional[str] = None,
    hf_eval_file: Optional[str] = None,
    train_size: Optional[int] = None,
    eval_size: Optional[int] = None,
):
    """Resolve train/eval ``Example`` lists from whichever source is configured.

    Local JSONL takes precedence over the Hub split so a refreshed local task
    file retrains the baseline with no code change.
    """
    if jsonl_train:
        train = load_examples_jsonl(jsonl_train, limit=train_size)
        if jsonl_eval:
            eval_ex = load_examples_jsonl(jsonl_eval, limit=eval_size)
        else:
            # No dedicated eval file: hold out the tail of the train file.
            n_eval = eval_size or max(1, len(train) // 10)
            eval_ex, train = train[-n_eval:], train[:-n_eval]
        return train, eval_ex

    if not hf_dataset:
        raise ValueError(
            "no data source configured: set data.jsonl_train or data.hf_dataset"
        )
    train = load_examples_hf(
        hf_dataset, hf_train_split, limit=train_size,
        config_name=hf_config, data_file=hf_train_file,
    )
    eval_ex = load_examples_hf(
        hf_dataset, hf_eval_split, limit=eval_size,
        config_name=hf_config, data_file=hf_eval_file,
    )
    return train, eval_ex
