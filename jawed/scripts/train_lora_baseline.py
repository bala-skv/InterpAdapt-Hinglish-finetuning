"""Uniform-LoRA fine-tuning baseline -- the Stage-2 "one real number".

This is the baseline the interpretability-guided Circuit-Routing Adapter will be
measured against: a *uniform* LoRA (every targeted head adapted equally) on the
English -> Hindi translation task. It reports one comparable number --

    mean per-token teacher-forced log-probability of the Hindi target on a
    held-out eval split, for the base model vs. the LoRA-adapted model --

using the same metric as Stage-1 tracing (see bala/sequence_scoring.py), so the
scales line up across the project.

Design notes
------------
* **fp16**, LoRA on ``q_proj`` + ``o_proj`` only. Those are the head-aligned
  projections the per-head routing mask targets (k/v are KV-group aligned), so
  the uniform baseline and the future masked adapter share an adapter surface.
* **Resumable.** Adapter weights, optimizer, scheduler, step counter and RNG are
  checkpointed atomically to the (durable) run dir. ``--resume <run_dir>`` picks
  up from the last checkpoint, and ``SIGUSR1`` (sent by SLURM before a timeout)
  triggers a graceful save so a chained job continues cleanly. Resume is by step
  count, not exact dataloader position -- fine for this baseline.
* Base vs. LoRA eval both come from the one adapted model via PEFT's
  ``disable_adapter()`` context, so there is no second model to load.

Usage::

    python scripts/train_lora_baseline.py --config configs/lora_baseline.yaml
    python scripts/train_lora_baseline.py --config configs/lora_baseline.yaml \
        --resume runs/lora-baseline-20260928-101500

Owner: Jawed.
"""
import _bootstrap  # noqa: F401

import argparse
import dataclasses
import os
import signal
import tempfile
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import List, Optional

from circuit_routing.config import ModelConfig, QuantConfig
from circuit_routing.data import (
    Pair,
    PROMPT_TEMPLATE,
    collate_batch,
    encode_example,
    load_splits,
)
from circuit_routing.logging_utils import get_logger, make_run_dir, write_json
from circuit_routing.model_loading import load_model, load_tokenizer, verify_base_checkpoint
from circuit_routing.seeding import set_seed


# --------------------------------------------------------------------------- #
# Config (self-contained; parsed straight from YAML)
# --------------------------------------------------------------------------- #
@dataclass
class LoraCfg:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "o_proj"])


@dataclass
class DataCfg:
    jsonl_train: Optional[str] = None
    jsonl_eval: Optional[str] = None
    hf_dataset: Optional[str] = "cfilt/iitb-english-hindi"
    hf_config: Optional[str] = None
    hf_train_split: str = "train"
    hf_eval_split: str = "validation"
    train_size: Optional[int] = 4000
    eval_size: Optional[int] = 200
    max_len: int = 256


@dataclass
class TrainCfg:
    seed: int = 0
    deterministic: bool = True
    batch_size: int = 4
    grad_accum_steps: int = 8
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_steps: int = 20
    max_steps: int = 400
    ckpt_every: int = 50
    eval_every: int = 0            # 0 = only at the end
    max_grad_norm: float = 1.0


@dataclass
class BaselineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    lora: LoraCfg = field(default_factory=LoraCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    output_dir: str = "runs"


_SECTIONS = {
    "model": ModelConfig,
    "quant": QuantConfig,
    "lora": LoraCfg,
    "data": DataCfg,
    "train": TrainCfg,
}


def _section(cls, overrides):
    if not overrides:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**overrides)


def load_baseline_config(path: Optional[str]) -> BaselineConfig:
    if path is None:
        return BaselineConfig()
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - set(_SECTIONS) - {"output_dir"}
    if unknown:
        raise ValueError(f"Unknown top-level sections: {sorted(unknown)}")
    return BaselineConfig(
        output_dir=raw.get("output_dir", "runs"),
        **{name: _section(cls, raw.get(name)) for name, cls in _SECTIONS.items()},
    )


# --------------------------------------------------------------------------- #
# Eval metric (teacher-forced target log-prob; mirrors Stage-1 scoring)
# --------------------------------------------------------------------------- #
def _common_prefix_len(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def score_pair(model, tokenizer, pair: Pair, device, max_len: int) -> Optional[float]:
    """Mean per-token log P(hi_target | prompt), teacher-forced. None if empty."""
    import torch

    prompt = PROMPT_TEMPLATE.format(en=pair.en)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + pair.hi, add_special_tokens=False)

    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] != prompt_ids:  # BPE merge across the boundary
        n_prompt = _common_prefix_len(prompt_ids, full_ids)

    full_ids = full_ids[:max_len]
    n_target = len(full_ids) - n_prompt
    if n_target <= 0:
        return None

    ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logprobs = logits.log_softmax(-1)
    total = 0.0
    for k in range(n_target):
        total += logprobs[n_prompt + k - 1, full_ids[n_prompt + k]].item()
    return total / n_target


def evaluate(model, tokenizer, pairs: List[Pair], device, max_len: int) -> dict:
    """Base-vs-LoRA mean target log-prob over the eval split.

    The base score is read with the adapter disabled, so both numbers come from
    the single adapted model.
    """
    import math

    model.eval()

    def _mean_logp(active: bool) -> tuple[float, int]:
        scores = []
        ctx = _adapter_context(model, active)
        with ctx:
            for p in pairs:
                s = score_pair(model, tokenizer, p, device, max_len)
                if s is not None:
                    scores.append(s)
        mean = sum(scores) / len(scores) if scores else float("nan")
        return mean, len(scores)

    lora_logp, n = _mean_logp(active=True)
    base_logp, _ = _mean_logp(active=False)
    return {
        "n_eval": n,
        "base_logp": base_logp,
        "lora_logp": lora_logp,
        "delta_logp": lora_logp - base_logp,
        "base_ppl": math.exp(-base_logp) if base_logp == base_logp else float("nan"),
        "lora_ppl": math.exp(-lora_logp) if lora_logp == lora_logp else float("nan"),
    }


def _adapter_context(model, active: bool):
    """Enable (LoRA) or disable (base) the adapter for scoring."""
    import contextlib

    if active:
        return contextlib.nullcontext()
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# Checkpointing (atomic writes to the durable run dir)
# --------------------------------------------------------------------------- #
def _adapter_state(model) -> dict:
    from peft import get_peft_model_state_dict

    return get_peft_model_state_dict(model)


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int) -> None:
    """Atomically write {adapter, optimizer, scheduler, step, RNG} to ``path``."""
    import torch

    payload = {
        "step": step,
        "adapter": _adapter_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(path: Path, model, optimizer, scheduler) -> int:
    """Restore state from ``path``; returns the step to resume from."""
    import torch
    from peft import set_peft_model_state_dict

    ckpt = torch.load(path, map_location="cpu")
    set_peft_model_state_dict(model, ckpt["adapter"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
    return int(ckpt["step"])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_model(cfg: BaselineConfig, tokenizer, log):
    """Load the pinned fp16 base and wrap it with a uniform LoRA adapter."""
    import torch
    from peft import LoraConfig, get_peft_model

    # quant_mode="none" -> load_model uses model.dtype (float16 per config).
    model = load_model(cfg.model, cfg.quant, quant_mode="none", device_map="auto")
    verify_base_checkpoint(cfg.model, model.config, tokenizer)

    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None
    log.info("base loaded: dtype=%s device_cap=%s", cfg.model.dtype, major)

    lora = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("LoRA r=%d on %s | trainable %d / %d (%.3f%%)",
             cfg.lora.rank, cfg.lora.target_modules, trainable, total,
             100.0 * trainable / total)
    return model


def train(cfg: BaselineConfig, run_dir: Path, resume: bool, log) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(cfg.model)

    log.info("loading data...")
    train_pairs, eval_pairs = load_splits(
        jsonl_train=cfg.data.jsonl_train,
        jsonl_eval=cfg.data.jsonl_eval,
        hf_dataset=cfg.data.hf_dataset,
        hf_config=cfg.data.hf_config,
        hf_train_split=cfg.data.hf_train_split,
        hf_eval_split=cfg.data.hf_eval_split,
        train_size=cfg.data.train_size,
        eval_size=cfg.data.eval_size,
    )
    log.info("train=%d eval=%d pairs", len(train_pairs), len(eval_pairs))
    if not train_pairs or not eval_pairs:
        raise SystemExit("empty train/eval split; check the data config")

    encoded = [encode_example(tokenizer, p, cfg.data.max_len) for p in train_pairs]
    loader = DataLoader(
        encoded,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=lambda ex: collate_batch(ex, tokenizer.pad_token_id),
        drop_last=True,
    )

    model = build_model(cfg, tokenizer, log)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.train.warmup_steps, cfg.train.max_steps
    )

    ckpt_path = run_dir / "ckpt.pt"
    start_step = 0
    if resume and ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        log.info("resumed from %s at step %d", ckpt_path, start_step)

    # SLURM sends SIGUSR1 shortly before the wall-time kill; save and exit 0 so
    # the chained job resumes instead of losing the last interval of training.
    stop = {"flag": False}

    def _on_usr1(signum, frame):
        log.info("SIGUSR1 received -> graceful checkpoint and exit")
        stop["flag"] = True

    signal.signal(signal.SIGUSR1, _on_usr1)

    log.info("training %d -> %d steps (bs=%d x accum=%d)",
             start_step, cfg.train.max_steps, cfg.train.batch_size,
             cfg.train.grad_accum_steps)

    data_iter = cycle(loader)
    step = start_step
    while step < cfg.train.max_steps and not stop["flag"]:
        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(cfg.train.grad_accum_steps):
            batch = {k: v.to(device) for k, v in next(data_iter).items()}
            out = model(**batch)
            (out.loss / cfg.train.grad_accum_steps).backward()
            loss_val += out.loss.item() / cfg.train.grad_accum_steps
        torch.nn.utils.clip_grad_norm_(trainable, cfg.train.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % 10 == 0:
            log.info("step %4d/%d  loss=%.4f  lr=%.2e",
                     step, cfg.train.max_steps, loss_val, scheduler.get_last_lr()[0])

        if cfg.train.ckpt_every and step % cfg.train.ckpt_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step)
            log.info("checkpoint @ step %d -> %s", step, ckpt_path)

        if cfg.train.eval_every and step % cfg.train.eval_every == 0:
            m = evaluate(model, tokenizer, eval_pairs, device, cfg.data.max_len)
            log.info("[eval @ %d] base=%.4f lora=%.4f delta=%+.4f",
                     step, m["base_logp"], m["lora_logp"], m["delta_logp"])
            model.train()

    save_checkpoint(ckpt_path, model, optimizer, scheduler, step)

    if stop["flag"] and step < cfg.train.max_steps:
        # Interrupted: leave a resumable checkpoint, no DONE marker, exit 0 so
        # the SLURM chain resubmits and continues.
        log.info("interrupted at step %d; checkpoint saved, exiting for resume", step)
        return {"interrupted": True, "step": step}

    log.info("training complete at step %d; running final eval...", step)
    metrics = evaluate(model, tokenizer, eval_pairs, device, cfg.data.max_len)
    metrics["step"] = step

    # Save the adapter separately so it can be reloaded without the optimizer.
    model.save_pretrained(str(run_dir / "adapter"))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Uniform-LoRA fine-tuning baseline.")
    parser.add_argument("--config", default="configs/lora_baseline.yaml")
    parser.add_argument(
        "--resume", default=None,
        help="Existing run dir to resume from its ckpt.pt (safe after a timeout).",
    )
    args = parser.parse_args()

    cfg = load_baseline_config(args.config)
    set_seed(cfg.train.seed, cfg.train.deterministic)

    resume = args.resume is not None
    if resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            raise SystemExit(f"--resume dir not found: {run_dir}")
    else:
        run_dir = make_run_dir(cfg.output_dir, "lora-baseline")

    log = get_logger("lora_baseline", run_dir)
    write_json(run_dir / "config.json", dataclasses.asdict(cfg))

    done_marker = run_dir / "DONE"
    if done_marker.exists():
        log.info("DONE marker present; nothing to do. Results in %s", run_dir / "results.json")
        return 0

    metrics = train(cfg, run_dir, resume, log)

    if metrics.get("interrupted"):
        return 0  # chained SLURM job will resume

    write_json(run_dir / "results.json", metrics)
    done_marker.write_text("ok\n", encoding="utf-8")

    # The one real number, printed for easy grep from the SLURM log.
    log.info(
        "RESULT n_eval=%d base_logp=%.4f lora_logp=%.4f delta_logp=%+.4f "
        "base_ppl=%.3f lora_ppl=%.3f",
        metrics["n_eval"], metrics["base_logp"], metrics["lora_logp"],
        metrics["delta_logp"], metrics["base_ppl"], metrics["lora_ppl"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
