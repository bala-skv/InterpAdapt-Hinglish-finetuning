"""Uniform-LoRA fine-tuning baseline -- the Stage-2 "one real number".

This is the baseline the interpretability-guided Circuit-Routing Adapter will be
measured against: a *uniform* LoRA (every targeted head adapted equally) on the
SAIL-2017 Romanized (Hinglish) code-mixed **sentiment classification** task
(``negative`` / ``neutral`` / ``positive``). It reports one comparable number --

    base-vs-LoRA sentiment **accuracy / macro-F1** on the frozen held-out
    validation split, framed as label-word generation and scored by ranking the
    three label verbalizers by teacher-forced log-probability --

using the same scoring machinery as Stage-1 tracing (see src/interp_adapt/tracing/sequence_scoring.py).

Metrics, config and curves are optionally logged to **Weights & Biases** (see the
``wandb:`` config section); the run is resumable so a chained SLURM job continues
the same W&B run.

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

    python scripts/stage2/train_lora_baseline.py --config configs/lora_baseline.yaml
    python scripts/stage2/train_lora_baseline.py --config configs/lora_baseline.yaml \
        --resume runs/lora-baseline-20260928-101500

Component: Stage 2 LoRA baseline
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
    Example,
    LABELS,
    PROMPT_TEMPLATE,
    VERBALIZER,
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
    hf_dataset: Optional[str] = "satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2"
    hf_config: Optional[str] = None
    hf_train_split: str = "train"
    hf_eval_split: str = "validation"
    # Use the de-duplicated sanitized files, NOT the auto-merged default config
    # (which mixes raw/ + sanitized/ and re-introduces the 12 train->eval leaks).
    hf_train_file: Optional[str] = "sanitized/train.txt"
    hf_eval_file: Optional[str] = "sanitized/validation.txt"
    train_size: Optional[int] = None
    eval_size: Optional[int] = None
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
class WandbCfg:
    # Optional experiment tracking. Set enabled=false to turn off entirely; it
    # also degrades gracefully (warn + continue) if wandb is missing or init
    # fails, so a run never dies because of logging.
    enabled: bool = True
    project: str = "babyshark-cra"
    entity: Optional[str] = None       # your W&B team/user; null -> default
    run_name: Optional[str] = None     # null -> the run-dir name
    mode: str = "online"               # online | offline | disabled
    tags: List[str] = field(default_factory=lambda: ["stage2", "lora-baseline", "sail2017"])


@dataclass
class BaselineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    lora: LoraCfg = field(default_factory=LoraCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)
    output_dir: str = "runs"


_SECTIONS = {
    "model": ModelConfig,
    "quant": QuantConfig,
    "lora": LoraCfg,
    "data": DataCfg,
    "train": TrainCfg,
    "wandb": WandbCfg,
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
# Eval metric (3-class sentiment accuracy / macro-F1; label-verbalizer ranking)
# --------------------------------------------------------------------------- #
def _label_logp(model, full_ids: List[int], n_prompt: int, device) -> float:
    """Mean per-token log P(label tokens | prompt), teacher-forced.

    Length-normalized so labels that tokenize into different numbers of pieces
    ("negative"/"neutral"/"positive") are ranked fairly.
    """
    import torch

    n_target = len(full_ids) - n_prompt
    ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logprobs = logits.log_softmax(-1)
    total = 0.0
    for k in range(n_target):
        total += logprobs[n_prompt + k - 1, full_ids[n_prompt + k]].item()
    return total / n_target


def classify(model, tokenizer, ex: Example, device, max_len: int, label_ids: dict) -> str:
    """Predict a label by picking the verbalizer with the highest mean log-prob."""
    prompt_ids = tokenizer.encode(
        PROMPT_TEMPLATE.format(text=ex.text), add_special_tokens=False
    )
    max_label = max(len(t) for t in label_ids.values())
    prompt_ids = prompt_ids[: max_len - max_label]  # always leave room for the label
    n_prompt = len(prompt_ids)

    best_label, best_score = LABELS[0], float("-inf")
    for label in LABELS:
        full_ids = prompt_ids + label_ids[label]
        score = _label_logp(model, full_ids, n_prompt, device)
        if score > best_score:
            best_score, best_label = score, label
    return best_label


def _macro_f1(gold: List[str], pred: List[str]) -> float:
    """Unweighted mean of per-class F1 over the fixed label set."""
    f1s = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def evaluate(model, tokenizer, examples: List[Example], device, max_len: int) -> dict:
    """Base-vs-LoRA sentiment accuracy + macro-F1 over the eval split.

    The base score is read with the adapter disabled, so both numbers come from
    the single adapted model.
    """
    model.eval()
    # Verbalizer token ids reused across every example.
    label_ids = {
        label: tokenizer.encode(VERBALIZER[label], add_special_tokens=False)
        for label in LABELS
    }
    gold = [ex.label for ex in examples]

    def _predict(active: bool) -> List[str]:
        ctx = _adapter_context(model, active)
        with ctx:
            return [classify(model, tokenizer, ex, device, max_len, label_ids)
                    for ex in examples]

    lora_pred = _predict(active=True)
    base_pred = _predict(active=False)
    n = len(examples)

    def _acc(pred):
        return sum(1 for g, p in zip(gold, pred) if g == p) / n if n else float("nan")

    base_acc, lora_acc = _acc(base_pred), _acc(lora_pred)
    return {
        "n_eval": n,
        "base_acc": base_acc,
        "lora_acc": lora_acc,
        "delta_acc": lora_acc - base_acc,
        "base_macro_f1": _macro_f1(gold, base_pred),
        "lora_macro_f1": _macro_f1(gold, lora_pred),
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
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log.info("base loaded: dtype=%s gpu=%s device_cap=%s", cfg.model.dtype, gpu_name, major)

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


def _init_wandb(cfg: BaselineConfig, run_dir: Path, log):
    """Start (or resume) a W&B run for this run dir; None if disabled/unavailable.

    The run id is persisted in ``run_dir/wandb_run_id`` so a chained SLURM resume
    continues the SAME W&B run instead of spawning a new one. Any failure (wandb
    not installed, no API key, offline FS) is downgraded to a warning so logging
    never crashes the training job.
    """
    wcfg = cfg.wandb
    if not wcfg.enabled:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed (pip install wandb); continuing without it.")
        return None

    id_file = run_dir / "wandb_run_id"
    run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
    try:
        run = wandb.init(
            project=wcfg.project,
            entity=wcfg.entity,
            name=wcfg.run_name or run_dir.name,
            id=run_id,
            resume="allow",
            mode=wcfg.mode,
            tags=list(wcfg.tags),
            config=dataclasses.asdict(cfg),
            dir=str(run_dir),
        )
    except Exception as exc:  # noqa: BLE001 - never let logging kill the run
        log.warning("wandb.init failed (%s); continuing without it.", exc)
        return None

    if not id_file.exists():
        id_file.write_text(run_id, encoding="utf-8")
    log.info("wandb run '%s' (project=%s mode=%s id=%s)",
             run.name, wcfg.project, wcfg.mode, run_id)
    return run


def train(cfg: BaselineConfig, run_dir: Path, resume: bool, log) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(cfg.model)

    log.info("loading data...")
    train_ex, eval_ex = load_splits(
        jsonl_train=cfg.data.jsonl_train,
        jsonl_eval=cfg.data.jsonl_eval,
        hf_dataset=cfg.data.hf_dataset,
        hf_config=cfg.data.hf_config,
        hf_train_split=cfg.data.hf_train_split,
        hf_eval_split=cfg.data.hf_eval_split,
        hf_train_file=cfg.data.hf_train_file,
        hf_eval_file=cfg.data.hf_eval_file,
        train_size=cfg.data.train_size,
        eval_size=cfg.data.eval_size,
    )
    log.info("train=%d eval=%d examples", len(train_ex), len(eval_ex))
    if not train_ex or not eval_ex:
        raise SystemExit("empty train/eval split; check the data config")

    encoded = [encode_example(tokenizer, p, cfg.data.max_len) for p in train_ex]
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

    wb = _init_wandb(cfg, run_dir, log)

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

        cur_lr = scheduler.get_last_lr()[0]
        if wb is not None:
            wb.log({"train/loss": loss_val, "train/lr": cur_lr}, step=step)

        if step % 10 == 0:
            log.info("step %4d/%d  loss=%.4f  lr=%.2e",
                     step, cfg.train.max_steps, loss_val, cur_lr)

        if cfg.train.ckpt_every and step % cfg.train.ckpt_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step)
            log.info("checkpoint @ step %d -> %s", step, ckpt_path)

        if cfg.train.eval_every and step % cfg.train.eval_every == 0:
            m = evaluate(model, tokenizer, eval_ex, device, cfg.data.max_len)
            log.info("[eval @ %d] base_acc=%.4f lora_acc=%.4f delta=%+.4f",
                     step, m["base_acc"], m["lora_acc"], m["delta_acc"])
            if wb is not None:
                wb.log({f"eval/{k}": v for k, v in m.items()
                        if isinstance(v, (int, float))}, step=step)
            model.train()

    save_checkpoint(ckpt_path, model, optimizer, scheduler, step)

    if stop["flag"] and step < cfg.train.max_steps:
        # Interrupted: leave a resumable checkpoint, no DONE marker, exit 0 so
        # the SLURM chain resubmits and continues.
        log.info("interrupted at step %d; checkpoint saved, exiting for resume", step)
        if wb is not None:
            wb.finish(exit_code=0)
        return {"interrupted": True, "step": step}

    log.info("training complete at step %d; running final eval...", step)
    metrics = evaluate(model, tokenizer, eval_ex, device, cfg.data.max_len)
    metrics["step"] = step

    # Save the adapter separately so it can be reloaded without the optimizer.
    model.save_pretrained(str(run_dir / "adapter"))

    if wb is not None:
        num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wb.log({f"final/{k}": v for k, v in num_metrics.items()}, step=step)
        wb.summary.update(num_metrics)
        wb.finish()
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
        "RESULT n_eval=%d base_acc=%.4f lora_acc=%.4f delta_acc=%+.4f "
        "base_macro_f1=%.4f lora_macro_f1=%.4f",
        metrics["n_eval"], metrics["base_acc"], metrics["lora_acc"],
        metrics["delta_acc"], metrics["base_macro_f1"], metrics["lora_macro_f1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
