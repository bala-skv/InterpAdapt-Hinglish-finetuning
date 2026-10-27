"""Stage-2 Circuit-Routing Adapter vs. controls -- the matched-budget comparison.

This trains ONE routing arm per invocation (chosen by ``cra.mask_mode``) on the
same frozen SAIL-2017 Romanized (Hinglish) sentiment task as the uniform-LoRA
baseline, and reports the same comparable number (base-vs-adapter accuracy /
macro-F1). Run it once per arm and read the ``RESULT`` lines / W&B side by side:

* ``uniform`` -- all-ones mask == a standard LoRA (reproduces the baseline; sanity).
* ``soft``    -- soft scaling ``M = f(s_i)`` from Daniel's Stage-1 causal scores
                 (novelty *(iii)*: the mask gates *inside* the LoRA update).
* ``topk``    -- hard top-k: adapt only the ``k`` highest-scoring heads per layer.
* ``random``  -- the KEY control: ``k`` random heads per layer at the SAME budget
                 as ``topk`` (tests the *sign* of the causal signal, not sparsity).

The adapter surface is :class:`circuit_routing.masked_lora.MaskedLoRALinear`
(unit-tested in ``scripts/test_masked_lora.py``), injected into every ``q_proj``
and ``o_proj``. ``top-k`` and ``random`` are budget-matched by construction --
:func:`adapted_budget` is asserted equal and logged, so a reviewer can verify the
comparison is fair.

Stage-1 scores file (``cra.scores_file``), required for soft/topk/random, is JSON::

    {"scores": [[<head0>, <head1>, ...], ...]}   # shape [num_layers][num_q_heads]

i.e. one causal score per (layer, query-head). ``uniform`` needs no scores.

Resumable + self-chaining exactly like the baseline (atomic ckpt to a durable run
dir, SIGUSR1 graceful save, DONE marker). Base vs. adapter eval both come from the
one model: the base pass simply zeroes every head mask, so there is no second
model to load.

Usage::

    python scripts/train_cra_compare.py --config configs/cra_compare.yaml
    python scripts/train_cra_compare.py --config configs/cra_compare.yaml \
        --mask-mode random --resume runs/cra-random-20260929-101500

Owner: Jawed.
"""
import _bootstrap  # noqa: F401

import argparse
import contextlib
import dataclasses
import json
import os
import signal
import tempfile
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional

from circuit_routing.config import ModelConfig, QuantConfig
from circuit_routing.data import (
    LABELS,
    PROMPT_TEMPLATE,
    VERBALIZER,
    Example,
    collate_batch,
    encode_example,
    load_splits,
)
from circuit_routing.logging_utils import get_logger, make_run_dir, write_json
from circuit_routing.masked_lora import (
    MaskedLoRALinear,
    MaskedLoraSpec,
    adapted_budget,
    inject_masked_lora,
    random_mask,
    random_mask_global,
    soft_mask_from_scores,
    soft_mask_from_scores_global,
    topk_mask_from_scores,
    topk_mask_from_scores_global,
    uniform_mask,
)
from circuit_routing.model_loading import load_model, load_tokenizer, verify_base_checkpoint
from circuit_routing.seeding import set_seed

_MASK_MODES = ("uniform", "soft", "topk", "random")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class CraCfg:
    mask_mode: str = "soft"                       # uniform | soft | topk | random
    # PRIMARY scope is "global": one scale/threshold shared across all L*H heads
    # (a noisy layer's best head is NOT forced to a full gate). "per_layer" is
    # the ablation control (per-layer min-max soft / per-layer top-k).
    scope: str = "global"                         # global | per_layer
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "o_proj"])
    # Stage-1 per-head causal scores; required for soft/topk/random.
    scores_file: Optional[str] = None
    # per_layer topk/random: heads to keep PER LAYER (matched budget).
    k: int = 4
    # global topk/random: heads to keep across ALL L*H heads (matched budget).
    k_total: int = 112
    # soft: min-max floor for f(s_i) in [floor, 1]; normalize=False -> raw scores.
    soft_floor: float = 0.0
    soft_normalize: bool = True
    # random: seed for the matched-budget random head set (report it).
    random_seed: int = 0


@dataclass
class DataCfg:
    jsonl_train: Optional[str] = None
    jsonl_eval: Optional[str] = None
    hf_dataset: Optional[str] = "satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2"
    hf_config: Optional[str] = None
    hf_train_split: str = "train"
    hf_eval_split: str = "validation"
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
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 30
    max_steps: int = 600
    ckpt_every: int = 100
    eval_every: int = 200
    max_grad_norm: float = 1.0


@dataclass
class WandbCfg:
    enabled: bool = True
    project: str = "babyshark-cra"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    mode: str = "online"
    tags: List[str] = field(default_factory=lambda: ["stage2", "cra", "sail2017"])


@dataclass
class CraConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    cra: CraCfg = field(default_factory=CraCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)
    output_dir: str = "runs"


_SECTIONS = {
    "model": ModelConfig,
    "quant": QuantConfig,
    "cra": CraCfg,
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


def load_cra_config(path: Optional[str]) -> CraConfig:
    if path is None:
        return CraConfig()
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - set(_SECTIONS) - {"output_dir"}
    if unknown:
        raise ValueError(f"Unknown top-level sections: {sorted(unknown)}")
    return CraConfig(
        output_dir=raw.get("output_dir", "runs"),
        **{name: _section(cls, raw.get(name)) for name, cls in _SECTIONS.items()},
    )


# --------------------------------------------------------------------------- #
# Stage-1 scores -> per-module head masks
# --------------------------------------------------------------------------- #
def load_layer_head_scores(path: str, num_layers: int, num_heads: int) -> List[List[float]]:
    """Load Daniel's per-(layer, query-head) causal scores as ``[L][H]``.

    Accepts either ``{"scores": [[...], ...]}`` or a bare 2D list. Validates the
    shape against the verified head layout so a malformed scores file fails loud
    *before* GPU time is spent.
    """
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    scores = obj["scores"] if isinstance(obj, dict) else obj
    if len(scores) != num_layers:
        raise ValueError(
            f"scores has {len(scores)} layers, model has {num_layers}"
        )
    for li, row in enumerate(scores):
        if len(row) != num_heads:
            raise ValueError(
                f"scores[{li}] has {len(row)} heads, expected {num_heads}"
            )
    return [[float(v) for v in row] for row in scores]


def build_head_masks(cfg: CraConfig, layout, log) -> Optional[Dict[str, "object"]]:
    """Map each targeted module's full name -> its length-``num_heads`` mask.

    ``uniform`` returns ``None`` (every wrapper defaults to all-ones). soft/topk/
    random read the Stage-1 scores and build one mask per (layer, projection). A
    layer's q_proj and o_proj share that layer's head scores, so the same heads
    are routed on both the query and the output projection.

    Scope is PRIMARY ``global`` (one scale/threshold across all ``L*H`` heads) or
    the ``per_layer`` ablation control.
    """
    mode = cfg.cra.mask_mode
    if mode == "uniform":
        return None
    if mode not in _MASK_MODES:
        raise ValueError(f"cra.mask_mode must be one of {_MASK_MODES}, got {mode!r}")
    if not cfg.cra.scores_file:
        raise ValueError(f"cra.mask_mode={mode!r} requires cra.scores_file (Stage-1 scores)")

    import torch

    scope = cfg.cra.scope
    if scope not in ("global", "per_layer"):
        raise ValueError(f"cra.scope must be 'global' or 'per_layer', got {scope!r}")

    scores = load_layer_head_scores(
        cfg.cra.scores_file, layout.num_layers, layout.num_attention_heads
    )
    n_heads = layout.num_attention_heads

    if scope == "global":
        if mode == "soft":
            grid = soft_mask_from_scores_global(scores, floor=cfg.cra.soft_floor)
        elif mode == "topk":
            grid = topk_mask_from_scores_global(scores, cfg.cra.k_total)
        else:  # random
            gen = torch.Generator().manual_seed(cfg.cra.random_seed)
            grid = random_mask_global(layout.num_layers, n_heads, cfg.cra.k_total, generator=gen)
        per_layer = [grid[li] for li in range(layout.num_layers)]
        log.info(
            "built GLOBAL %s masks | active_heads=%d/%d (%s)",
            mode, int((grid != 0).sum().item()), grid.numel(),
            f"k_total={cfg.cra.k_total}" if mode in ("topk", "random") else "dense soft",
        )
    else:  # per_layer ablation control
        k = cfg.cra.k
        gen = torch.Generator().manual_seed(cfg.cra.random_seed) if mode == "random" else None

        def _mask_for_layer(row: List[float]):
            if mode == "soft":
                return soft_mask_from_scores(
                    row, normalize=cfg.cra.soft_normalize, floor=cfg.cra.soft_floor
                )
            if mode == "topk":
                return topk_mask_from_scores(row, k)
            return random_mask(n_heads, k, generator=gen)

        per_layer = [_mask_for_layer(row) for row in scores]
        log.info(
            "built PER-LAYER %s masks for %d layers (%s), k=%s per layer",
            mode, layout.num_layers, cfg.cra.target_modules,
            k if mode in ("topk", "random") else "dense",
        )

    # Attach the layer mask to both head-aligned projections in that layer.
    masks: Dict[str, object] = {}
    for name, leaf, layer_idx in _iter_layer_projections(cfg.cra.target_modules, layout.num_layers):
        masks[name] = per_layer[layer_idx]
    return masks


def _iter_layer_projections(target_modules, num_layers):
    """Yield ``(full_name, leaf, layer_idx)`` for Qwen2 decoder projections.

    Qwen2 attention modules live at ``model.layers.<L>.self_attn.<proj>``.
    """
    for layer_idx in range(num_layers):
        for leaf in target_modules:
            name = f"model.layers.{layer_idx}.self_attn.{leaf}"
            yield name, leaf, layer_idx


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_cra_model(cfg: CraConfig, tokenizer, log):
    """Load the pinned fp16 base and inject the masked-LoRA routing adapter.

    Returns ``(model, wrappers, layout)``. Only the wrappers' low-rank factors
    train; the base stays frozen.
    """
    import torch

    model = load_model(cfg.model, cfg.quant, quant_mode="none", device_map="auto")
    layout = verify_base_checkpoint(cfg.model, model.config, tokenizer)

    head_masks = build_head_masks(cfg, layout, log)
    spec = MaskedLoraSpec(
        rank=cfg.cra.rank,
        alpha=cfg.cra.alpha,
        dropout=cfg.cra.dropout,
        target_modules=tuple(cfg.cra.target_modules),
    )
    wrappers = inject_masked_lora(model, layout, spec, head_masks=head_masks)
    if not wrappers:
        raise SystemExit(
            f"no target modules {cfg.cra.target_modules} were wrapped; check the model"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # Fail fast if the base was not frozen: LoRA should train <<1% of params.
    # (A regression here silently trains all ~1.5B params -> AdamW state -> OOM.)
    if trainable > 0.01 * total:
        raise SystemExit(
            f"expected LoRA-only training (<1% of params) but {100.0 * trainable / total:.2f}% "
            f"are trainable ({trainable}/{total}) -- the base model was not frozen"
        )
    budget = adapted_budget(wrappers)
    active = sum(w.active_heads for w in wrappers)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None
    log.info(
        "CRA[%s/%s] r=%d on %s | wrappers=%d active_heads=%d budget=%d | "
        "trainable %d / %d (%.3f%%) | gpu=%s cap=%s",
        cfg.cra.mask_mode, cfg.cra.scope, cfg.cra.rank, cfg.cra.target_modules, len(wrappers),
        active, budget, trainable, total, 100.0 * trainable / total, gpu_name, major,
    )
    return model, wrappers, layout


@contextlib.contextmanager
def base_pass(wrappers: List[MaskedLoRALinear]):
    """Temporarily zero every head mask so the model emits its FROZEN base output.

    Used to read the base score from the single adapted model (no second load).
    """
    import torch

    saved = [w.head_mask.clone() for w in wrappers]
    try:
        for w in wrappers:
            w.set_head_mask(torch.zeros(w.num_heads))
        yield
    finally:
        for w, m in zip(wrappers, saved):
            w.set_head_mask(m)


# --------------------------------------------------------------------------- #
# Eval (base vs CRA sentiment accuracy / macro-F1; verbalizer ranking)
# --------------------------------------------------------------------------- #
def _label_logp(model, full_ids: List[int], n_prompt: int, device) -> float:
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
    prompt_ids = tokenizer.encode(
        PROMPT_TEMPLATE.format(text=ex.text), add_special_tokens=False
    )
    max_label = max(len(t) for t in label_ids.values())
    prompt_ids = prompt_ids[: max_len - max_label]
    n_prompt = len(prompt_ids)

    best_label, best_score = LABELS[0], float("-inf")
    for label in LABELS:
        full_ids = prompt_ids + label_ids[label]
        score = _label_logp(model, full_ids, n_prompt, device)
        if score > best_score:
            best_score, best_label = score, label
    return best_label


def _macro_f1(gold: List[str], pred: List[str]) -> float:
    f1s = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def evaluate(model, wrappers, tokenizer, examples: List[Example], device, max_len: int) -> dict:
    """Base-vs-CRA accuracy + macro-F1. The base pass zeroes every head mask."""
    model.eval()
    label_ids = {
        label: tokenizer.encode(VERBALIZER[label], add_special_tokens=False)
        for label in LABELS
    }
    gold = [ex.label for ex in examples]

    def _predict() -> List[str]:
        return [classify(model, tokenizer, ex, device, max_len, label_ids) for ex in examples]

    cra_pred = _predict()
    with base_pass(wrappers):
        base_pred = _predict()
    n = len(examples)

    def _acc(pred):
        return sum(1 for g, p in zip(gold, pred) if g == p) / n if n else float("nan")

    base_acc, cra_acc = _acc(base_pred), _acc(cra_pred)
    return {
        "n_eval": n,
        "base_acc": base_acc,
        "cra_acc": cra_acc,
        "delta_acc": cra_acc - base_acc,
        "base_macro_f1": _macro_f1(gold, base_pred),
        "cra_macro_f1": _macro_f1(gold, cra_pred),
    }


# --------------------------------------------------------------------------- #
# Checkpointing (atomic; only the low-rank factors + optimizer state)
# --------------------------------------------------------------------------- #
def _adapter_state(model) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int) -> None:
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
    import torch

    ckpt = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in ckpt["adapter"].items():
            if name in params:
                params[name].copy_(tensor.to(params[name].device, params[name].dtype))
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
    return int(ckpt["step"])


# --------------------------------------------------------------------------- #
# W&B (same graceful-degradation pattern as the baseline)
# --------------------------------------------------------------------------- #
def _init_wandb(cfg: CraConfig, run_dir: Path, log):
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
    tags = list(wcfg.tags) + [f"mask:{cfg.cra.mask_mode}"]
    try:
        run = wandb.init(
            project=wcfg.project,
            entity=wcfg.entity,
            name=wcfg.run_name or run_dir.name,
            id=run_id,
            resume="allow",
            mode=wcfg.mode,
            tags=tags,
            config=dataclasses.asdict(cfg),
            dir=str(run_dir),
        )
    except Exception as exc:  # noqa: BLE001 - logging must never kill training
        log.warning("wandb.init failed (%s); continuing without it.", exc)
        return None

    if not id_file.exists():
        id_file.write_text(run_id, encoding="utf-8")
    log.info("wandb run '%s' (project=%s mode=%s id=%s)",
             run.name, wcfg.project, wcfg.mode, run_id)
    return run


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: CraConfig, run_dir: Path, resume: bool, log) -> dict:
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

    model, wrappers, layout = build_cra_model(cfg, tokenizer, log)
    budget = adapted_budget(wrappers)
    active_heads = sum(w.active_heads for w in wrappers)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    # Snapshot the matched-budget accounting for the write-up / reviewer.
    write_json(run_dir / "budget.json", {
        "mask_mode": cfg.cra.mask_mode,
        "scope": cfg.cra.scope,
        "adapted_budget": budget,
        "active_heads": active_heads,
        "num_wrappers": len(wrappers),
        "rank": cfg.cra.rank,
        "k_per_layer": cfg.cra.k if cfg.cra.scope == "per_layer" and cfg.cra.mask_mode in ("topk", "random") else None,
        "k_total": cfg.cra.k_total if cfg.cra.scope == "global" and cfg.cra.mask_mode in ("topk", "random") else None,
        "gpu": gpu_name,
    })

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

    stop = {"flag": False}

    def _on_usr1(signum, frame):
        log.info("SIGUSR1 received -> graceful checkpoint and exit")
        stop["flag"] = True

    signal.signal(signal.SIGUSR1, _on_usr1)

    log.info("training CRA[%s] %d -> %d steps (bs=%d x accum=%d) budget=%d active_heads=%d",
             cfg.cra.mask_mode, start_step, cfg.train.max_steps, cfg.train.batch_size,
             cfg.train.grad_accum_steps, budget, active_heads)

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
            m = evaluate(model, wrappers, tokenizer, eval_ex, device, cfg.data.max_len)
            log.info("[eval @ %d] base_acc=%.4f cra_acc=%.4f delta=%+.4f",
                     step, m["base_acc"], m["cra_acc"], m["delta_acc"])
            if wb is not None:
                wb.log({f"eval/{k}": v for k, v in m.items()
                        if isinstance(v, (int, float))}, step=step)
            model.train()

    save_checkpoint(ckpt_path, model, optimizer, scheduler, step)

    if stop["flag"] and step < cfg.train.max_steps:
        log.info("interrupted at step %d; checkpoint saved, exiting for resume", step)
        if wb is not None:
            wb.finish(exit_code=0)
        return {"interrupted": True, "step": step}

    log.info("training complete at step %d; running final eval...", step)
    metrics = evaluate(model, wrappers, tokenizer, eval_ex, device, cfg.data.max_len)
    metrics.update({
        "step": step,
        "mask_mode": cfg.cra.mask_mode,
        "scope": cfg.cra.scope,
        "adapted_budget": budget,
        "active_heads": active_heads,
        "gpu": gpu_name,
    })

    if wb is not None:
        num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wb.log({f"final/{k}": v for k, v in num_metrics.items()}, step=step)
        wb.summary.update({**num_metrics, "mask_mode": cfg.cra.mask_mode,
                           "scope": cfg.cra.scope, "gpu": gpu_name})
        wb.finish()
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-2 CRA vs. matched-budget controls.")
    parser.add_argument("--config", default="configs/cra_compare.yaml")
    parser.add_argument(
        "--mask-mode", default=None, choices=_MASK_MODES,
        help="Override cra.mask_mode (uniform|soft|topk|random) for this run.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Existing run dir to resume from its ckpt.pt (safe after a timeout).",
    )
    args = parser.parse_args()

    cfg = load_cra_config(args.config)
    if args.mask_mode:
        cfg.cra.mask_mode = args.mask_mode
    set_seed(cfg.train.seed, cfg.train.deterministic)

    resume = args.resume is not None
    if resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            raise SystemExit(f"--resume dir not found: {run_dir}")
    else:
        run_dir = make_run_dir(cfg.output_dir, f"cra-{cfg.cra.mask_mode}")

    log = get_logger("cra_compare", run_dir)
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

    log.info(
        "RESULT mask_mode=%s scope=%s n_eval=%d base_acc=%.4f cra_acc=%.4f delta_acc=%+.4f "
        "base_macro_f1=%.4f cra_macro_f1=%.4f budget=%d active_heads=%d gpu=%s",
        metrics["mask_mode"], metrics["scope"], metrics["n_eval"], metrics["base_acc"], metrics["cra_acc"],
        metrics["delta_acc"], metrics["base_macro_f1"], metrics["cra_macro_f1"],
        metrics["adapted_budget"], metrics["active_heads"], metrics["gpu"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
