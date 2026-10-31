"""
mlp_routing.py - route a LoRA update through the MLP blocks of selected layers.

The Stage-1 MLP patching (``scripts/stage1/full_trace.py --component mlp``) scores
each layer's whole MLP output, not individual neurons, so the natural routing unit
here is a LAYER: one scalar gate per layer, applied to that layer's MLP LoRA.

A layer gate is implemented with the existing :class:`MaskedLoRALinear` by treating
the output of the projection as ONE "head" (``num_heads=1``, ``head_dim=out_features``,
``head_axis="out"``). Its zero-gate guarantee (a gate of 0 contributes exactly
nothing) and the ``base_pass`` evaluation trick therefore carry over unchanged.

Gate rules mirror the head rules (``masked_lora.py``), over the 28 layers:

* uniform : every layer gets gate 1 (standard LoRA on the MLP);
* soft    : gate_l = max(0, s_l) / max_l s_l;
* topk    : gate_l = 1 for the k layers with the highest score, else 0;
* random  : gate_l = 1 for k layers drawn at random (matched budget to topk).

Layers whose gate is 0 get no adapter at all, so they cost no memory.

Component: Stage 2 masked LoRA (MLP extension)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .masked_lora import MaskedLoRALinear, MaskedLoraSpec

MLP_LEAVES = ("gate_proj", "up_proj", "down_proj")


def load_layer_scores(path: str | Path, num_layers: int) -> List[float]:
    """Read ``{"scores": [s_0, ..., s_{L-1}]}`` (one MLP score per layer)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = data["scores"] if isinstance(data, dict) else data
    if len(scores) != num_layers or any(isinstance(s, (list, tuple)) for s in scores):
        raise ValueError(f"{path}: expected {num_layers} per-layer scores, got "
                         f"{len(scores)} entries (is this a head-score file?)")
    return [float(s) for s in scores]


def uniform_layer_gates(num_layers: int) -> torch.Tensor:
    return torch.ones(num_layers)


def soft_layer_gates(scores: Sequence[float]) -> torch.Tensor:
    s = torch.as_tensor(list(scores), dtype=torch.float32)
    top = float(s.max())
    if top <= 0:
        raise ValueError("soft layer gates need at least one positive score")
    return s.clamp_min(0.0) / top


def topk_layer_gates(scores: Sequence[float], k: int) -> torch.Tensor:
    s = torch.as_tensor(list(scores), dtype=torch.float32)
    if not 0 <= k <= s.numel():
        raise ValueError(f"k={k} out of range for {s.numel()} layers")
    gates = torch.zeros(s.numel())
    if k > 0:
        gates[torch.topk(s, k).indices] = 1.0
    return gates


def random_layer_gates(num_layers: int, k: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if not 0 <= k <= num_layers:
        raise ValueError(f"k={k} out of range for {num_layers} layers")
    gates = torch.zeros(num_layers)
    if k > 0:
        gates[torch.randperm(num_layers, generator=generator)[:k]] = 1.0
    return gates


def build_layer_gates(mode: str, num_layers: int, scores: Optional[Sequence[float]], k: int,
                      seed: int) -> torch.Tensor:
    if mode == "uniform":
        return uniform_layer_gates(num_layers)
    if scores is None:
        raise ValueError(f"mlp mode {mode!r} needs MLP scores (cra.mlp_scores_file)")
    if mode == "soft":
        return soft_layer_gates(scores)
    if mode == "topk":
        return topk_layer_gates(scores, k)
    if mode == "random":
        return random_layer_gates(num_layers, k, torch.Generator().manual_seed(seed))
    raise ValueError(f"unknown mlp mode {mode!r}")


def inject_mlp_lora(model, spec: MaskedLoraSpec, gates: torch.Tensor,
                    modules: Sequence[str] = ("down_proj",)) -> List[MaskedLoRALinear]:
    """Wrap ``model.layers.<l>.mlp.<leaf>`` with a layer-gated LoRA for every layer
    whose gate is non-zero. Does NOT freeze the base (``inject_masked_lora`` does that
    and must run first, even with an empty attention target list)."""
    for leaf in modules:
        if leaf not in MLP_LEAVES:
            raise ValueError(f"unknown MLP module {leaf!r}; choose from {MLP_LEAVES}")
    inserted: List[MaskedLoRALinear] = []
    for layer_idx, gate in enumerate(gates.tolist()):
        if gate == 0.0:
            continue
        mlp = model.get_submodule(f"model.layers.{layer_idx}.mlp")
        for leaf in modules:
            base = getattr(mlp, leaf)
            wrapper = MaskedLoRALinear(
                base, num_heads=1, head_dim=base.out_features, head_axis="out",
                rank=spec.rank, alpha=spec.alpha, dropout=spec.dropout,
                mask=torch.tensor([gate], dtype=torch.float32),
            )
            setattr(mlp, leaf, wrapper)
            inserted.append(wrapper)
    return inserted


def effective_params(w: MaskedLoRALinear) -> int:
    """LoRA parameters that can receive a gradient under the current gate.

    out-axis wrapper (q_proj, MLP): A is trained whole, B only for gated rows;
    in-axis wrapper (o_proj): B is trained whole, A only for gated columns.
    A wrapper whose gate is entirely zero trains nothing.
    """
    if w.active_heads == 0:
        return 0
    gated = w.rank * w.active_heads * w.head_dim
    ungated = w.rank * (w.base.in_features if w.head_axis == "out" else w.base.out_features)
    return gated + ungated


def total_effective_params(wrappers: Sequence[MaskedLoRALinear]) -> int:
    return sum(effective_params(w) for w in wrappers)


def split_wrappers(wrappers: Sequence[MaskedLoRALinear]) -> Dict[str, List[MaskedLoRALinear]]:
    """Attention wrappers (num_heads > 1) versus MLP layer-gate wrappers (num_heads == 1)."""
    return {"attn": [w for w in wrappers if w.num_heads > 1],
            "mlp": [w for w in wrappers if w.num_heads == 1]}
