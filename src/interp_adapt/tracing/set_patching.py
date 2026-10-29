"""
set_patching.py  -  patch SEVERAL heads at once (Stage 1 interaction experiments)

Component: Stage 1 head-interaction analysis (scripts/stage1/entangle.py)
Depends on: patching.py (same hook target, constants and capture format)

Extends OProjPatchSingle from one head to a SET of (layer, head) pairs, which
may sit in different layers. The hook target (o_proj INPUT), the position
(last prompt token) and the dtype handling are identical, so a one-element
set gives the same logits as OProjPatchSingle.

    stored = capture_activations(model, src_ids, src_pos)        # patching.py
    logits = set_patched_logits(model, tgt_ids, tgt_pos,
                                [(21, 2), (27, 6)], stored)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import torch

from .patching import HEAD_DIM, NUM_HEADS, NUM_LAYERS


class OProjPatchSet:
    """Context manager: substitute several heads' o_proj inputs at patch_pos.

    `stored` is the dict returned by patching.capture_activations:
    layer_idx -> Tensor [12, 128].
    """

    def __init__(self, model, patch_pos: int, heads: Iterable[tuple[int, int]],
                 stored: dict[int, torch.Tensor]) -> None:
        self.model = model
        self.patch_pos = patch_pos
        self.stored = stored
        self.by_layer: dict[int, list[int]] = defaultdict(list)
        for layer, head in heads:
            if not (0 <= layer < NUM_LAYERS and 0 <= head < NUM_HEADS):
                raise ValueError(f"head out of range: L{layer}H{head}")
            if head not in self.by_layer[layer]:
                self.by_layer[layer].append(head)
        self.handles: list = []

    def __enter__(self) -> "OProjPatchSet":
        for layer, heads in self.by_layer.items():
            src = {h: self.stored[layer][h] for h in heads}   # [128] each
            pos = self.patch_pos

            def _hook(_module, args, src=src, pos=pos):
                x = args[0].clone()
                for h, v in src.items():
                    x[0, pos, h * HEAD_DIM:(h + 1) * HEAD_DIM] = v.to(
                        device=x.device, dtype=x.dtype)
                return (x,)

            attn = self.model.model.layers[layer].self_attn
            self.handles.append(attn.o_proj.register_forward_pre_hook(_hook))
        return self

    def __exit__(self, *_) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


@torch.no_grad()
def set_patched_logits(
    model,
    input_ids: torch.Tensor,
    patch_pos: int,
    heads: Optional[Iterable[tuple[int, int]]],
    stored: Optional[dict[int, torch.Tensor]],
) -> torch.Tensor:
    """Forward pass with a set of heads patched. Empty/None heads = unpatched."""
    heads = list(heads or [])
    if not heads or stored is None:
        return model(input_ids=input_ids, use_cache=False).logits
    with OProjPatchSet(model, patch_pos, heads, stored):
        return model(input_ids=input_ids, use_cache=False).logits
