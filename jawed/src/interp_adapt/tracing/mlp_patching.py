"""
mlp_patching.py  -  Stage 1 activation patching for the per-layer MLP block

Component: Stage 1 activation patching (MLP companion to patching.py)
Task:       Stage 1  (same gate: stage1_sanity_checks.py must PASS before heatmap)
Depends on: src/interp_adapt/tracing/model_loading.py  (pinned fp16 loader)
            src/circuit_routing/hooks.py          (verified module layout)

Why this file
-------------
``patching.py`` localizes the *attention* circuit by patching each head's
``o_proj`` INPUT. The MLP block has **no head partition** -- it is one dense
feed-forward per layer -- so the natural causal unit here is the WHOLE MLP
output (what the block writes back into the residual stream). This module is the
per-layer analogue of ``OProj*`` so we can ask, alongside the 336-head heatmap,
"how much of the clean->corrupted gap does layer L's MLP carry?".

Design contract (mirrors patching.py)
--------------------------------------
  Hook target  : layer.mlp OUTPUT  (post-hook; = down_proj output, the residual
                 contribution of the MLP, taken AFTER post_attention_layernorm)
  Granularity  : one [1536] vector per layer at patch_pos (no head reshape)
  patch_pos    : index of the LAST PROMPT TOKEN (from score_sequence)
  Capture dtype: fp16  (match the attention trace; NF4 noise smears scores)

Three-run protocol (identical shape to patching.py)
---------------------------------------------------
  stored           = capture_mlp_activations(model, hi_ids, patch_pos)
  logits_corrupted = mlp_patched_logits(model, en_ids, patch_pos, None, None)
  logits_patched   = mlp_patched_logits(model, en_ids, patch_pos, L, stored)
  # Recovery computed by sequence_scoring.recovery(), not here.

"Our current heads"
-------------------
To probe the MLPs that sit in the SAME layers as the Stage-2 top-k heads, pass
those layer indices to :class:`MLPPatchLayers` (e.g. the layers appearing in
stage1_head_scores.json's top set). That isolates the MLP contribution of the
layers your attention routing already trusts.

Architecture constants  (verified on the reference GPU environment)
--------------------------------------------------------------------
  28 layers, hidden 1536, MLP output width == hidden == 1536
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch

NUM_LAYERS = 28
HIDDEN_SIZE = 1536
_CAPTURE_DTYPE = torch.float16  # fp16 trace; change only with team sign-off


def _mlp_output_tensor(output):
    """Qwen2MLP.forward returns a plain Tensor; be tolerant of a tuple wrap."""
    return output[0] if isinstance(output, tuple) else output


class MLPCapture:
    """Context manager: records each layer's MLP OUTPUT at patch_pos (read-only).

    Captured per layer: Tensor [1536] fp16 CPU
    """

    def __init__(self, model, patch_pos: int) -> None:
        self.model = model
        self.patch_pos = patch_pos
        self.handles: list = []
        self.activations: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        pos = self.patch_pos

        def _hook(_module, _args, output):
            y = _mlp_output_tensor(output)  # [B, T, 1536]
            self.activations[layer_idx] = (
                y[0, pos].detach().to(dtype=_CAPTURE_DTYPE).cpu()
            )

        return _hook

    def __enter__(self) -> "MLPCapture":
        for idx, layer in enumerate(self.model.model.layers):
            self.handles.append(
                layer.mlp.register_forward_hook(self._make_hook(idx))
            )
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


class MLPPatchSingle:
    """Substitutes ONE layer's MLP output at patch_pos."""

    def __init__(self, model, patch_pos: int, layer: int,
                 stored: dict[int, torch.Tensor]) -> None:
        self.model = model
        self.patch_pos = patch_pos
        self.layer = layer
        self.stored = stored
        self.handles: list = []

    def __enter__(self) -> "MLPPatchSingle":
        src = self.stored[self.layer]  # [1536] fp16 CPU
        pos = self.patch_pos
        target = self.model.model.layers[self.layer]

        def _hook(_module, _args, output):
            y = _mlp_output_tensor(output).clone()
            y[0, pos, :] = src.to(device=y.device, dtype=y.dtype)
            return y

        self.handles.append(target.mlp.register_forward_hook(_hook))
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


class MLPPatchLayers:
    """Patches the MLP output of a GIVEN SET of layers at patch_pos.

    Pass the layer indices that host your current top-k heads to probe only the
    MLPs co-located with the attention circuit you already route.
    """

    def __init__(self, model, patch_pos: int, layers: Iterable[int],
                 stored: dict[int, torch.Tensor]) -> None:
        self.model = model
        self.patch_pos = patch_pos
        self.layers = sorted(set(int(l) for l in layers))
        self.stored = stored
        self.handles: list = []

    def _make_hook(self, layer_idx: int):
        src = self.stored[layer_idx]  # [1536]
        pos = self.patch_pos

        def _hook(_module, _args, output):
            y = _mlp_output_tensor(output).clone()
            y[0, pos, :] = src.to(device=y.device, dtype=y.dtype)
            return y

        return _hook

    def __enter__(self) -> "MLPPatchLayers":
        for idx in self.layers:
            layer = self.model.model.layers[idx]
            self.handles.append(layer.mlp.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


class MLPPatchAll:
    """Patches ALL 28 layers' MLP outputs (sanity check: recovery ~ 1.0)."""

    def __init__(self, model, patch_pos: int,
                 stored: dict[int, torch.Tensor]) -> None:
        self.model = model
        self.patch_pos = patch_pos
        self.stored = stored
        self.handles: list = []

    def _make_hook(self, layer_idx: int):
        src = self.stored[layer_idx]
        pos = self.patch_pos

        def _hook(_module, _args, output):
            y = _mlp_output_tensor(output).clone()
            y[0, pos, :] = src.to(device=y.device, dtype=y.dtype)
            return y

        return _hook

    def __enter__(self) -> "MLPPatchAll":
        for idx, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.mlp.register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *_) -> bool:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


@torch.no_grad()
def capture_mlp_activations(
    model,
    input_ids: torch.Tensor,
    patch_pos: int,
) -> dict[int, torch.Tensor]:
    """Run one forward pass; return each layer's MLP OUTPUT at patch_pos.

    Returns
    -------
    dict  layer_idx -> Tensor [1536]  fp16  CPU
    """
    with MLPCapture(model, patch_pos) as cap:
        model(input_ids=input_ids, use_cache=False)
    if len(cap.activations) != NUM_LAYERS:
        raise RuntimeError(
            f"Expected {NUM_LAYERS} layers captured, got {len(cap.activations)}."
        )
    return cap.activations


@torch.no_grad()
def mlp_patched_logits(
    model,
    input_ids: torch.Tensor,
    patch_pos: int,
    layer: Optional[int],
    stored: Optional[dict[int, torch.Tensor]],
) -> torch.Tensor:
    """Forward pass with optional single-layer MLP-output patching.

    Pass layer=None / stored=None for an unpatched (corrupted) run.

    Returns logits [B, T, vocab_size].
    """
    if layer is None or stored is None:
        return model(input_ids=input_ids, use_cache=False).logits
    with MLPPatchSingle(model, patch_pos, layer, stored):
        return model(input_ids=input_ids, use_cache=False).logits
