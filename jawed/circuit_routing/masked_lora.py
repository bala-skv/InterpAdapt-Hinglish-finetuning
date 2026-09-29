"""Per-head masked LoRA -- the Stage-2 Circuit-Routing Adapter surface.

The proposal's core mechanism is a LoRA update whose per-head contribution is
gated by an interpretability-derived mask ``M`` (novelty *(iii)*: the mask acts
*inside* the LoRA update, not on the base weights)::

    h = W0 x + M (dot) (B A x)                                       (soft scaling)

This module implements that gate for the two head-aligned projections the mask
targets:

* ``q_proj``  -- output is ``[.., Hq * head_dim]``; the head partition is on the
  **output** axis, so the mask scales the LoRA delta per output head
  (``head_axis="out"``).
* ``o_proj``  -- input is ``[.., Hq * head_dim]`` (the concatenated attention
  heads); the head partition is on the **input** axis, so the mask gates each
  head's contribution *before* the down-projection (``head_axis="in"``).

One wrapper covers every routing variant the mid-submission compares, chosen only
by how the length-``num_heads`` mask vector is built:

* **uniform** -- ``mask = 1`` everywhere  -> identical to a standard LoRA (the
  baseline surface, so CRA and the baseline share weights up to the mask).
* **soft**    -- ``mask = f(s_i)`` from the Stage-1 causal scores ``s_i``.
* **hard top-k** -- binary mask keeping the ``k`` highest-scoring heads.
* **random**  -- binary mask over a random head set at the *same* sparsity as
  top-k (the key matched-budget control).

The zeroed-head guarantee -- a head whose mask entry is ``0`` contributes exactly
zero to the LoRA output -- is what makes the "random mask at matched budget"
comparison meaningful, and is unit-tested in
``scripts/test_masked_lora.py``.

Owner: Jawed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn

from .model_loading import HeadLayout

# Which axis of a projection carries the attention-head partition.
_HEAD_AXIS = {"q_proj": "out", "o_proj": "in"}


def _expand_head_mask(mask: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``[num_heads]`` -> ``[num_heads * head_dim]`` by repeating each head entry."""
    if mask.shape != (num_heads,):
        raise ValueError(f"head mask must be shape ({num_heads},), got {tuple(mask.shape)}")
    return mask.repeat_interleave(head_dim)


# --------------------------------------------------------------------------- #
# Mask builders (length == num query heads; one scalar gate per head)
# --------------------------------------------------------------------------- #
def uniform_mask(num_heads: int) -> torch.Tensor:
    """All-ones mask -> the wrapper behaves as a standard (uniform) LoRA."""
    return torch.ones(num_heads)


def soft_mask_from_scores(
    scores: Sequence[float],
    *,
    normalize: bool = True,
    floor: float = 0.0,
) -> torch.Tensor:
    """Continuous per-head gate ``M = f(s_i)`` for soft scaling.

    ``f`` here is min-max scaling to ``[floor, 1]`` (monotone in ``s_i``, as the
    proposal requires while leaving ``f`` otherwise unspecified). Pass
    ``normalize=False`` to use the raw scores as the gate.
    """
    s = torch.as_tensor(list(scores), dtype=torch.float32)
    if not normalize:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return torch.full_like(s, 1.0)
    scaled = (s - lo) / (hi - lo)
    return floor + (1.0 - floor) * scaled


def topk_mask_from_scores(scores: Sequence[float], k: int) -> torch.Tensor:
    """Binary mask (hard top-k): 1 on the ``k`` highest-scoring heads, else 0."""
    s = torch.as_tensor(list(scores), dtype=torch.float32)
    n = s.numel()
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for {n} heads")
    mask = torch.zeros(n)
    if k > 0:
        top = torch.topk(s, k).indices
        mask[top] = 1.0
    return mask


def random_mask(num_heads: int, k: int, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Binary mask over ``k`` random heads -- the matched-sparsity control.

    Use the SAME ``k`` as :func:`topk_mask_from_scores` so the random-mask
    baseline and the causal mask adapt an equal number of heads (matched budget).
    """
    if not 0 <= k <= num_heads:
        raise ValueError(f"k={k} out of range for {num_heads} heads")
    mask = torch.zeros(num_heads)
    if k > 0:
        perm = torch.randperm(num_heads, generator=generator)[:k]
        mask[perm] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
class MaskedLoRALinear(nn.Module):
    """Frozen base ``nn.Linear`` + a per-head-masked LoRA delta.

    Parameters
    ----------
    base : the original (frozen) projection to wrap. Its weights are never
        modified; only the low-rank ``A``/``B`` factors train.
    num_heads, head_dim : the attention-head partition of the masked axis.
    head_axis : ``"out"`` for ``q_proj`` (mask the delta's output heads),
        ``"in"`` for ``o_proj`` (mask the input heads before ``A``).
    rank, alpha, dropout : standard LoRA hyper-parameters. Scaling is
        ``alpha / rank``.
    mask : optional initial length-``num_heads`` gate (defaults to uniform / all
        ones). Stored as a non-trainable buffer -- the mask comes from Stage-1
        scores, it is not learned.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        num_heads: int,
        head_dim: int,
        head_axis: str,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if head_axis not in ("in", "out"):
            raise ValueError(f"head_axis must be 'in' or 'out', got {head_axis!r}")
        masked_dim = base.in_features if head_axis == "in" else base.out_features
        if masked_dim != num_heads * head_dim:
            raise ValueError(
                f"{head_axis}_features={masked_dim} != num_heads*head_dim="
                f"{num_heads * head_dim}"
            )

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_axis = head_axis
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = base.in_features, base.out_features
        device = base.weight.device
        # The low-rank factors train in fp32 even when the base is fp16. On an
        # fp16 base, fp16 LoRA weights overflow within a few steps -> inf/nan
        # grads that clip_grad_norm cannot recover -> loss=nan forever. This
        # mirrors PEFT's autocast_adapter_dtype=True (why the uniform-LoRA
        # baseline is stable); the forward runs the delta in fp32 and casts back.
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32, device=device))
        # Standard LoRA init: A ~ Kaiming, B = 0 -> delta starts at exactly 0.
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        if mask is None:
            mask = uniform_mask(num_heads)
        self.register_buffer("head_mask", mask.to(dtype=torch.float32, device=device))

    # -- mask management ---------------------------------------------------- #
    def set_head_mask(self, mask: torch.Tensor) -> None:
        """Replace the per-head gate (length ``num_heads``)."""
        if mask.shape != (self.num_heads,):
            raise ValueError(f"mask must be ({self.num_heads},), got {tuple(mask.shape)}")
        self.head_mask = mask.to(dtype=torch.float32, device=self.head_mask.device)

    @property
    def active_heads(self) -> int:
        """Number of heads with a non-zero gate (the adapted-head count)."""
        return int((self.head_mask != 0).sum().item())

    # -- forward ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Run the LoRA path in fp32 (factors + mask are fp32) for stability on an
        # fp16 base, then cast the delta back to the base output dtype before add.
        xf = x.to(self.lora_A.dtype)
        mask = self.head_mask

        if self.head_axis == "in":
            # Gate each input head BEFORE the low-rank projection: a zeroed head
            # never enters A, so it contributes nothing to the delta.
            in_gate = _expand_head_mask(mask, self.num_heads, self.head_dim)
            xd = self.dropout(xf) * in_gate
            delta = torch.nn.functional.linear(xd, self.lora_A)          # [.., rank]
            delta = torch.nn.functional.linear(delta, self.lora_B)       # [.., out]
        else:  # "out": M (dot) (B A x), gate the delta per OUTPUT head.
            delta = torch.nn.functional.linear(self.dropout(xf), self.lora_A)
            delta = torch.nn.functional.linear(delta, self.lora_B)
            out_gate = _expand_head_mask(mask, self.num_heads, self.head_dim)
            delta = delta * out_gate

        return base_out + (self.scaling * delta).to(base_out.dtype)


# --------------------------------------------------------------------------- #
# Injection + matched-budget accounting
# --------------------------------------------------------------------------- #
@dataclass
class MaskedLoraSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Sequence[str] = ("q_proj", "o_proj")


def _iter_target_linears(model, target_modules: Sequence[str]):
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in target_modules and isinstance(module, nn.Linear):
            yield name, module


def inject_masked_lora(
    model,
    layout: HeadLayout,
    spec: MaskedLoraSpec,
    *,
    head_masks: Optional[dict] = None,
) -> List[MaskedLoRALinear]:
    """Replace every targeted ``nn.Linear`` in ``model`` with a masked LoRA.

    ``head_masks`` optionally maps a module's full name to its length-``num_heads``
    mask; modules with no entry start uniform (all ones). Returns the list of
    inserted wrappers so callers can set masks or read ``active_heads`` later.
    """
    # Freeze the ENTIRE base model first -- only the LoRA factors created below
    # (``lora_A``/``lora_B``, requires_grad=True by default) should train. Each
    # wrapper also freezes its own wrapped base weight, but the rest of the model
    # (embeddings, MLP, k/v_proj, lm_head, norms) must be frozen here too, or
    # AdamW allocates optimizer state for all ~1.5B params -> CUDA OOM.
    for p in model.parameters():
        p.requires_grad_(False)

    inserted: List[MaskedLoRALinear] = []
    replacements = list(_iter_target_linears(model, spec.target_modules))
    for name, linear in replacements:
        leaf = name.rsplit(".", 1)[-1]
        head_axis = _HEAD_AXIS.get(leaf)
        if head_axis is None:
            raise ValueError(f"no head axis known for target module {leaf!r}")
        mask = None if head_masks is None else head_masks.get(name)
        wrapper = MaskedLoRALinear(
            linear,
            num_heads=layout.num_attention_heads,
            head_dim=layout.head_dim,
            head_axis=head_axis,
            rank=spec.rank,
            alpha=spec.alpha,
            dropout=spec.dropout,
            mask=mask,
        )
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, leaf, wrapper)
        inserted.append(wrapper)
    return inserted


def adapted_budget(wrappers: Sequence[MaskedLoRALinear]) -> int:
    """Total (rank x adapted-head-dim) across wrappers -- the matched budget.

    Two routing variants are budget-matched iff this number is equal. A random
    mask built with the same ``k`` as a top-k mask yields the same budget.
    """
    return sum(w.rank * w.active_heads * w.head_dim for w in wrappers)
