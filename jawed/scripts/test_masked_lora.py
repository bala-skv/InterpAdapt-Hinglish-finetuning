"""Unit tests for the per-head masked LoRA (Stage-2 adapter surface).

The headline invariant -- **a head whose mask entry is 0 contributes exactly zero
to the LoRA output** -- is what makes the matched-budget "random mask" control a
fair comparison, so it is tested directly for both head axes.

Run (needs torch; no GPU required)::

    python scripts/test_masked_lora.py

Exits non-zero on the first failed assertion. Owner: Jawed.
"""
import _bootstrap  # noqa: F401

import torch
from torch import nn

from circuit_routing.masked_lora import (
    MaskedLoRALinear,
    adapted_budget,
    random_mask,
    soft_mask_from_scores,
    topk_mask_from_scores,
    uniform_mask,
)

torch.manual_seed(0)

NUM_HEADS = 12
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM   # 1536, Qwen2.5-1.5B q/o dim
RANK = 8
ALPHA = 16


def _wrapper(head_axis: str, *, mask=None, train_delta: bool = True) -> MaskedLoRALinear:
    """A wrapper with a NON-zero delta so masking is observable (B starts at 0)."""
    base = nn.Linear(HIDDEN, HIDDEN, bias=False)
    w = MaskedLoRALinear(
        base,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        head_axis=head_axis,
        rank=RANK,
        alpha=ALPHA,
        mask=mask,
    )
    if train_delta:  # give B a non-zero value so the delta is not trivially 0
        with torch.no_grad():
            w.lora_B.copy_(torch.randn_like(w.lora_B) * 0.1)
    return w


def _delta(w: MaskedLoRALinear, x: torch.Tensor) -> torch.Tensor:
    """The LoRA contribution alone = wrapper output - frozen base output."""
    return w(x) - w.base(x)


def test_zero_mask_head_contributes_zero_out_axis():
    """q_proj (head_axis='out'): a 0-gated head's OUTPUT slice gets no delta."""
    w = _wrapper("out")
    x = torch.randn(2, 5, HIDDEN)

    full = _delta(w, x)
    assert full.abs().sum() > 0, "delta is trivially zero; test cannot detect masking"

    mask = uniform_mask(NUM_HEADS)
    zeroed = 3
    mask[zeroed] = 0.0
    w.set_head_mask(mask)
    masked = _delta(w, x)

    sl = slice(zeroed * HEAD_DIM, (zeroed + 1) * HEAD_DIM)
    assert torch.allclose(masked[..., sl], torch.zeros_like(masked[..., sl])), \
        "zeroed output head still received a LoRA delta"

    # Every other output head is untouched.
    others = [h for h in range(NUM_HEADS) if h != zeroed]
    for h in others:
        s = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        assert torch.allclose(masked[..., s], full[..., s]), f"head {h} changed"
    print("PASS  zeroed head contributes zero (q_proj / out-axis)")


def test_zero_mask_head_contributes_zero_in_axis():
    """o_proj (head_axis='in'): a 0-gated INPUT head contributes nothing.

    Zeroing input head h must equal simply deleting head h's slice from x.
    """
    w = _wrapper("in")
    x = torch.randn(2, 5, HIDDEN)

    mask = uniform_mask(NUM_HEADS)
    zeroed = 7
    mask[zeroed] = 0.0
    w.set_head_mask(mask)
    masked = _delta(w, x)

    # Reference: drop head 7 from the input, keep uniform mask.
    x_dropped = x.clone()
    x_dropped[..., zeroed * HEAD_DIM:(zeroed + 1) * HEAD_DIM] = 0.0
    w.set_head_mask(uniform_mask(NUM_HEADS))
    reference = _delta(w, x_dropped)

    assert torch.allclose(masked, reference, atol=1e-5), \
        "masking input head != removing that head's contribution"
    print("PASS  zeroed head contributes zero (o_proj / in-axis)")


def test_uniform_mask_equals_plain_lora():
    """An all-ones mask reproduces a standard (un-gated) LoRA exactly."""
    for axis in ("out", "in"):
        w = _wrapper(axis)
        x = torch.randn(3, 4, HIDDEN)
        gated = _delta(w, x)
        # Reference plain-LoRA delta (no gate).
        ref = torch.nn.functional.linear(
            torch.nn.functional.linear(x, w.lora_A), w.lora_B
        ) * w.scaling
        assert torch.allclose(gated, ref, atol=1e-5), f"uniform != plain LoRA ({axis})"
    print("PASS  uniform mask == plain LoRA (both axes)")


def test_soft_mask_scales_delta():
    """Soft mode scales each head's delta by its (monotone) gate value."""
    scores = torch.linspace(-1.0, 1.0, NUM_HEADS).tolist()
    soft = soft_mask_from_scores(scores)  # min-max -> [0, 1], monotone
    assert soft.min() >= 0.0 and soft.max() <= 1.0 + 1e-6
    assert torch.all(soft[1:] >= soft[:-1] - 1e-6), "soft mask not monotone in scores"

    w = _wrapper("out", mask=uniform_mask(NUM_HEADS))
    x = torch.randn(2, 3, HIDDEN)
    full = _delta(w, x)
    w.set_head_mask(soft)
    scaled = _delta(w, x)

    for h in range(NUM_HEADS):
        s = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        assert torch.allclose(scaled[..., s], soft[h] * full[..., s], atol=1e-5), \
            f"soft scaling wrong at head {h}"
    print("PASS  soft mask scales each head's delta by f(s_i)")


def test_topk_and_random_matched_budget():
    """Top-k keeps exactly k heads; random control has the SAME budget."""
    scores = torch.randn(NUM_HEADS).tolist()
    k = 5

    top = topk_mask_from_scores(scores, k)
    assert int(top.sum()) == k, "top-k did not keep exactly k heads"
    # The k ones are the k highest scores.
    expected = set(torch.topk(torch.tensor(scores), k).indices.tolist())
    assert set(torch.nonzero(top).flatten().tolist()) == expected

    gen = torch.Generator().manual_seed(1)
    rnd = random_mask(NUM_HEADS, k, generator=gen)
    assert int(rnd.sum()) == k, "random mask has wrong sparsity"

    w_top = _wrapper("out", mask=top)
    w_rnd = _wrapper("out", mask=rnd)
    assert adapted_budget([w_top]) == adapted_budget([w_rnd]), \
        "top-k and random mask are NOT budget-matched"
    assert w_top.active_heads == k and w_rnd.active_heads == k
    print("PASS  top-k / random masks are budget-matched at k heads")


def test_delta_is_differentiable_through_mask():
    """Gradients flow to A/B for gated heads and are exactly zero for masked ones."""
    mask = uniform_mask(NUM_HEADS)
    mask[0] = 0.0
    w = _wrapper("out", mask=mask)
    x = torch.randn(1, 2, HIDDEN)
    _delta(w, x).pow(2).sum().backward()
    assert w.lora_A.grad is not None and w.lora_B.grad is not None
    # B's row block for the zeroed OUTPUT head receives no gradient.
    sl = slice(0, HEAD_DIM)
    assert torch.allclose(w.lora_B.grad[sl], torch.zeros_like(w.lora_B.grad[sl])), \
        "masked output head leaked gradient into B"
    print("PASS  gradients flow for active heads, zero for masked head")


def main() -> None:
    test_zero_mask_head_contributes_zero_out_axis()
    test_zero_mask_head_contributes_zero_in_axis()
    test_uniform_mask_equals_plain_lora()
    test_soft_mask_scales_delta()
    test_topk_and_random_matched_budget()
    test_delta_is_differentiable_through_mask()
    print("\nAll masked-LoRA tests passed.")


if __name__ == "__main__":
    main()
