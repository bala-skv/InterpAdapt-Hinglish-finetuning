# BabyShark — Mid Submission Report

**Group Members:** Bala, Daniel (Ashish), Jawed, Satyam
**Model:** Qwen2.5-1.5B (fp16)
**Metric:** Teacher-Forced Target Probability Recovery (Stage 1) · Macro-F1 (Stage 2 sentiment)

---

## Hardware Used
- **Bala:** ADA (gnode074: RTX 2080 Ti, sm_75)
- **Daniel:** ADA (gnode090, gnode071: RTX 2080 Ti, sm_75)
- **Jawed:** ADA (RTX 2080 Ti, sm_75 — Phase-0 gates + Stage-2 LoRA baseline)
- **Satyam:** Colab Pro only (Data acquisition and viz)

*Note: All results presented below are strictly sourced from ADA (`fp16`) to eliminate compute capability hardware variation between nodes.*

---

## 1. Tracing Methodology (Daniel's Handoff)

We traced the computational circuits for Hindi/English generation across three modalities to separate "Language Generation" from "Script Rendering".

* **en <-> hi-latn**: Pure Language Translation (English -> Hindi in Latin)
* **hi-deva <-> hi-latn**: Pure Script Transliteration (Hindi in Devanagari -> Hindi in Latin)
* **en <-> hi-deva**: Both Language and Script Translation (English -> Hindi in Devanagari)

### Activation Patching
To measure causal influence, we patched the residual stream vectors at `q_proj` and `o_proj` modules for each of the 28 layers and 12 attention heads. We used an evaluation metric based on full-sequence Teacher-Forced Target Probability. If restoring a head's activation from the clean run into the corrupted run recovered the target log-probability, the head was marked with a high causal score.

The trace was evaluated exhaustively across 38 specific concepts (averaging out noise) in an automated SLURM batch job. To handle token-merging across boundaries, BPE prefixes were dynamically aligned across targets.

### Sanity Checks & Causal Trace Properties
Before conducting the full trace, we verified the methodology against four strict gates on Ada (`fp16`):
1. **Re-injecting unchanged activations** yielded exact bit-for-bit logits.
2. **Patching zero heads** yielded exactly `0.0` recovery.
3. **Patching all 336 attention heads** yielded exactly `1.000` recovery on the first predicted token, and `0.8125` recovery when averaged across all subword tokens of the target word. We proved this mathematically: the 0.8125 multi-token ceiling is not a plumbing bug, but an artifact of the frozen KV-cache. Because `patch_pos` only intercepts the final prompt token, generating the second subword (e.g., the `la` in `neela`) forces the model to attend to the unpatched, corrupted English KV-cache prefix, dropping the log probability. We have decided to stick with the multi-token `_score` metric for the final heatmaps because it reflects true generation probability. We report the raw `0.8125` recovery instead of scaling it to `1.0` so that heatmap comparisons stay on a comparable scale across concepts with varying subword lengths.
4. **Per-layer sweeps** showed that no single layer recovers a large chunk of the probability on its own (peak of `0.093` at L22 for `sky_colour`), proving that the causal circuit for language and script is highly distributed. Individual head scores are small and consistent with a distributed circuit.

---

## 2. Stage 1 Headline Results

We generated a 28-layer x 12-head heatmap for each of the 3 specific conditions.

![Final Heatmaps](./final_full_heatmaps.png)
*(See attached `final_full_heatmaps.png`)*

### Score Correlation Analysis
To determine if Language and Script translation utilize the same circuits inside the model, we flattened the 336-element heatmaps and ran Pearson correlation checks across the conditions.

| Contrast Pair | Pearson Correlation (r) |
|---|---|
| Language vs Script (`en_hi-latn` vs `hi-deva_hi-latn`) | **0.5460** |
| Language vs Both (`en_hi-latn` vs `en_hi-deva`) | **0.6299** |
| Script vs Both (`hi-deva_hi-latn` vs `en_hi-deva`) | **0.6566** |

**Conclusion:** The heatmaps show significant overlap (r > 0.54) between Pure Language and Pure Script tasks. This indicates that Qwen2.5-1.5B's circuit for translating between languages heavily overlaps with its circuit for transliterating scripts. When both tasks are required simultaneously (English -> Devanagari), the resulting circuit is highly correlated with both independent circuits.

---

## 3. Stage 2 — Uniform-LoRA Baseline (Jawed)

As the reference point the Circuit-Routing Adapter must beat, we fine-tuned a **uniform LoRA** (rank 8 on `q_proj` + `o_proj`, 0.089% of parameters, `fp16`, 600 steps) on the frozen **SAIL-2017 Romanized (Hinglish) code-mixed sentiment** task (Satyam's sanitized splits: 10,068 train / 1,260 validation; 3 classes: negative / neutral / positive). The primary metric is **macro-F1**; predictions rank the three label verbalizers by mean teacher-forced log-probability.

| Metric | Base Qwen2.5-1.5B | + Uniform LoRA | Δ |
|---|---|---|---|
| Accuracy | 0.358 | **0.630** | +0.272 |
| Macro-F1 | 0.319 | **0.607** | +0.288 |

The base model sits at chance (0.319 macro-F1 ≈ the balanced 3-class floor), while a tiny uniform LoRA nearly doubles it. This establishes the **matched-budget baseline** against which the interpretability-guided routing masks (soft `f(s_i)` scaling and hard top-k over the Stage-1 head scores, implemented in `jawed/circuit_routing/masked_lora.py`) will be compared. Training run: [W&B `mn88apmt`](https://wandb.ai/phdiiitmohammed-iiit-hyderabad/babyshark-cra/runs/mn88apmt).

---

