# BabyShark — Mid Submission Report

**Group Members:** Bala, Daniel (Ashish), Jawed, Satyam
**Model:** Qwen2.5-1.5B (fp16)
**Metric:** Teacher-Forced Target Probability Recovery

---

## Hardware Used
- **Bala:** ADA (gnode074: RTX 2080 Ti, sm_75)
- **Daniel:** ADA (gnode090, gnode071: RTX 2080 Ti, sm_75)
- **Jawed:** ADA (Upcoming LoRA stages)
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

## 3. Concrete Timeline

With Daniel's Stage 1 Trace complete, the project timeline follows strictly:

1. **Jawed (Stage 2 - Mid October):** Use the high-scoring head coordinates from `runs/full_trace_results.json` to configure localized LoRA adapters. Will target only the highly-correlated overlapping heads to improve English->Hinglish finetuning without catastrophic forgetting.
2. **Satyam (Late October):** Collate full-scale benchmark datasets and assist Jawed with Colab-side notebook prototyping.
3. **Bala (Early November):** Aggregate metrics, perform end-to-end benchmarking of Jawed's adapters against base models, and finalize Phase 3.
4. **Final Submission (November):** Complete codebase delivery.
