# InterpAdapt: Interpretability-Guided Circuit Routing for Hindi-English LoRA Fine-Tuning

[![Paper](https://img.shields.io/badge/Paper-paper.pdf-red.svg)](paper/paper.pdf)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SAIL--2017-yellow.svg)](https://huggingface.co/datasets/satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2)
[![W&B Dashboard](https://img.shields.io/badge/W&B-babyshark--cra--ling-blue.svg)](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling)

**Team BabyShark**  
Balasubramanian K (2025121002) · Daniel Ashish Abraham (2025121010) · Mohammed Jawed Khan (2025801009) · Satyam Arora (2026901003)

---

## 1. Project Summary

**InterpAdapt** investigates whether causal activation-patching signals can identify language-specific sub-networks in a multilingual language model (`Qwen/Qwen2.5-1.5B`), and whether routing parameter-efficient adapters (LoRA) exclusively to those causally localized circuits improves fine-tuning on Hindi-English code-mixed downstream NLP.

The project evaluates two successive stages:
- **Stage 1 (Causal Localization):** Using teacher-forced activation patching to trace cross-lingual and cross-script circuits across 28 layers $\times$ 12 query heads (336 heads total) and 28 MLP down-projection layers under three matched language/script conditions: English (`en`), Romanized Hindi (`hi_latn`), and Devanagari Hindi (`hi_deva`).
- **Stage 2 (Circuit-Routing Adapter - CRA):** Fine-tuning masked LoRA adapters on SAIL-2017 Romanized Hindi-English sentiment classification (3-way classification: negative, neutral, positive) across principal 3-seed arms and single-seed exploratory variants, comparing causal circuit routing against matched random-routing controls and uniform parameter baselines.

---

## 2. Key Findings & Negative Results

### Stage 1: Circuit Localization & Head Interactions (H1, H2)
Evaluated on the primary 39-concept linguistic set under the language contrast (`en <-> hi_latn`, holding script constant):
- **Top Causal Attention Heads:** Causal recovery is concentrated in late-layer attention heads: **L22H0** ($+0.044$), **L21H2** ($+0.043$), and **L22H6** ($+0.039$). Across all 336 heads, **14 heads** pass the family-wise error rate (FWER) permutation test, and 137 heads have a 95% bootstrap CI strictly $> 0$.
- **Top Causal MLP Layers:** Late MLP blocks exhibit high causal restoration: **L17** ($+0.131$), **L22** ($+0.117$), and **L23** ($+0.110$). Overall, **15 of 28 MLP layers** are statistically significant.
- **Head Non-Independence:** Joint patching recovers *more* than the linear sum of single-head effects for small sets ($k=20$: joint recovery $0.587$ vs. sum of singles $0.495$), but saturates at larger subsets ($k=100$: joint recovery $0.949$ vs. sum of singles $1.053$).
- **Pairwise Interactions:** Among 1,225 shortlisted head pairs (all combinations among the top 50 heads), **137 pairs** show statistically robust interactions; 123 lie outside the empirical range of random control pairs ($[-0.0014, +0.0012]$), and 17 pairs exhibit negative (redundant) interactions.
- **Conclusion for H2:** **Hypothesis H2 (head independence) is not supported.** Single-head patching scores do not linearly determine the collective effect of a circuit.

### Stage 2: Circuit Routing vs. Matched Controls (H3, H4)
Evaluated on SAIL-2017 validation macro-F1 ($N=1,260$):
- **Base Model Zero-Shot:** $0.319$ macro-F1 (near the balanced 3-class chance floor).
- **Uniform LoRA Baseline:** $0.604 \pm 0.005$ macro-F1 ($1,376,256$ active LoRA parameters).
- **Attention Heads ($k=112$):** Top-$k$ reaches $0.592 \pm 0.007$; matched Random reaches $0.594 \pm 0.003$.  
  Paired difference (Top-$k$ minus Random): **$-0.002$** (95% bootstrap CI: $[-0.009, +0.004]$; spans zero).
- **Attention Heads ($k=34$):** Top-$k$ reaches $0.576 \pm 0.001$; matched Random reaches $0.583 \pm 0.004$.  
  Paired difference (Top-$k$ minus Random): **$-0.007$** (95% bootstrap CI: $[-0.014, +0.000]$; spans zero).
- **MLP Layers ($k=3$):** Top-$k$ reaches $0.573 \pm 0.003$; matched Random reaches $0.575 \pm 0.001$.  
  Paired difference (Top-$k$ minus Random): **$-0.001$** (95% bootstrap CI: $[-0.011, +0.009]$; spans zero).
- **Combined Heads + MLP ($k=112+3$):** Top-$k$ reaches $0.590 \pm 0.004$; matched Random reaches $0.592 \pm 0.005$.  
  Paired difference (Top-$k$ minus Random): **$-0.002$** (95% bootstrap CI: $[-0.009, +0.005]$; spans zero).
- **Top-$k112$ vs. Uniform LoRA:** Paired difference: **$-0.012$** (95% bootstrap CI: $[-0.020, -0.004]$). Uniform LoRA reliably outperforms the top-112 sub-network by $1.2$ macro-F1 points.
- **Conclusions for H3 & H4:**
  - **Hypothesis H3 (Score-guided routing outperforms random routing) is not supported.** Routing LoRA updates by causal patching scores shows no reliable benefit over budget-matched random allocation.
  - **Hypothesis H4 (Concentrated parameter efficiency) is only partially supported.** Sparse sub-networks adapt effectively above base, but uniform LoRA with the full parameter budget consistently achieves higher performance.
  - *Evaluation Note:* The 95% bootstrap confidence intervals resample validation examples to quantify evaluation noise; run-to-run variation is captured across the 3 independent training seeds.

---

## 3. Concept Sets & Tracing Data

We studied three concept sets to assess localization and cross-sample sensitivity:

1. **38-Concept Pilot Set (`data/tracing/tracing_set_v1.json`):**  
   Initial exploratory set across 38 validated linguistic contrasts evaluated across English (`en`), Romanized Hindi (`hi_latn`), and Devanagari Hindi (`hi_deva`).
2. **39-Concept Primary Linguistic Set (`results/concepts/linguistic_v1_ada/`):**  
   Primary benchmark evaluated on ADA. Thresholds fixed in advance required base model target log-probability $\log p \ge -8.0$ and contrast score $\ge 0.5$. Exactly 39 of 50 concepts met these criteria fixed in advance (`orig_rules/keep.json`, SHA-256: `2ef6938c45a538ce`). When validation thresholds were loosened during the run, 45 concepts passed (`ling45`). The 39 concepts passing thresholds fixed in advance serve as the primary source for Stage 2 score matrices (`data/stage2_scores/ling39_heads.json` and `ling39_mlp.json`).
3. **120-Concept Sensitivity Set (`results/concepts/linguistic_v3_a100_s35_r2/`):**  
   Extended set of 120 concepts (six concepts per grammatical category). 99 concepts pass the thresholds fixed in advance; 25 concept IDs are shared with the main set. The Hindi and Romanized prompt text was checked by one author (Satyam Arora) as an author check (not independent multi-annotator validation). Executed on an A100 GPU, this serves as a sensitivity analysis confirming qualitative circuit overlap rather than an independent confirmation.

---

## 4. Stage 2 Experimental Arms

| Arm | Description | Type | Seeds | Macro-F1 (mean $\pm$ sd) | Accuracy | Trainable Params |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **`H-uniform`** | Uniform LoRA on q/o across all 28 layers (Full head budget reference) | Principal | 3 | $0.604 \pm 0.005$ | 0.629 | 1,376,256 |
| **`H-topk112`** | CRA: LoRA on top 112 causal heads (Hard top-$k$) | Principal | 3 | $0.592 \pm 0.007$ | 0.620 | 917,504 |
| **`H-random112`** | CRA control: LoRA on 112 random heads (Matched budget) | Principal | 3 | $0.594 \pm 0.003$ | 0.621 | 868,352--917,504 |
| **`H-topk34`** | CRA: LoRA on top 34 causal heads (~10% budget) | Principal | 3 | $0.576 \pm 0.001$ | 0.607 | 512,000 |
| **`H-random34`** | CRA control: LoRA on 34 random heads (Matched budget) | Principal | 3 | $0.583 \pm 0.004$ | 0.613 | 512,000--561,152 |
| **`M-topk3`** | CRA: LoRA on down_proj of top 3 MLP layers | Principal | 3 | $0.573 \pm 0.003$ | 0.604 | 251,904 |
| **`M-random3`** | CRA control: LoRA on down_proj of 3 random MLP layers | Principal | 3 | $0.575 \pm 0.001$ | 0.603 | 251,904 |
| **`HM-topk`** | CRA: Top-112 heads + Top-3 MLP layers | Principal | 3 | $0.590 \pm 0.004$ | 0.616 | 1,169,408 |
| **`HM-random`** | CRA control: 112 random heads + 3 random MLP layers | Principal | 3 | $0.592 \pm 0.005$ | 0.619 | 1,120,256--1,169,408 |
| **`H-soft`** | CRA: Soft routing scaled per head by normalized causal recovery | Exploratory | 1 (s0) | 0.568 | 0.601 | 1,038,336 |
| **`M-uniform`** | LoRA on down_proj across all 28 MLP layers | Exploratory | 1 (s0) | 0.606 | 0.629 | 2,351,104 |
| **`M-soft`** | CRA: down_proj LoRA scaled per layer by normalized recovery | Exploratory | 1 (s0) | 0.585 | 0.613 | 1,427,456 |
| **`HM-topk34`** | CRA: Top-34 heads + Top-3 MLP layers | Exploratory | 1 (s0) | 0.584 | 0.611 | 763,904 |
| **`HM-random34`** | CRA control: 34 random heads + 3 random MLP layers | Exploratory | 1 (s0) | 0.588 | 0.617 | 763,904 |

*(Note: Random-routing parameter ranges reflect variations in the number of transformer layers touched across random seeds: touching an additional attention layer activates its ungated projection weight slice).*

---

## 5. Repository Structure

```text
InterpAdapt-Hinglish-finetuning/
├── paper/                     Paper PDF, LaTeX tables, macros, and bibliographies
│   ├── paper.pdf              Compiled project paper (12 pages including refs & appendix)
│   ├── numbers.tex            Macro declarations for all paper metrics (\P...)
│   ├── refs.bib, custom.bib   Bibliographic databases (anthology.bib, custom.bib, refs.bib)
│   └── tables/                LaTeX table sources (tab_stage2_ling, tab_heads_ling, etc.)
├── src/                       Core package source code
│   ├── circuit_routing/       Masked LoRA linear layers, routing wrappers, and accounting
│   │   ├── hooks.py           Activation hook utilities
│   │   ├── masked_lora.py     MaskedLoRALinear implementation and mask generation
│   │   └── mlp_routing.py     MLP layer-gate wrappers and effective parameter counting
│   └── interp_adapt/tracing/  Activation patching and scoring mechanics
│       ├── patching.py        Single-head causal restoration patching
│       ├── mlp_patching.py    MLP layer patching primitives
│       └── sequence_scoring.py Teacher-forced target log-probability scoring
├── scripts/                   Execution and reproduction scripts
│   ├── phase0/                Phase 0 verification and memory pilots
│   ├── stage1/                Stage 1 tracing, validation, and interaction analysis
│   │   ├── validate_tracing_set_v1.py   Validates concept sets against thresholds
│   │   ├── full_trace.py                Executes activation patching across all heads
│   │   ├── analyze_trace.py             Produces heatmaps, rankings, and summary tables
│   │   ├── entangle.py                  Joint necessity, sufficiency, and pair sweeps
│   │   └── plot_entangle.py             Generates sufficiency and necessity curves
│   ├── stage2/                Stage 2 CRA training and analysis
│   │   ├── train_cra_compare.py         Main training entry point for all arms
│   │   ├── build_scores_from_run.py     Builds Stage 2 routing score matrices
│   │   ├── summarize_study.py           Computes seed statistics and paired bootstrap CIs
│   │   └── push_adapters.py             Uploads checkpoints and model cards to Hugging Face
│   └── pipeline/
│       └── run_concepts.py    Automated concept pipeline (heads, MLPs, interactions)
├── configs/                   Experiment YAML configurations
│   ├── cra_ling.yaml          Primary Stage 2 config (rank 8, alpha 16, lr 1e-4, 600 steps)
│   └── cra_compare.yaml       Comparison / soft routing configuration
├── data/                      Datasets and exported routing matrices
│   ├── tracing/               Concept definition files (tracing_set_v1.json, etc.)
│   └── stage2_scores/         Routing scores (ling39_heads.json, ling39_mlp.json)
├── results/                   Retained machine-generated experimental artifacts
│   ├── stage1/                Full traces, interaction pairs, and correlation analyses
│   ├── stage2/ling_study/     Stage 2 per-run outputs, summary.json, and summary.md
│   └── concepts/              Pipeline runs (linguistic_v1_ada, orig_rules)
├── figures/                   Presentation-quality figures
│   ├── stage1/                Heatmaps and entangle/ curves
│   └── concepts/              MLP layer profiles and concept histograms
├── tests/                     Fast contract and regression tests
├── cluster/                   SLURM job scripts for cluster environments
├── docs/                      Experiment reports, data dictionaries, and architectural notes
└── requirements.txt           Pinned dependency manifest
```

---

## 6. Environment & Hardware Setup

### Environment Setup
- Python 3.10+ recommended.
- PyTorch with CUDA support.

Install dependencies and editable package:
```bash
pip install -r requirements.txt
pip install -e .
```

### Base Model
Pinned Hugging Face model and revision:
- Identifier: `Qwen/Qwen2.5-1.5B`
- Commit revision: `8faed761d45a263340a0528343f099c05c9a4323`

### Hardware Requirements & Runtime Estimates
- **GPU Requirements:** Training and tracing experiments require an **NVIDIA GeForce RTX 2080 Ti** (11 GB VRAM, `torch.float16`) or equivalent cluster GPU.
- **Estimated Runtime:**
  - Single Stage 2 training run (600 steps): ~45 minutes (~33 min training at 3.3 s/step + 12 min validation).
  - Full Stage 2 study (30 runs across all arms & seeds): ~22 GPU-hours (~4 wall-clock hours with 6 parallel workers).
  - Stage 1 full head trace (39 concepts $\times$ 3 conditions): ~20--30 minutes on GPU.
- **CPU Verification:** Metric summaries, compilation checks, unit tests, and score table generation run entirely on CPU without requiring a GPU.

---

## 7. Commands to Reproduce Paper Tables & Figures

### Fast Verification Commands (CPU Only, No GPU Required)
```bash
# 1. Bytecode compilation check (<2 seconds)
python -m compileall -q src scripts

# 2. Run repository contract unit tests (<1 second)
python -m unittest tests.test_repository_contract

# 3. Generate Stage 2 study summary, seed statistics, and paired bootstrap CIs (<1 second)
python scripts/stage2/summarize_study.py --study ling_study

# 4. Compute Stage 1 cross-condition correlations (<1 second)
python scripts/stage1/compute_correlations.py

# 5. Re-export Stage 2 routing score matrices from the 39 accepted concepts (<1 second)
python scripts/stage2/build_scores_from_run.py \
  --run results/concepts/linguistic_v1_ada/orig_rules \
  --keep results/concepts/linguistic_v1_ada/orig_rules/keep.json \
  --name ling39

# 6. Render layer-by-head heatmaps from saved trace (<2 seconds)
python scripts/stage1/plot_heatmaps_v2.py

# 7. Render sufficiency and necessity interaction curves from saved sweep (<10 seconds)
python scripts/stage1/plot_entangle.py
```

### Full Tracing & Training Commands (GPU Required)

#### Table 1 & Table 4: Stage 2 Routing Evaluation (`tab_stage2_ling.tex`)
Reproduce individual Stage 2 arms (RTX 2080 Ti, ~45 min per run):
```bash
# Uniform LoRA baseline
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm H-uniform --seed 0

# Hard Top-112 Attention Heads
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm H-topk112 --seed 0

# Matched Random-112 Attention Heads control
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm H-random112 --seed 0

# Hard Top-3 MLP Layers
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm M-topk3 --seed 0

# Matched Random-3 MLP Layers control
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm M-random3 --seed 0

# Combined Top-112 Heads + Top-3 MLP Layers
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm HM-topk --seed 0

# Combined Random-112 Heads + Random-3 MLP Layers control
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm HM-random --seed 0
```
Or launch all runs in sequence/parallel via the study script:
```bash
bash scripts/stage2/launch_study.sh
```

#### Table 2 & Table 3: Top Heads & MLP Layers (`tab_heads_ling.tex`, `tab_mlp_ling.tex`)
Analyze activation tracing results and print component rankings (CPU, uses saved trace):
```bash
python scripts/stage1/analyze_trace.py \
  --trace results/concepts/linguistic_v1_ada/orig_rules/heads_trace.json \
  --out results/concepts/linguistic_v1_ada/orig_rules/heads_analysis
```

#### Stage 1 Validation & Tracing from Scratch (GPU Required)
```bash
# Validate concept rules against acceptance thresholds
python scripts/stage1/validate_tracing_set_v1.py

# Run full head activation patching
python scripts/stage1/full_trace.py --data data/tracing/tracing_set_v1.json
```

#### Table 5: LoRA Hyperparameters (`tab_hparams.tex`)
Hyperparameters locked in `configs/cra_ling.yaml`: rank 8, alpha 16, dropout 0.05, lr 1e-4, weight decay 0.01, warmup 30 steps, max steps 600, effective batch size 32 ($4 \times 8$).

---

## 8. Paper Build Note

The compiled paper is provided at:
```text
paper/paper.pdf
```
To compile the LaTeX source from scratch (requires a TeX distribution with `pdflatex` and `bibtex`):
```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
*(Note: If building from an external Overleaf export, ensure `numbers.tex`, `tables/`, `acl.sty`, `acl_natbib.bst`, and the bibliography files `refs.bib` / `custom.bib` are in the LaTeX search path).*

---

## 9. Limitations

Mirrored from Section 8 of the paper:
1. **Single Model Scale:** Evaluated exclusively on `Qwen2.5-1.5B`. Circuit structures and routing dynamics may differ in larger or differently trained architectures.
2. **Single Downstream Task:** Downstream evaluation is restricted to SAIL-2017 3-way sentiment classification on Romanized Hindi-English.
3. **Split Restriction:** All evaluation numbers are reported on the validation split ($N=1,260$); the test set labels are held out.
4. **Seed Budget:** Principal comparisons use 3 random seeds; while sufficient to detect major effects, wider seed sweeps would give higher statistical precision.
5. **Objective Mismatch:** Stage 1 discovers circuits via last-token restoration on isolated linguistic word-pairs, whereas Stage 2 fine-tunes full sequence representations for sentiment classification.
6. **Signed Scoring:** Top-$k$ head selection ranks components by signed recovery score; heads with strong negative scores (inhibiting opposite-language tokens) are treated as non-participating rather than selectively routed.
7. **Active vs. Allocated Parameters:** Parameter accounting reports *effective* trainable parameters (parameters receiving non-zero gradients under the routing masks). However, GPU memory allocation instantiates standard LoRA tensors across all 28 layers ($1,376,256$ allocated parameters for attention routing).

---

## 10. Links & Artifacts

- **Downstream Dataset:** [`satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2`](https://huggingface.co/datasets/satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2)
- **Weights & Biases Dashboard:** [`skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling`](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling)
- **Hugging Face Model Repositories:** Fine-tuned adapter checkpoints and metrics under [`bala-skv`](https://huggingface.co/bala-skv):
  - `H-uniform`: [bala-skv/babyshark-cra-H-uniform](https://huggingface.co/bala-skv/babyshark-cra-H-uniform)
  - `H-topk112`: [bala-skv/babyshark-cra-H-topk112](https://huggingface.co/bala-skv/babyshark-cra-H-topk112)
  - `H-random112`: [bala-skv/babyshark-cra-H-random112](https://huggingface.co/bala-skv/babyshark-cra-H-random112)
  - `H-topk34`: [bala-skv/babyshark-cra-H-topk34](https://huggingface.co/bala-skv/babyshark-cra-H-topk34)
  - `H-random34`: [bala-skv/babyshark-cra-H-random34](https://huggingface.co/bala-skv/babyshark-cra-H-random34)
  - `H-soft`: [bala-skv/babyshark-cra-H-soft](https://huggingface.co/bala-skv/babyshark-cra-H-soft)
  - `M-uniform`: [bala-skv/babyshark-cra-M-uniform](https://huggingface.co/bala-skv/babyshark-cra-M-uniform)
  - `M-topk3`: [bala-skv/babyshark-cra-M-topk3](https://huggingface.co/bala-skv/babyshark-cra-M-topk3)
  - `M-random3`: [bala-skv/babyshark-cra-M-random3](https://huggingface.co/bala-skv/babyshark-cra-M-random3)
  - `M-soft`: [bala-skv/babyshark-cra-M-soft](https://huggingface.co/bala-skv/babyshark-cra-M-soft)
  - `HM-topk`: [bala-skv/babyshark-cra-HM-topk](https://huggingface.co/bala-skv/babyshark-cra-HM-topk)
  - `HM-random`: [bala-skv/babyshark-cra-HM-random](https://huggingface.co/bala-skv/babyshark-cra-HM-random)
  - `HM-topk34`: [bala-skv/babyshark-cra-HM-topk34](https://huggingface.co/bala-skv/babyshark-cra-HM-topk34)
  - `HM-random34`: [bala-skv/babyshark-cra-HM-random34](https://huggingface.co/bala-skv/babyshark-cra-HM-random34)

---

## 11. Team Contributions

- **Balasubramanian K:** Head-interaction analysis, Circuit-Routing Adapter (CRA) architecture design, Stage 2 experimental evaluation, and paired bootstrap statistical testing.
- **Daniel Ashish Abraham:** Stage 1 causal localization run on ADA, teacher-forced activation patching mechanics, theoretical formulation, and trace data analysis.
- **Mohammed Jawed Khan:** MLP-layer patching primitives, layer-gate wrapper implementation, and preliminary routing explorations.
- **Satyam Arora:** Concept-set design, 120-concept replication pipeline execution, and Hindi linguistic & Romanization data verification.
