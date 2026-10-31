# InterpAdapt: Interpretability-Guided Circuit Routing for Hindi-English LoRA Fine-Tuning

[![Paper](https://img.shields.io/badge/Paper-paper.pdf-red.svg)](paper/paper.pdf)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SAIL--2017-yellow.svg)](https://huggingface.co/datasets/satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2)
[![W&B Dashboard](https://img.shields.io/badge/W&B-babyshark--cra--ling-blue.svg)](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling)

**Team BabyShark**  
Balasubramanian K (2025121002) · Daniel Ashish Abraham (2025121010) · Jawed Mohammed Khan (2025801009) · Satyam Arora (2026901003)

---

## 1. Project Summary

**InterpAdapt** investigates whether causal activation-patching signals can identify language-specific sub-networks in a modern multilingual LLM (`Qwen/Qwen2.5-1.5B`), and whether routing parameter-efficient adapters (LoRA) exclusively to those causally localized circuits improves fine-tuning for Hindi-English code-mixed NLP.

The project is structured into two main phases:
- **Stage 1 (Causal Localization):** Using teacher-forced activation patching to trace cross-lingual and cross-script circuits across 28 layers $\times$ 12 query heads and 28 MLP down-projection layers.
- **Stage 2 (Circuit-Routing Adapter - CRA):** Fine-tuning masked LoRA adapters on SAIL-2017 Romanized Hindi-English sentiment classification across 14 experimental arms, comparing causal circuit routing against matched random-routing controls and uniform parameter baselines.

---

## 2. Key Findings & Negative Results

1. **Stage 1 Circuit Localization:**  
   Activation patching reveals concentrated language-sensitive computation in middle-to-late transformer layers:
   - Top causal attention heads are concentrated in layers 7–10 and 15–22 (e.g., L9H3, L7H1, L2H5, L15H11, L22H0).
   - Top causal MLP layers are concentrated in layers 17–25 (e.g., L17, L22, L23, L24, L25).
   - Head interaction tests show sub-additive scaling: a small circuit of ~112 heads recovers >90% of total restoration score, but individual head effects do not act independently.

2. **Stage 2 Routing Outcome (Core Negative Result):**  
   Routing LoRA adapters exclusively to the causally localized circuits **yields no statistically reliable advantage over matched random routing** on SAIL-2017:
   - **Attention Heads (k=112):** Top-$k$ minus Random = $-0.002$ Macro-F1 (95% CI: $[-0.009, +0.004]$; contains 0).
   - **MLP Layers (k=3):** Top-$k$ minus Random = $-0.001$ Macro-F1 (95% CI: $[-0.011, +0.009]$; contains 0).
   - **Combined Heads + MLP (k=112+3):** Top-$k$ minus Random = $-0.002$ Macro-F1 (95% CI: $[-0.009, +0.005]$; contains 0).
   - **Uniform LoRA Dominance:** Standard uniform LoRA across all 28 layers achieves $0.604 \pm 0.005$ Macro-F1 ($1,376,256$ parameters), reliably outperforming the 112-head sub-network by $+0.012$ Macro-F1 (95% CI: $[+0.004, +0.020]$).

All paired bootstrap confidence intervals span zero for top-$k$ versus matched random controls, demonstrating that causal patching recovery does not translate into preferential LoRA allocation sites for downstream task adaptation.

---

## 3. Concept Sets & Tracing Data

We evaluated three concept sets spanning pilot exploration, primary evaluation, and cross-dataset replication:

1. **38-Concept Pilot Set (`data/tracing/tracing_set_v1.json`):**  
   Initial discovery set across 38 validated linguistic contrasts evaluated across English (`en`), Romanized Hindi (`hi_latn`), and Devanagari Hindi (`hi_deva`).
2. **39-Concept Primary Linguistic Set (`results/concepts/linguistic_v1_ada/`):**  
   Primary set governed by pre-registered acceptance criteria: base model target log-probability $\log p \ge -8.0$ and contrast score $\ge 0.5$. Exactly 39 concepts satisfy these criteria (`orig_rules/keep.json`, SHA-256: `2ef6938c45a538ce`). Under loosened thresholds, 45 concepts are retained (`ling45`). Stage 2 score matrices (`data/stage2_scores/ling39_heads.json` and `ling39_mlp.json`) are built directly from these 39 concepts.
3. **120-Concept Scaled Replication Set (`results/concepts/linguistic_v3_a100_s35_r2/`):**  
   Extended linguistic evaluation across 120 concepts (six concepts per grammatical category, human-verified). 99 concepts meet pre-registered criteria; all 120 qualify under loosened criteria. Executed on cluster A100 GPUs, confirming cross-hardware and cross-sample circuit stability.

---

## 4. Repository Structure

```text
InterpAdapt-Hinglish-finetuning/
├── paper/                     Final camera-ready PDF and LaTeX source files
│   ├── paper.pdf              Final compiled paper (13 pages, camera-ready)
│   ├── numbers.tex            Macro declarations for all paper metrics
│   ├── refs.bib, custom.bib   Bibliographic references
│   └── tables/                All LaTeX table sources (tab_stage2_ling, etc.)
├── src/                       Core package source code
│   ├── circuit_routing/       Masked LoRA linear layers, routing wrappers, accounting
│   │   ├── hooks.py           Forward hooks for activation capture and inspection
│   │   ├── masked_lora.py     MaskedLoRALinear implementation and mask builders
│   │   └── mlp_routing.py     MLP layer-gate wrappers and effective parameter counting
│   └── interp_adapt/tracing/  Activation patching and scoring mechanics
│       ├── patching.py        Single-head causal restoration patching
│       ├── mlp_patching.py    MLP layer patching primitives
│       └── sequence_scoring.py Sequence log-probability scoring
├── scripts/                   Execution and reproduction scripts
│   ├── stage1/                Stage 1 tracing, validation, and analysis
│   │   ├── validate_tracing_set_v1.py   Validates concept sets against thresholds
│   │   ├── full_trace.py                Executes activation patching across all heads
│   │   ├── analyze_trace.py             Produces heatmaps, rankings, and summary tables
│   │   ├── entangle.py                  Joint necessity, sufficiency, and pair sweeps
│   │   └── plot_entangle.py             Generates sufficiency and necessity curves
│   ├── stage2/                Stage 2 CRA training and analysis
│   │   ├── train_cra_compare.py         Main training entry point for all 14 arms
│   │   ├── build_scores_from_run.py     Builds Stage 2 routing score matrices
│   │   ├── summarize_study.py           Computes seed statistics and paired bootstrap CIs
│   │   └── push_adapters.py             Uploads checkpoints and model cards to Hugging Face
│   └── pipeline/
│       └── run_concepts.py    End-to-end concept pipeline (heads, MLPs, interactions)
├── configs/                   Experiment YAML configurations
│   ├── cra_ling.yaml          Primary Stage 2 config (rank 8, alpha 16, lr 1e-4, 600 steps)
│   └── cra_compare.yaml       Comparison / soft routing configuration
├── data/                      Datasets and exported routing matrices
│   ├── tracing/               Concept definition files (tracing_set_v1.json, etc.)
│   └── stage2_scores/         ling39_heads.json and ling39_mlp.json
├── results/                   Retained machine-generated experimental artifacts
│   ├── stage1/                Full traces, analysis results, and interaction pairs
│   ├── stage2/ling_study/     Stage 2 per-run outputs, summary.json, and summary.md
│   └── concepts/              Pipeline runs (linguistic_v1_ada, orig_rules)
├── figures/                   Presentation-quality figures
│   ├── stage1/                full_heatmaps.png, entangle/ curves and sweeps
│   └── concepts/              MLP layer profiles and concept histograms
├── docs/                      Technical reports, data dictionaries, and architectural notes
└── requirements.txt           Pinned dependency manifest
```

---

## 5. Environment & Hardware Setup

### Python Environment
- Python 3.10+ is recommended.
- PyTorch with CUDA 11.8+ or 12.1+.

Install dependencies from the frozen manifests:
```bash
pip install -r requirements.txt
pip install -e .
```

### Base Model
The base model is pinned to the exact revision:
- Model identifier: `Qwen/Qwen2.5-1.5B`
- Hugging Face commit hash: `8faed761d45a263340a0528343f099c05c9a4323`

### Hardware & Reproducibility Specs
- **Primary Stage 1 Tracing & Stage 2 CRA:** NVIDIA GeForce RTX 2080 Ti (11 GB VRAM) using `torch.float16`.
- **120-Concept Replications:** Cluster NVIDIA A100-SXM4-80GB GPU.
- **Seeds:** All multi-seed experiments evaluate seeds `0`, `1`, `2`.

---

## 6. Commands to Reproduce Paper Tables & Figures

### Paper Table 1: Stage 2 Routing Performance (`tab_stage2_ling.tex`)
Table 1 reports validation Macro-F1, accuracy, and active parameters across all 14 arms.

To run Stage 2 training arms (or execute all via `bash scripts/stage2/launch_study.sh`):
```bash
# Uniform LoRA baseline (all 28 layers, 1,376,256 effective params)
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm H-uniform --seed 0

# Hard Top-112 Heads (917,504 effective params)
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm H-topk112 --seed 0

# Matched Random-112 Heads (868,352 - 917,504 effective params)
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm H-random112 --seed 0

# Hard Top-3 MLP Layers (251,904 effective params)
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm M-topk3 --seed 0

# Matched Random-3 MLP Layers (251,904 effective params)
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm M-random3 --seed 0

# Combined Hard Top-112 Heads + Top-3 MLP Layers (1,169,408 effective params)
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm HM-topk --seed 0

# Combined Random-112 Heads + Random-3 MLP Layers (1,120,256 - 1,169,408 effective params)
python scripts/stage2/train_cra_compare.py --config configs/cra_ling.yaml --arm HM-random --seed 0
```

To aggregate results, compute seed means, standard deviations, and paired bootstrap 95% CIs:
```bash
python scripts/stage2/summarize_study.py --study ling_study
```
*(Aggregate outputs are preserved at `results/stage2/ling_study/summary.md` and `summary.json`).*

---

### Paper Table 2 & 3: Top Causal Heads & MLP Layers (`tab_heads_ling.tex`, `tab_mlp_ling.tex`)
Rankings of the top-5 causal attention heads and top causal MLP layers on the `en <-> hi_latn` contrast:
```bash
# Validate concept rules and threshold acceptance
python scripts/stage1/validate_tracing_set_v1.py

# Trace and analyze attention heads
python scripts/stage1/analyze_trace.py \
  --trace results/concepts/linguistic_v1_ada/orig_rules/heads_trace.json \
  --out results/concepts/linguistic_v1_ada/orig_rules/heads_analysis

# Build Stage 2 routing score JSON files from the 39 accepted concepts
python scripts/stage2/build_scores_from_run.py \
  --run results/concepts/linguistic_v1_ada/orig_rules \
  --keep results/concepts/linguistic_v1_ada/orig_rules/keep.json \
  --name ling39
```

---

### Paper Table 4: Replication & Robustness (`tab_robust_ling.tex`)
Cross-concept consistency across the 38-concept pilot, 39-concept linguistic set, and 120-concept replication set:
```bash
python scripts/stage1/compute_correlations.py
```

---

### Paper Table 5: Hyperparameters (`tab_hparams.tex`)
Hyperparameters locked in `configs/cra_ling.yaml`:
- Base model: `Qwen/Qwen2.5-1.5B` (fp16)
- LoRA rank: $r = 8$, LoRA scaling: $\alpha = 16$, LoRA dropout: $0.05$
- Target modules: `q_proj`, `o_proj` (attention) and `down_proj` (MLP)
- Optimizer: AdamW, Learning rate: $1\times 10^{-4}$, Weight decay: $0.01$
- Batch size: 4 per device $\times$ 8 gradient accumulation steps = 32 effective batch size
- Steps: 600 max steps with 30 warmup steps; Evaluation frequency: every 50 steps

---

### Paper Figures: Heatmaps & Interaction Curves
1. **Layer-by-Head Heatmaps (`heatmaps.png`, `full_heatmaps.png`):**  
   Displays causal recovery scores across all 28 layers $\times$ 12 heads for `en <-> hi_latn`, `hi_deva <-> hi_latn`, and `en <-> hi_deva`:
   ```bash
   python scripts/stage1/plot_heatmaps_v2.py
   ```
   *(Rendered figure: `figures/stage1/full_heatmaps.png`)*

2. **Sufficiency & Necessity Curves (`curve_en_hi-latn.pdf`, `necessity_en_hi-latn.pdf`):**  
   Evaluates cumulative circuit recovery and necessity degradation:
   ```bash
   python scripts/stage1/entangle.py curve
   python scripts/stage1/entangle.py necessity
   python scripts/stage1/plot_entangle.py
   ```
   *(Rendered figures: `figures/stage1/entangle/curve_en_hi-latn.pdf` and `necessity_en_hi-latn.pdf`)*

3. **MLP Layer Recovery Profile (`mlp_en_hi-latn.png`):**  
   Displays causal restoration across transformer MLP blocks:
   *(Rendered figure: `figures/concepts/linguistic_v1_ada/mlp/mlp_en_hi-latn.png`)*

---

## 7. Model Checkpoints & Remote Links

### Downstream Task Dataset
- **SAIL-2017 Sentiment (Romanized Hindi-English):**  
  [`satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2`](https://huggingface.co/datasets/satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2)

### Experiment Dashboard
- **Weights & Biases Dashboard:**  
  [`skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling`](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling)

### Hugging Face Adapter Checkpoints
All 14 Stage 2 fine-tuned adapter weights and training logs are available on Hugging Face:
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

## 8. Paper

The camera-ready paper PDF is located at:
```text
paper/paper.pdf
```
LaTeX sources, table fragments, and bibliography entries are organized in `paper/`.
