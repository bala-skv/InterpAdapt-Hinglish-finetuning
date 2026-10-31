# InterpAdapt Stage 2 Linguistic Study (`ling_study`) -- Experiment Report

> **Task:** SAIL-2017 Hinglish sentiment classification (3-way: negative, neutral, positive).  
> **Base Model:** `Qwen/Qwen2.5-1.5B` (fp16) | **Zero-shot base Macro-F1:** `0.3194`  
> **Downstream Dataset:** [`satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2`](https://huggingface.co/datasets/satyam-arora-iiit-hyderabad/babyshark-sail2017-stage2)  
> **Concept Tracing Source:** Linguistic concept set `ling39` (39 concepts passing pre-registered validation thresholds)  
> **W&B Project Dashboard:** [https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling)  

---

## 1. Stage 2 Model Checkpoints & W&B Tracking

All 14 experimental arms are hosted on Hugging Face Hub under [`bala-skv`](https://huggingface.co/bala-skv) and tracked on Weights & Biases:

| Arm | Description | Seeds | Macro-F1 (mean $\pm$ sd) | Accuracy | Effective Params | Hugging Face Repository | W&B Run Links |
|---|---|---|---|---|---|---|---|
| **H-uniform** | Uniform LoRA on q_proj and o_proj across all 28 layers (Full head budget reference) | 3 | 0.6037 $\pm$ 0.0054 | 0.6288 | 1,376,256 | [H-uniform](https://huggingface.co/bala-skv/babyshark-cra-H-uniform) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/kizusvrn), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/cmsfl0hj), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/r9luy3cc) |
| **H-topk112** | CRA: LoRA on 112 attention heads with highest causal recovery (Hard top-k) | 3 | 0.5919 $\pm$ 0.0070 | 0.6204 | 917,504 | [H-topk112](https://huggingface.co/bala-skv/babyshark-cra-H-topk112) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/vk75cxof), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/xn0eu43x), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/9ofa8zlb) |
| **H-random112** | CRA control: LoRA on 112 random heads (Budget-matched to H-topk112) | 3 | 0.5944 $\pm$ 0.0032 | 0.6206 | 868,352 | [H-random112](https://huggingface.co/bala-skv/babyshark-cra-H-random112) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/8j11ripl), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/qzmme4t8), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/7mtdsg64) |
| **H-topk34** | CRA: LoRA on 34 highest-recovery heads (~10% budget) | 3 | 0.5765 $\pm$ 0.0014 | 0.6074 | 512,000 | [H-topk34](https://huggingface.co/bala-skv/babyshark-cra-H-topk34) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/ct0k8q6t), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/ua7pmxoi), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/3sla7o1n) |
| **H-random34** | CRA control: LoRA on 34 random heads (Budget-matched to H-topk34) | 3 | 0.5830 $\pm$ 0.0039 | 0.6127 | 512,000 | [H-random34](https://huggingface.co/bala-skv/babyshark-cra-H-random34) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/id52y560), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/290okuei), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/9m19j0b4) |
| **H-soft** | CRA: Soft routing with LoRA scaled per head by normalized causal recovery | 1 | 0.5685 | 0.6008 | 1,038,336 | [H-soft](https://huggingface.co/bala-skv/babyshark-cra-H-soft) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/82tlxqwj) |
| **M-uniform** | LoRA on down_proj across all 28 MLP layers (Full MLP budget reference) | 1 | 0.6057 | 0.6294 | 2,351,104 | [M-uniform](https://huggingface.co/bala-skv/babyshark-cra-M-uniform) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/j7957kov) |
| **M-topk3** | CRA: LoRA on down_proj of 3 MLP layers with highest causal recovery | 3 | 0.5734 $\pm$ 0.0033 | 0.6040 | 251,904 | [M-topk3](https://huggingface.co/bala-skv/babyshark-cra-M-topk3) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/vvcd6zh4), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/g5nzotkk), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/tpk8vzrx) |
| **M-random3** | CRA control: LoRA on down_proj of 3 random MLP layers | 3 | 0.5746 $\pm$ 0.0013 | 0.6034 | 251,904 | [M-random3](https://huggingface.co/bala-skv/babyshark-cra-M-random3) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/gti35wqw), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/ky1o41ti), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/3hd29bpf) |
| **M-soft** | CRA: down_proj LoRA scaled per layer by normalized MLP recovery | 1 | 0.5846 | 0.6127 | 1,427,456 | [M-soft](https://huggingface.co/bala-skv/babyshark-cra-M-soft) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/4amq6hgx) |
| **HM-topk** | CRA: Top-112 heads + Top-3 MLP layers | 3 | 0.5898 $\pm$ 0.0040 | 0.6164 | 1,169,408 | [HM-topk](https://huggingface.co/bala-skv/babyshark-cra-HM-topk) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/im73rgmz), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/tdxw4060), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/eec03wm1) |
| **HM-random** | CRA control: 112 random heads + 3 random MLP layers | 3 | 0.5922 $\pm$ 0.0055 | 0.6185 | 1,120,256 | [HM-random](https://huggingface.co/bala-skv/babyshark-cra-HM-random) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/x8tmkawl), [s1](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/dqa0wojs), [s2](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/62aj5jin) |
| **HM-topk34** | CRA: Top-34 heads + Top-3 MLP layers | 1 | 0.5842 | 0.6111 | 763,904 | [HM-topk34](https://huggingface.co/bala-skv/babyshark-cra-HM-topk34) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/0jehiq4o) |
| **HM-random34** | CRA control: 34 random heads + 3 random MLP layers | 1 | 0.5879 | 0.6175 | 763,904 | [HM-random34](https://huggingface.co/bala-skv/babyshark-cra-HM-random34) | [s0](https://wandb.ai/skv-balasubramanian-iiit-hyderabad/babyshark-cra-ling/runs/nsl95wfq) |

---

## 2. Hugging Face Repositories (Quick Links)

### MLP & Combined Attention + MLP Routing (M-* and HM-*)
- **`M-uniform`**: https://huggingface.co/bala-skv/babyshark-cra-M-uniform
- **`M-topk3`**: https://huggingface.co/bala-skv/babyshark-cra-M-topk3
- **`M-soft`**: https://huggingface.co/bala-skv/babyshark-cra-M-soft
- **`M-random3`**: https://huggingface.co/bala-skv/babyshark-cra-M-random3
- **`HM-topk34`**: https://huggingface.co/bala-skv/babyshark-cra-HM-topk34
- **`HM-topk`**: https://huggingface.co/bala-skv/babyshark-cra-HM-topk
- **`HM-random34`**: https://huggingface.co/bala-skv/babyshark-cra-HM-random34
- **`HM-random`**: https://huggingface.co/bala-skv/babyshark-cra-HM-random

### Attention Head Routing (H-*)
- **`H-uniform`**: https://huggingface.co/bala-skv/babyshark-cra-H-uniform
- **`H-topk112`**: https://huggingface.co/bala-skv/babyshark-cra-H-topk112
- **`H-random112`**: https://huggingface.co/bala-skv/babyshark-cra-H-random112
- **`H-topk34`**: https://huggingface.co/bala-skv/babyshark-cra-H-topk34
- **`H-random34`**: https://huggingface.co/bala-skv/babyshark-cra-H-random34
- **`H-soft`**: https://huggingface.co/bala-skv/babyshark-cra-H-soft

---

## 3. Paired Hypothesis Tests (Same Seed, Matched Budget)

| Comparison | Per-Seed Macro-F1 Differences | Mean Diff | Paired Bootstrap 95% CI | Reading / Finding |
|---|---|---|---|---|
| heads k=112: top-k minus random | -0.0004, +0.0000, -0.0070 | -0.0024 | [-0.0089, +0.0037] | not distinguishable (mixed signs, or CI includes 0) |
| heads k=112: top-k minus uniform LoRA | -0.0133, -0.0060, -0.0159 | -0.0118 | [-0.0205, -0.0035] | consistent direction and CI excludes 0 |
| heads k=34: top-k minus random | -0.0120, -0.0054, -0.0023 | -0.0065 | [-0.0137, +0.0004] | not distinguishable (mixed signs, or CI includes 0) |
| MLP layers k=3: top-k minus random | -0.0024, +0.0011, -0.0024 | -0.0012 | [-0.0112, +0.0086] | not distinguishable (mixed signs, or CI includes 0) |
| heads+MLP: top-k minus random | -0.0047, +0.0001, -0.0024 | -0.0023 | [-0.0094, +0.0047] | not distinguishable (mixed signs, or CI includes 0) |
| heads+MLP top-k minus heads-only top-k | -0.0028, -0.0054, +0.0020 | -0.0021 | [-0.0069, +0.0029] | not distinguishable (mixed signs, or CI includes 0) |
| heads+MLP at 34 heads: top-k minus random | -0.0036 | -0.0036 | [-0.0179, +0.0098] | single seed: indicative only (the interval covers evaluation noise, not seed-to-seed variation) |
| heads+MLP minus heads-only at 34 heads (top-k) | +0.0086 | +0.0086 | [-0.0037, +0.0212] | single seed: indicative only (the interval covers evaluation noise, not seed-to-seed variation) |

---

## 4. Parameter Accounting Summary

Under the revised Stage 2 accounting:
- **Uniform LoRA (q, o across all 28 layers):** 1,376,256 trainable parameters (reference).
- **Head Top-k 112:** 917,504 effective parameters (224 active half-heads in q and o).
- **Head Top-k 34 (~10%):** 512,000 effective parameters.
- **MLP Top-k 3:** 251,904 effective parameters (down_proj in 3 layers).
- **Combined HM-topk (112 heads + 3 MLP layers):** 1,169,408 effective parameters.
- **Combined HM-topk34 (34 heads + 3 MLP layers):** 763,904 effective parameters.

---

## 5. Loading Adapters with `circuit_routing`

```python
import torch
from transformers import AutoTokenizer
from circuit_routing.model_loading import load_model_and_tokenizer
from circuit_routing.masked_lora import apply_masked_lora

# Load base model
model, tokenizer = load_model_and_tokenizer('Qwen/Qwen2.5-1.5B', dtype='float16', device='cuda')

# Download and apply checkpoint from HF
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id='bala-skv/babyshark-cra-HM-topk', filename='seed0/ckpt.pt')
ckpt = torch.load(ckpt_path, map_location='cuda')
model.load_state_dict(ckpt['model_state_dict'], strict=False)
```

