# Stage 1 report: `linguistic_v1_ada`

Generated 2026-10-30T23:13:57Z by `scripts/pipeline/run_concepts.py`; git `898cc18`; GPU `NVIDIA GeForce RTX 2080 Ti`; concept file `/home2/ashish123/InterpAdapt-Hinglish-finetuning/data/tracing/linguistic_v1.json` (sha256 `19d9d3b16163df2e`).

Numbers below are copied from the step outputs in this folder. Section 8 says how to read them.

## 1. Concept set

45 concepts used, 20 categories.

| Category | Concepts |
|---|---|
| Animals | 3 (fewer than 5: noisy) |
| Cognition, emotion, state, and relation | 3 (fewer than 5: noisy) |
| Events and natural phenomena | 3 (fewer than 5: noisy) |
| People, kinship, and social roles | 3 (fewer than 5: noisy) |
| Places and locations | 3 (fewer than 5: noisy) |
| Plants and natural objects | 3 (fewer than 5: noisy) |
| Affective and evaluative properties | 2 (fewer than 5: noisy) |
| Artifacts and tools | 2 (fewer than 5: noisy) |
| Body and sensory organs | 2 (fewer than 5: noisy) |
| Cognition and emotion | 2 (fewer than 5: noisy) |
| Colour and sensory properties | 2 (fewer than 5: noisy) |
| Communication | 2 (fewer than 5: noisy) |
| Contact, creation, and consumption | 2 (fewer than 5: noisy) |
| Food and substances | 2 (fewer than 5: noisy) |
| Manner, time, frequency, and location | 2 (fewer than 5: noisy) |
| Motion and posture | 2 (fewer than 5: noisy) |
| Perception and sensory experience | 2 (fewer than 5: noisy) |
| Social interaction, possession, and change | 2 (fewer than 5: noisy) |
| Time, quantity, and shape | 2 (fewer than 5: noisy) |
| Size, temperature, and physical state | 1 (fewer than 5: noisy) |

## 2. Validation

| Contrast | Usable | Mean contrast | Min contrast |
|---|---|---|---|
| en<->hi_deva | 50/50 | 8.26 | 2.45 |
| en<->hi_latn | 48/50 | 11.41 | 4.07 |
| hi_deva<->hi_latn | 45/50 | 2.09 | -0.19 |

## 3. Single heads

**en_hi-latn** (45 concepts, mean contrast 11.68)

- Family-wise threshold 0.0168; significant: L21H2, L22H0, L22H6, L16H9, L9H3, L22H4, L15H11, L18H11, L17H7, L21H5, L20H2, L0H8, L15H9, L21H3, L10H2, L5H5, L2H5, L20H8
- Heads with a 95% interval above zero: 117 (about 8.4 expected by chance)

| Head | Mean recovery | 95% CI | In top 5 (bootstrap) | Leave-one-out max shift |
|---|---|---|---|---|
| L21H2 | +0.0450 | [+0.0345, +0.0551] | 100% | 5% (E300-225) |
| L22H0 | +0.0439 | [+0.0369, +0.0503] | 100% | 4% (E300-284) |
| L22H6 | +0.0421 | [+0.0310, +0.0536] | 94% | 7% (E300-256) |
| L16H9 | +0.0393 | [+0.0328, +0.0468] | 97% | 6% (E300-225) |
| L9H3 | +0.0383 | [+0.0314, +0.0462] | 93% | 8% (E300-284) |

**hi-deva_hi-latn** (45 concepts, mean contrast 2.31)

- Family-wise threshold 0.0841; significant: L21H2, L22H6, L25H8, L17H6, L17H7
- Heads with a 95% interval above zero: 118 (about 8.4 expected by chance)

| Head | Mean recovery | 95% CI | In top 5 (bootstrap) | Leave-one-out max shift |
|---|---|---|---|---|
| L21H2 | +0.2974 | [+0.2635, +0.3340] | 100% | 2% (E300-164) |
| L22H6 | +0.1979 | [+0.1681, +0.2326] | 100% | 6% (E300-111) |
| L25H8 | +0.1276 | [+0.1074, +0.1480] | 100% | 3% (E300-284) |
| L17H6 | +0.1158 | [+0.0976, +0.1354] | 100% | 4% (E300-284) |
| L17H7 | +0.1078 | [+0.0893, +0.1292] | 94% | 6% (E300-111) |

**en_hi-deva** (45 concepts, mean contrast 8.32)

- Family-wise threshold 0.0302; significant: L22H6, L21H2, L27H2, L16H9, L17H7, L18H10, L21H5, L22H0, L9H3, L25H8, L22H4, L9H2
- Heads with a 95% interval above zero: 126 (about 8.4 expected by chance)

| Head | Mean recovery | 95% CI | In top 5 (bootstrap) | Leave-one-out max shift |
|---|---|---|---|---|
| L22H6 | +0.1021 | [+0.0860, +0.1215] | 100% | 4% (E300-076) |
| L21H2 | +0.0908 | [+0.0752, +0.1084] | 100% | 5% (E300-076) |
| L27H2 | +0.0879 | [+0.0749, +0.1039] | 100% | 4% (E300-286) |
| L16H9 | +0.0688 | [+0.0605, +0.0777] | 100% | 2% (E300-016) |
| L17H7 | +0.0545 | [+0.0457, +0.0645] | 85% | 5% (E300-076) |


## 4. Leading heads per category (descriptive)

Mean recovery of the three best heads within each category. No multiple-comparison correction; use this to generate hypotheses, not to claim them.

**en_hi-latn**

| Category | n | Top heads (mean recovery) |
|---|---|---|
| Animals | 3 | L21H2 +0.064, L22H6 +0.061, L16H9 +0.050 |
| Cognition, emotion, state, and relation | 3 | L22H4 +0.044, L22H0 +0.041, L21H2 +0.040 |
| Events and natural phenomena | 3 | L22H4 +0.068, L22H0 +0.058, L21H0 +0.051 |
| People, kinship, and social roles | 3 | L21H2 +0.065, L22H0 +0.057, L9H3 +0.054 |
| Places and locations | 3 | L22H0 +0.058, L22H6 +0.053, L21H2 +0.047 |
| Plants and natural objects | 3 | L22H0 +0.066, L21H2 +0.053, L16H9 +0.052 |
| Affective and evaluative properties | 2 | L21H2 +0.109, L11H8 +0.094, L9H2 +0.090 |
| Artifacts and tools | 2 | L21H2 +0.070, L10H2 +0.060, L9H3 +0.049 |
| Body and sensory organs | 2 | L22H0 +0.049, L21H2 +0.038, L22H4 +0.036 |
| Cognition and emotion | 2 | L2H5 +0.030, L22H0 +0.024, L16H9 +0.023 |
| Colour and sensory properties | 2 | L14H9 +0.045, L9H3 +0.043, L6H5 +0.040 |
| Communication | 2 | L22H6 +0.036, L20H8 +0.032, L8H6 +0.030 |
| Contact, creation, and consumption | 2 | L16H9 +0.097, L21H2 +0.093, L22H6 +0.089 |
| Food and substances | 2 | L23H5 +0.070, L9H3 +0.057, L22H0 +0.050 |
| Manner, time, frequency, and location | 2 | L7H8 +0.044, L11H8 +0.037, L20H5 +0.035 |
| Motion and posture | 2 | L16H9 +0.050, L0H8 +0.036, L21H2 +0.036 |
| Perception and sensory experience | 2 | L16H9 +0.051, L10H2 +0.041, L22H0 +0.040 |
| Social interaction, possession, and change | 2 | L2H5 +0.038, L9H3 +0.032, L10H2 +0.025 |
| Time, quantity, and shape | 2 | L15H11 +0.076, L10H2 +0.070, L22H0 +0.062 |
| Size, temperature, and physical state | 1 | L22H6 +0.167, L21H2 +0.091, L16H9 +0.068 |

**hi-deva_hi-latn**

| Category | n | Top heads (mean recovery) |
|---|---|---|
| Animals | 3 | L21H2 +0.239, L27H2 +0.149, L25H8 +0.124 |
| Cognition, emotion, state, and relation | 3 | L21H2 +0.371, L25H8 +0.150, L17H6 +0.147 |
| Events and natural phenomena | 3 | L22H6 +0.321, L26H3 +0.300, L21H2 +0.296 |
| People, kinship, and social roles | 3 | L21H2 +0.329, L22H6 +0.179, L25H8 +0.133 |
| Plants and natural objects | 3 | L21H2 +0.447, L22H6 +0.243, L25H8 +0.208 |
| Affective and evaluative properties | 2 | L21H2 +0.284, L22H6 +0.260, L25H8 +0.200 |
| Artifacts and tools | 2 | L21H2 +0.231, L25H8 +0.132, L22H6 +0.098 |
| Body and sensory organs | 2 | L21H2 +0.229, L22H6 +0.139, L25H8 +0.123 |
| Cognition and emotion | 2 | L21H2 +0.258, L22H6 +0.156, L17H6 +0.123 |
| Colour and sensory properties | 2 | L22H6 +0.308, L21H2 +0.265, L10H5 +0.149 |
| Communication | 2 | L21H2 +0.323, L22H6 +0.194, L17H6 +0.122 |
| Contact, creation, and consumption | 2 | L21H2 +0.372, L22H6 +0.318, L25H8 +0.169 |
| Food and substances | 2 | L21H2 +0.270, L27H2 +0.195, L22H6 +0.148 |
| Manner, time, frequency, and location | 2 | L22H6 +0.164, L21H2 +0.158, L27H2 +0.105 |
| Motion and posture | 2 | L21H2 +0.469, L22H6 +0.274, L25H8 +0.182 |
| Perception and sensory experience | 2 | L21H2 +0.273, L22H6 +0.206, L17H6 +0.110 |
| Places and locations | 2 | L21H2 +0.326, L22H6 +0.201, L25H8 +0.142 |
| Social interaction, possession, and change | 2 | L21H2 +0.202, L22H6 +0.138, L17H7 +0.060 |
| Time, quantity, and shape | 2 | L21H2 +0.179, L22H6 +0.088, L17H6 +0.081 |
| Size, temperature, and physical state | 1 | L21H2 +0.359, L22H6 +0.278, L25H3 +0.237 |

**en_hi-deva**

| Category | n | Top heads (mean recovery) |
|---|---|---|
| Animals | 3 | L16H9 +0.094, L22H6 +0.086, L27H2 +0.082 |
| Cognition, emotion, state, and relation | 3 | L19H6 +0.103, L18H10 +0.084, L22H6 +0.084 |
| Events and natural phenomena | 3 | L22H6 +0.078, L27H2 +0.077, L16H9 +0.076 |
| People, kinship, and social roles | 3 | L21H2 +0.180, L22H6 +0.150, L27H2 +0.126 |
| Places and locations | 3 | L22H6 +0.156, L21H2 +0.136, L16H9 +0.105 |
| Plants and natural objects | 3 | L27H2 +0.061, L21H2 +0.051, L22H0 +0.048 |
| Affective and evaluative properties | 2 | L22H6 +0.153, L9H2 +0.145, L27H2 +0.131 |
| Artifacts and tools | 2 | L21H2 +0.086, L9H0 +0.077, L22H6 +0.071 |
| Body and sensory organs | 2 | L21H2 +0.138, L27H2 +0.113, L22H6 +0.099 |
| Cognition and emotion | 2 | L21H2 +0.087, L22H6 +0.083, L27H2 +0.072 |
| Colour and sensory properties | 2 | L22H6 +0.187, L27H2 +0.109, L21H2 +0.098 |
| Communication | 2 | L10H2 +0.059, L18H2 +0.058, L22H6 +0.056 |
| Contact, creation, and consumption | 2 | L21H2 +0.123, L22H6 +0.114, L27H2 +0.083 |
| Food and substances | 2 | L23H5 +0.080, L18H10 +0.075, L27H2 +0.059 |
| Manner, time, frequency, and location | 2 | L27H2 +0.141, L22H6 +0.134, L0H8 +0.097 |
| Motion and posture | 2 | L22H6 +0.142, L27H2 +0.140, L21H2 +0.104 |
| Perception and sensory experience | 2 | L27H2 +0.070, L21H2 +0.070, L10H2 +0.059 |
| Social interaction, possession, and change | 2 | L21H2 +0.098, L22H6 +0.068, L27H2 +0.063 |
| Time, quantity, and shape | 2 | L22H6 +0.121, L21H2 +0.119, L10H2 +0.101 |
| Size, temperature, and physical state | 1 | L22H6 +0.134, L27H2 +0.075, L16H9 +0.064 |


## 5. MLP layers

| Contrast | max-\|mean\| threshold | Significant | max-\|t\| threshold | Significant | Top 5 layers (mean) |
|---|---|---|---|---|---|
| en_hi-latn | 0.0506 | L5, L12, L15, L17, L21, L22, L23, L24, L25, L27 | 3.12 | L1, L3, L5, L7, L9, L10, L12, L15, L16, L17, L18, L21, L22, L23, L24, L25, L27 | L17 +0.127, L22 +0.123, L23 +0.107, L24 +0.096, L15 +0.073 |
| hi-deva_hi-latn | 0.1487 | L8, L9, L10, L21, L22, L24, L25, L27 | 3.05 | L0, L1, L2, L3, L4, L5, L7, L8, L9, L10, L14, L17, L19, L21, L22, L23, L24, L25, L27 | L22 +0.304, L9 +0.207, L25 +0.194, L21 +0.164, L24 +0.151 |
| en_hi-deva | 0.1030 | L11, L17, L22, L23, L24, L25, L27 | 3.11 | L0, L1, L2, L3, L5, L7, L8, L9, L11, L13, L15, L17, L18, L20, L22, L23, L24, L25 | L25 +0.167, L22 +0.167, L23 +0.164, L27 +0.152, L24 +0.137 |

Figures: `figures/concepts/linguistic_v1_ada/mlp/`.

## 6. Head interactions

### en_hi-latn

Joint patching curve (top-k heads patched together vs the sum of their single-head recoveries):

| k | Joint | Sum of singles | Random k | Joint - sum [95% CI] |
|---|---|---|---|---|
| 1 | +0.045 | +0.045 | -0.003 | +0.000 [+0.000, +0.000] |
| 2 | +0.090 | +0.089 | +0.000 | +0.001 [-0.000, +0.002] |
| 3 | +0.149 | +0.131 | -0.005 | +0.018 [+0.013, +0.023] |
| 5 | +0.237 | +0.209 | +0.018 | +0.029 [+0.020, +0.037] |
| 10 | +0.393 | +0.325 | +0.007 | +0.068 [+0.049, +0.087] |
| 20 | +0.598 | +0.500 | +0.031 | +0.099 [+0.050, +0.141] |
| 50 | +0.863 | +0.830 | +0.072 | +0.033 [-0.086, +0.141] |
| 100 | +0.952 | +1.061 | +0.040 | -0.109 [-0.287, +0.040] |
| 336 | +0.891 | +0.268 | - | +0.623 [+0.510, +0.733] |

Necessity: all heads together +0.848; correlation with sufficiency +0.62; top five: L21H2 +0.0720 (suff. +0.0450), L21H0 +0.0668 (suff. +0.0145), L22H4 +0.0665 (suff. +0.0325), L16H9 +0.0613 (suff. +0.0393), L20H11 +0.0589 (suff. +0.0033).

Pairs: 1225 shortlist pairs, 190 random pairs, 45 concepts. Random-pair interactions (2.5-97.5%): [-0.0014, +0.0011].

| Single-head baseline | max-\|mean\| threshold | Significant | max-\|t\| threshold | Significant |
|---|---|---|---|---|
| v2 | 0.0176 | 2 | 4.26 | 166 |
| same-run | 0.0171 | 2 | 4.23 | 170 |

Robust pairs (max-|t| under both baselines): **166**

| Pair | I | t | 95% CI |
|---|---|---|---|
| L21H2+L21H3 | +0.0081 | +11.8 | [+0.0068, +0.0094] |
| L17H8+L21H2 | +0.0054 | +10.4 | [+0.0044, +0.0064] |
| L22H0+L22H4 | +0.0117 | +9.8 | [+0.0095, +0.0140] |
| L20H8+L20H9 | +0.0085 | +9.4 | [+0.0069, +0.0104] |
| L6H5+L10H5 | +0.0055 | +9.4 | [+0.0044, +0.0066] |
| L16H9+L17H8 | +0.0076 | +9.3 | [+0.0060, +0.0092] |
| L17H6+L17H8 | +0.0033 | +8.8 | [+0.0026, +0.0041] |
| L17H7+L21H2 | +0.0098 | +8.5 | [+0.0076, +0.0118] |
| L16H9+L17H6 | +0.0041 | +8.4 | [+0.0031, +0.0050] |
| L17H6+L21H2 | +0.0058 | +8.4 | [+0.0045, +0.0071] |
| L20H8+L22H4 | +0.0046 | +8.3 | [+0.0035, +0.0058] |
| L18H11+L19H11 | -0.0044 | -7.9 | [-0.0055, -0.0034] |
| L5H2+L5H5 | +0.0462 | +7.7 | [+0.0358, +0.0587] |
| L20H9+L20H11 | +0.0061 | +7.6 | [+0.0046, +0.0075] |
| L17H8+L21H3 | +0.0021 | +7.6 | [+0.0015, +0.0026] |
| L17H7+L22H6 | +0.0075 | +7.6 | [+0.0056, +0.0093] |
| L17H6+L20H2 | +0.0015 | +7.5 | [+0.0011, +0.0019] |
| L20H9+L22H4 | +0.0033 | +7.5 | [+0.0025, +0.0042] |
| L17H7+L17H8 | +0.0119 | +7.5 | [+0.0091, +0.0152] |
| L17H7+L21H3 | +0.0034 | +7.4 | [+0.0026, +0.0043] |
| L8H5+L22H4 | +0.0025 | +7.3 | [+0.0018, +0.0032] |
| L22H3+L22H4 | +0.0045 | +7.3 | [+0.0034, +0.0058] |
| L17H6+L17H7 | +0.0066 | +7.3 | [+0.0050, +0.0083] |
| L20H2+L21H2 | +0.0027 | +7.3 | [+0.0020, +0.0034] |
| L10H2+L16H9 | +0.0056 | +7.2 | [+0.0041, +0.0071] |
| L8H5+L17H7 | +0.0029 | +7.1 | [+0.0021, +0.0036] |
| L8H5+L18H11 | +0.0016 | +7.0 | [+0.0012, +0.0021] |
| L10H10+L17H8 | +0.0017 | +7.0 | [+0.0012, +0.0021] |
| L5H5+L18H11 | +0.0026 | +6.9 | [+0.0019, +0.0033] |
| L20H8+L22H0 | +0.0035 | +6.9 | [+0.0025, +0.0045] |

(136 more in `entangle/en_hi-latn/analysis.json`.)

Diagnostics: same-layer pairs are 3.8% of the shortlist pairs and 18 of the 166 robust pairs; mean interaction +0.00403 for same-layer pairs vs +0.00055 for cross-layer pairs. Spearman correlation of I with the product of the two single-head effects: +0.37.

Figures: `figures/concepts/linguistic_v1_ada/entangle/en_hi-latn/`.


## 7. Steps

| Step | Status | Minutes | GPU |
|---|---|---|---|
| check | ok | 0.0 |  |
| validate | failed | 5.8 |  |
| heads | ok | 63.4 | NVIDIA GeForce RTX 2080 Ti |
| heads_analysis | failed | 0.1 |  |
| mlp | ok | 19.8 | NVIDIA GeForce RTX 2080 Ti |
| mlp_analysis | ok | 0.1 |  |
| curve | ok | 3.7 | NVIDIA GeForce RTX 2080 Ti |
| noise | ok | 21.5 | NVIDIA GeForce RTX 2080 Ti |
| shortlist | ok | 0.0 |  |
| pairs | ok | 120.4 | NVIDIA GeForce RTX 2080 Ti |
| interactions | ok | 0.1 |  |
| plots | ok | 0.1 |  |
| report | running | 0.0 |  |

## 8. How to read this report

- **Recovery** is the fraction of the clean-corrupted contrast restored by patching, at the last prompt token; 1 means fully restored.
- **Family-wise tests** control the chance of any false positive across all heads, layers or pairs tested. max-|mean| uses one threshold for everything (it was the test planned for the pairs); max-|t| standardises each item by its spread across concepts and was added after a few noisy early-layer pairs inflated the max-|mean| threshold. Report both.
- **Robust pairs** pass max-|t| with both single-head baselines (v2 trace of this run, and the singles re-measured inside the pairs run). They are the conservative claim.
- **A significant interaction** means joint patching is non-additive for that pair on these concepts. It is not by itself evidence of a mechanism; see the same-layer and curvature diagnostics.
- **Per-category tables** are descriptive: no correction, and categories with few concepts are noisy.
- **Shortlist**: pairs are tested only among the shortlist heads, so heads outside it are untested, not ruled out.
