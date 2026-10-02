# Reproducibility

The active implementation preserves the project's locked experimental choices:

- Base checkpoint: `Qwen/Qwen2.5-1.5B`
- Revision: `8faed761d45a263340a0528343f099c05c9a4323`
- Seeds: `0, 1, 2`
- Stage 1 precision: `fp16`
- Qwen layout: 28 layers, 12 query heads, 2 KV groups, head dimension 128
- Stage 1 scoring: mean teacher-forced full-sequence target log-probability
- Stage 1 conditions: English, Romanized Hindi, Devanagari Hindi

Run-specific configurations and generated artifacts should be retained under
`results/` or the run directory specified by the relevant configuration.
