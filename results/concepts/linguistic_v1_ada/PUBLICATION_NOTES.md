# Publication and reproducibility notes

- The original concept input contains 50 concepts.
- concepts_used.json contains 45 concepts.
- Both heads_trace.json and mlp_trace.json contain 45 per-concept entries
  for each of the three contrasts. The validation report gives usable
  counts of 50, 48, and 45 for the three contrasts, respectively.
- The original REPORT.md listed 44 concepts for hi-deva_hi-latn.
  This publishing copy corrects that count to 45 based on the trace files.
- The manifest records failed initial validation and heads_analysis attempts.
  Later pipeline outputs and analyses were generated successfully. The
  original logs and manifest are retained to preserve this execution history.
- Category-level results are descriptive and should be interpreted cautiously:
  most categories contain only one to three concepts.
- The manifest records source revision 898cc18 and an RTX 2080 Ti device.

- The experiment was run from source commit 898cc18. The publishing branch is based on origin/main at 64c8df4; the manifest preserves the experiment's source revision. The result bundle does not itself contain a complete source-code snapshot.
