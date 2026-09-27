# Offline token optimization benchmark

NOT_VALIDATED: live paired inference and populated human annotations required

uncalibrated ceil(Unicode characters/4), including schema; not official Gemini tokens

| Metric | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| Mean input estimate stage1 | 8409.58 | 2357.47 | 71.97% |
| Mean input estimate stage2 | 5097.15 | 3700.96 | 27.39% |
| Fixed workload input estimate | 2048912 | 737963 | 63.98% |
| Local preparation seconds | 0.124 | 4.933 | — |

Stage 2 workload: fixed_historical_stage1_candidates_not_gold. Outputs, model latency and stage quality were not measured.
Human positive pairs: 0; retrieval recall: None.
Historical LLM candidates excluded (not human false negatives): 31.
Full diagnostics and provenance are in report.json and retrieval_diagnostics.json.
