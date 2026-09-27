# Offline token optimization benchmark

NOT_VALIDATED: live paired inference and populated human annotations required

uncalibrated ceil(Unicode characters/4), including schema; not official Gemini tokens

| Metric | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| Mean input estimate stage1 | 8409.58 | 4460.59 | 46.96% |
| Mean input estimate stage2 | 5097.15 | 3700.96 | 27.39% |
| Fixed workload input estimate | 2048912 | 1158587 | 43.45% |
| Local preparation seconds | 0.119 | 5.010 | — |

Stage 2 workload: fixed_historical_stage1_candidates_not_gold. Outputs, model latency and stage quality were not measured.
Human positive pairs: 0; retrieval recall: None.
Historical LLM candidates excluded (not human false negatives): 0.
Full diagnostics and provenance are in report.json and retrieval_diagnostics.json.
