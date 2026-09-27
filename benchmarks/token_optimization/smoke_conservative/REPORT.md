# Offline token optimization benchmark

NOT_VALIDATED: live paired inference and populated human annotations required

uncalibrated ceil(Unicode characters/4), including schema; not official Gemini tokens

| Metric | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| Mean input estimate stage1 | 8644.70 | 4594.63 | 46.85% |
| Mean input estimate stage2 | 5278.17 | 3815.49 | 27.71% |
| Fixed workload input estimate | 13243632 | 9523953 | 28.09% |
| Local preparation seconds | 0.144 | 6.648 | — |

Stage 2 workload: all_patterns_probe_not_actual_pipeline_workload. Outputs, model latency and stage quality were not measured.
Human positive pairs: 0; retrieval recall: None.
Historical LLM candidates excluded (not human false negatives): 0.
Full diagnostics and provenance are in report.json and retrieval_diagnostics.json.
