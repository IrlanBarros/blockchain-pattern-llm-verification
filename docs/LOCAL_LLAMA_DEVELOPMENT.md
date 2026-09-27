# Local Llama Development

This guide runs the existing two-stage pipeline against an external
OpenAI-compatible server. The initial target is `llama.cpp` on CPU; the same
provider can later point to vLLM. The pipeline never starts a server, downloads
a model, or sends a local run to Gemini.

## Architecture and coupling audit

Before this integration, `requests.py` already produced a useful canonical
request containing model, messages, generation options, and the public JSON
Schema. The provider-specific parts were concentrated in `client.py`, which
created `google-genai`, called `generate_content`, interpreted Gemini response
objects, implemented retries, and exposed Gemini Batch helpers. Token usage was
read from Gemini `usage_metadata`; `artifacts.py` persisted a fixed Gemini
provider/API family. Checkpoints themselves were provider-neutral, but resume
metadata did not identify a provider or model fingerprint.

The synchronous path is now:

```text
Stage 1 / Stage 2
  -> canonical request + existing public JSON Schema
  -> provider client
       Gemini: google-genai generateContent
       OpenAI-compatible: POST /v1/chat/completions
  -> canonical response
  -> existing parsing, semantic normalization, checkpoints, aggregation
```

`ProviderConfig` and `ProviderCapabilities` select the implementation. The
OpenAI-compatible adapter translates only the transport contract; it does not
change prompts, retrieval, thresholds, catalog, schemas, aggregation, or
evaluation. Gemini stays available for historical reproduction. Remote batch
is a Gemini capability; requesting `--mode batch` with the local provider
fails explicitly.

## Requirements and model placement

Build or install a recent `llama.cpp` that includes `llama-server`. Its current
server API documents `/health`, `/v1/models`, `/v1/chat/completions`, native
JSON Schema response formats, and the input-token endpoint. See the
[official llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

A CPU-only source build can be created outside this repository with:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
export LLAMA_SERVER_BIN="$PWD/build/bin/llama-server"
```

Pin the llama.cpp commit used for research instead of building an unrecorded
moving branch, and optionally expose it through `LOCAL_LLM_SERVER_COMMIT`.

For the 16 GB CPU-only development machine, start with a 7B/8B instruct GGUF
using Q4_K_M or a comparable quantization. Put weights outside Git, for example
under `models/`; `models/`, `.cache/models/`, and `*.gguf` are ignored. No model
is downloaded automatically.

## Configuration

The local provider has these environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `local`, or `openai_compatible` |
| `LOCAL_LLM_BACKEND` | `llama.cpp` | Provenance label, for example `llama.cpp` or `vllm` |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8080` | Server root; a trailing `/v1` is normalized |
| `LOCAL_LLM_MODEL` | `local-model` | Model/alias sent to the server |
| `LOCAL_LLM_API_KEY` | unset | Optional bearer token; never persisted |
| `LOCAL_LLM_TIMEOUT_SECONDS` | `300` | Per-request HTTP timeout |
| `LOCAL_LLM_CONCURRENCY` | `1` | Recorded concurrency setting; execution is serial today |
| `LOCAL_LLM_STRUCTURED_OUTPUT` | `json_schema` | `json_schema`, `json_object`, or `prompt_only` |
| `LOCAL_LLM_SCHEMA_DIALECT` | backend-dependent | `llama_cpp` or `openai` response-format shape |
| `LOCAL_LLM_TOP_P` | `1.0` | Sampling parameter |
| `LOCAL_LLM_STOP` | unset | Stop strings separated by `||` |
| `LOCAL_LLM_CONTEXT_WINDOW` | `8192` | Context configured on the server |
| `LOCAL_LLM_MAX_ATTEMPTS` | `4` | Bounded attempts for transient/invalid responses |
| `LOCAL_LLM_COUNT_TOKENS_WHEN_MISSING` | `true` | Use the server tokenizer endpoint when usage is absent |
| `LLAMA_MODEL_PATH` | unset | Absolute/local GGUF path for provenance and server script |
| `LOCAL_LLM_MODEL_SHA256` | unset | Precomputed model hash |
| `LOCAL_LLM_HASH_MODEL` | `true` | Hash the GGUF once at run setup when no hash is supplied |
| `LOCAL_LLM_QUANTIZATION` | inferred from filename | Explicit quantization override |
| `LOCAL_LLM_COMPUTE_MODE` | `unknown` | Provenance override when GPU-layer inference is insufficient |
| `LOCAL_LLM_SERVER_VERSION` | unset | Optional server version provenance |
| `LOCAL_LLM_SERVER_COMMIT` | unset | Optional llama.cpp/vLLM commit provenance |

`--temperature`, `--top-p`, repeated `--stop`, `--seed`, and the existing
stage-specific max-token flags are persisted. Local defaults are temperature
`0` and seed `42`; these improve repeatability but do not promise bitwise
determinism.

## Start llama.cpp on CPU

In terminal 1:

```bash
export LLAMA_MODEL_PATH=/absolute/path/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
export LLAMA_MODEL_ALIAS=local-model
export LLAMA_CONTEXT_SIZE=8192
export LLAMA_THREADS=4
export LLAMA_GPU_LAYERS=0
export LLAMA_PARALLEL=1
./scripts/start_llama_server.sh
```

The script defaults to `127.0.0.1:8080`, one parallel slot, and no GPU layers.
Override `LLAMA_SERVER_BIN`, `LLAMA_HOST`, or `LLAMA_PORT` as needed. It uses
`exec` and leaves server lifecycle under operator control.

## Connectivity check

With the server running:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8080
export LOCAL_LLM_MODEL=local-model
PYTHONPATH=. .venv/bin/python scripts/check_local_llm.py
```

This manual check is not part of pytest. It probes health/model metadata, sends
one small schema-constrained request, validates its JSON, and prints usage and
latency. An offline server returns a clear non-zero result.

## First local smoke run

In terminal 2:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_BACKEND=llama.cpp
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8080
export LOCAL_LLM_MODEL=local-model
export LLAMA_MODEL_PATH=/absolute/path/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
export LOCAL_LLM_STRUCTURED_OUTPUT=json_schema
export LOCAL_LLM_SCHEMA_DIALECT=llama_cpp
export LOCAL_LLM_CONTEXT_WINDOW=8192
export LOCAL_LLM_CONCURRENCY=1

RETRIEVAL_CONFIDENCE_THRESHOLD=0.35 \
RETRIEVAL_REQUIRE_LEXICAL_ANCHOR=false \
SEMANTIC_BACKEND=lsa \
PYTHONPATH=. .venv/bin/python run_pipeline.py run \
  --input data/smoke/llama_local_smoke.csv \
  --provider local \
  --profile optimized \
  --mode sync \
  --temperature 0 \
  --seed 42 \
  --run-id llama_local_smoke_v1
```

The deterministic 12-issue sample mixes human yes/no cases, simple and hard
cases, background architecture, false friends, insufficient pattern
discussion, and Time-Constrained Access. Its selection rationale is in
`data/smoke/LLAMA_LOCAL_SMOKE_SELECTION.md`. It is a development smoke set, not
an unbiased article evaluation. The existing 200 labeled cases have already
influenced development and must now be treated as development/reference data.

## Resume and status

Interrupting with Ctrl-C preserves completed rows. Run the exact same smoke
command and run ID to resume. A different provider configuration or model
fingerprint is rejected; use a new run ID for a changed model.

Status is offline and does not contact either provider:

```bash
PYTHONPATH=. .venv/bin/python run_pipeline.py status \
  --run-dir outputs/runs/llama_local_smoke_v1
```

It reports provider/backend/models, completed and pending Stage 1/2 work,
errors, elapsed time, tokens, and throughput.

## Structured output and failures

Use `json_schema` when the server implements constrained decoding. For servers
that implement only JSON object mode, select `json_object`; `prompt_only` is an
explicit last resort. Set `LOCAL_LLM_SCHEMA_DIALECT=openai` for the nested
OpenAI/vLLM JSON Schema shape. There is no silent capability fallback.

The adapter requires exactly one JSON object and validates required fields,
types, enum values, and closed-object fields before the existing semantic
normalizers run. Empty, malformed, extra-text, truncated, schema-invalid,
timeout, connection, and retryable HTTP responses remain errors; they are never
converted into a `no` verdict. Transient transport and structurally invalid
outputs retry up to the configured bound. Semantic normalization errors remain
auditable errors rather than being silently repaired.

## Context and token accounting

An offline measurement over the frozen optimized 200-case request set found:

| Request | Maximum prompt characters | `ceil(chars/4)` diagnostic only |
|---|---:|---:|
| Stage 1 | 22,554 | 5,639 |
| Stage 2 | 21,175 | 5,294 |

The configured maximum completion is 2,048 tokens for each stage. Character
division is not a Llama token count and is retained only as a clearly labeled
fallback diagnostic. An 8,192-token context is the practical minimum tested by
budget and can be tight for the longest prompt plus maximum output; use 12,288
or 16,384 if the selected GGUF/server and available RAM permit it. The server's
reported usage is authoritative; when missing, the adapter can call the
server's own tokenizer endpoint and labels that source `server_tokenized`.

## Artifacts and telemetry

Local runs preserve the existing `stage1_results.csv`, `stage2_results.csv`,
`issue_results.csv`, raw ledgers, checkpoints, manifests, and evaluation
compatibility. Additional/relevant artifacts are:

- `run_metadata.json`: provider configuration, generation parameters, hashes,
  model path/file/size/hash when known, quantization, context, endpoint,
  CPU/GPU mode, optional server version/commit, and server information;
- `provider_runtime.json`: `/health` and `/v1/models` probe captured once per
  process;
- `token_calls.jsonl`: each attempt with token source, latency, status,
  parsing, stop reason, error, stage/issue/pattern, and optional llama timings;
- `token_summary.json` and `local_llm_summary.json`: logical calls, final
  successes/failures, retries, token totals, median/p95 latency, requests/hour,
  issues/hour, per-stage throughput, and tokens/second when supplied.

Unknown measurements are stored as `null`, never invented as zero. No API key
is written to metadata.

## Tests

The standard suite uses mocked HTTP and needs no GGUF, network, llama.cpp, GPU,
or Gemini credential:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
```

No real-provider request is made by these tests. Use the manual connectivity
check and then the 12-case smoke only when a local server is intentionally
available.

## Pointing the same provider at vLLM

No Stage code changes are needed. Configure the endpoint and schema dialect:

```bash
export LLM_PROVIDER=openai_compatible
export LOCAL_LLM_BACKEND=vllm
export LOCAL_LLM_BASE_URL=http://gpu-host:8000
export LOCAL_LLM_MODEL=your-served-model-name
export LOCAL_LLM_SCHEMA_DIALECT=openai
```

The current pipeline remains serial and `--mode batch` still means a remote
batch API, so it remains unsupported for this provider. Future concurrency or
server-side continuous batching can evolve behind the same provider contract.
