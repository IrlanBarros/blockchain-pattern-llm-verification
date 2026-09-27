#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_MODEL_PATH:?Defina LLAMA_MODEL_PATH para um arquivo GGUF local}"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_CONTEXT_SIZE="${LLAMA_CONTEXT_SIZE:-8192}"
LLAMA_THREADS="${LLAMA_THREADS:-4}"
LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-0}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_MODEL_ALIAS="${LLAMA_MODEL_ALIAS:-local-model}"

exec "${LLAMA_SERVER_BIN}" \
  --model "${LLAMA_MODEL_PATH}" \
  --alias "${LLAMA_MODEL_ALIAS}" \
  --host "${LLAMA_HOST}" \
  --port "${LLAMA_PORT}" \
  --ctx-size "${LLAMA_CONTEXT_SIZE}" \
  --threads "${LLAMA_THREADS}" \
  --n-gpu-layers "${LLAMA_GPU_LAYERS}" \
  --parallel "${LLAMA_PARALLEL}" \
  --metrics
