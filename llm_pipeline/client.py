"""Integração isolada com o SDK oficial Google Gen AI."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Iterable, Mapping

from .utils import clean_scalar, json_dumps, object_to_dict

_SUCCESS_FINISH_REASONS = {"", "STOP", "FINISH_REASON_UNSPECIFIED"}
_TERMINAL_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def is_retryable_api_error(exc: Exception) -> bool:
    """Conservatively retry transient transport/service failures only."""
    text = str(exc).casefold()
    permanent = ("unauthenticated", "invalid api key", "permission denied", "forbidden", "invalid argument", "bad request", " 400", " 401", " 403")
    if any(marker in text for marker in permanent):
        return False
    transient = ("resource_exhausted", "rate limit", "too many requests", "timeout", "timed out", "connection", "network", "unavailable", "temporar", " 429", " 500", " 502", " 503", " 504")
    return any(marker in text for marker in transient)


def is_global_api_error(exc: Exception) -> bool:
    """Errors unlikely to improve for later records in the same invocation."""
    text = str(exc).casefold()
    return any(marker in text for marker in ("resource_exhausted", "quota", "rate limit", "too many requests", "unauthenticated", "invalid api key", "permission denied", "forbidden"))


def _get(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return clean_scalar(enum_value)
    enum_name = getattr(value, "name", None)
    if enum_name is not None:
        return clean_scalar(enum_name)
    text = clean_scalar(value)
    if "." in text and text.split(".", 1)[0].endswith("Reason"):
        return text.rsplit(".", 1)[-1]
    return text


def _response_text(response: Any) -> str:
    # O atalho response.text é útil, mas pode lançar quando a resposta foi
    # bloqueada ou não possui candidato. A extração manual mantém o erro claro.
    try:
        shortcut = _get(response, "text")
        if shortcut:
            return str(shortcut).strip()
    except Exception:
        pass

    candidates = _get(response, "candidates", default=[]) or []
    parts_text: list[str] = []
    for candidate in candidates:
        content = _get(candidate, "content")
        parts = _get(content, "parts", default=[]) or []
        for part in parts:
            text = _get(part, "text")
            if text is not None:
                parts_text.append(str(text))
    return "".join(parts_text).strip()


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError(f"Resposta estruturada deve ser objeto JSON; recebido {type(parsed).__name__}")
    return parsed


def parse_response_payload(response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extrai JSON, metadados de uso e motivo de término da resposta Gemini."""
    candidates = _get(response, "candidates", default=[]) or []
    candidate = candidates[0] if candidates else None
    finish_reason = _enum_text(_get(candidate, "finish_reason", "finishReason"))
    finish_message = clean_scalar(_get(candidate, "finish_message", "finishMessage"))

    prompt_feedback = _get(response, "prompt_feedback", "promptFeedback")
    block_reason = _enum_text(_get(prompt_feedback, "block_reason", "blockReason"))
    block_message = clean_scalar(_get(prompt_feedback, "block_reason_message", "blockReasonMessage"))

    if block_reason:
        raise ValueError(
            "Prompt bloqueado pelo Gemini API: "
            f"block_reason={block_reason}; detalhe={block_message or 'não informado'}"
        )
    if finish_reason not in _SUCCESS_FINISH_REASONS:
        raise ValueError(
            "Resposta incompleta ou bloqueada: "
            f"finish_reason={finish_reason}; detalhe={finish_message or 'não informado'}"
        )

    text = _response_text(response)
    if not text:
        raise ValueError("Resposta Gemini sem conteúdo textual")

    usage = _get(response, "usage_metadata", "usageMetadata")
    usage_dict = object_to_dict(usage) if usage is not None else {}
    if not isinstance(usage_dict, Mapping):
        usage_dict = {}

    def usage_value(*names: str) -> Any:
        for name in names:
            if name in usage_dict:
                return usage_dict[name]
        return ""

    metadata = {
        "message_id": clean_scalar(_get(response, "response_id", "responseId")),
        "response_model": clean_scalar(_get(response, "model_version", "modelVersion")),
        "stop_reason": finish_reason,
        "usage": {
            "input_tokens": usage_value("prompt_token_count", "promptTokenCount"),
            "output_tokens": usage_value("candidates_token_count", "candidatesTokenCount"),
            "total_tokens": usage_value("total_token_count", "totalTokenCount"),
            "thoughts_tokens": usage_value("thoughts_token_count", "thoughtsTokenCount"),
            "cached_input_tokens": usage_value(
                "cached_content_token_count", "cachedContentTokenCount"
            ),
        },
        "raw_text": text,
    }

    parsed_attr = _get(response, "parsed")
    if isinstance(parsed_attr, Mapping):
        return dict(parsed_attr), metadata
    return _parse_json_text(text), metadata


def import_genai() -> Any:
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pacote google-genai não instalado. Execute: pip install -r requirements.txt"
        ) from exc
    return genai


def create_client() -> Any:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_key and not google_key:
        raise RuntimeError(
            "Defina GEMINI_API_KEY (recomendado) ou GOOGLE_API_KEY antes de chamar a API."
        )
    if gemini_key and google_key and gemini_key != google_key:
        raise RuntimeError(
            "GEMINI_API_KEY e GOOGLE_API_KEY estão definidas com valores diferentes. "
            "Mantenha apenas uma para evitar usar a chave errada."
        )
    genai = import_genai()
    return genai.Client()


def close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def call_sync(client: Any, params: dict[str, Any], attempts: int = 4) -> Any:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(**params)
        except Exception as exc:  # SDK expõe subclasses diferentes por versão
            last_error = exc
            if attempt == attempts or not is_retryable_api_error(exc):
                break
            print(f"[retry] tentativa {attempt}/{attempts} falhou: {exc}", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    assert last_error is not None
    raise last_error


def call_batch_create(
    client: Any,
    *,
    model: str,
    requests: list[dict[str, Any]],
    display_name: str,
    attempts: int = 4,
) -> Any:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.batches.create(
                model=model,
                src=requests,
                config={"display_name": display_name},
            )
        except Exception as exc:
            last_error = exc
            if attempt == attempts or not is_retryable_api_error(exc):
                break
            print(
                f"[batch retry] tentativa {attempt}/{attempts} falhou: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(delay * 2, 30)
    assert last_error is not None
    raise last_error


def batch_state_name(batch: Any) -> str:
    return _enum_text(_get(batch, "state"))


def wait_for_batch(client: Any, batch_name: str, poll_seconds: int) -> Any:
    while True:
        batch = client.batches.get(name=batch_name)
        state = batch_state_name(batch)
        stats = object_to_dict(_get(batch, "completion_stats", "completionStats", default={}))
        print(f"[batch] {batch_name}: {state} {stats}")
        if state in _TERMINAL_BATCH_STATES:
            return batch
        time.sleep(poll_seconds)


def batch_inlined_responses(batch: Any) -> list[Any]:
    dest = _get(batch, "dest")
    responses = _get(dest, "inlined_responses", "inlinedResponses", default=[]) or []
    return list(responses)


def batch_custom_id(inline_response: Any) -> str:
    metadata = _get(inline_response, "metadata", default={}) or {}
    return clean_scalar(_get(metadata, "custom_id", "customId"))


def batch_error_text(inline_response: Any) -> str:
    error = _get(inline_response, "error")
    if error is None:
        return ""
    return json_dumps(object_to_dict(error))


def batch_generated_response(inline_response: Any) -> Any:
    return _get(inline_response, "response")


def chunk_inline_requests(
    requests: list[dict[str, Any]],
    *,
    max_items: int,
    max_bytes: int,
) -> Iterable[list[dict[str, Any]]]:
    """Divide por quantidade e tamanho estimado para respeitar o limite de 20 MB."""
    if max_items <= 0:
        raise ValueError("max_items deve ser maior que zero")
    if max_bytes <= 0:
        raise ValueError("max_bytes deve ser maior que zero")

    current: list[dict[str, Any]] = []
    current_bytes = 2  # colchetes JSON
    for request in requests:
        request_bytes = len(json.dumps(request, ensure_ascii=False, default=str).encode("utf-8")) + 1
        if request_bytes > max_bytes:
            custom_id = clean_scalar((request.get("metadata") or {}).get("custom_id"))
            raise ValueError(
                f"Requisição {custom_id or '<sem-id>'} excede sozinha o limite de batch "
                f"configurado ({request_bytes} > {max_bytes} bytes)."
            )
        if current and (len(current) >= max_items or current_bytes + request_bytes > max_bytes):
            yield current
            current = []
            current_bytes = 2
        current.append(request)
        current_bytes += request_bytes
    if current:
        yield current
