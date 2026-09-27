"""Provider-neutral inference configuration and OpenAI-compatible HTTP client.

The pipeline keeps one canonical request/response contract.  Gemini consumes it
directly through its SDK; OpenAI-compatible servers are adapted here without
changing Stage 1, Stage 2, schemas, retrieval, or aggregation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Mapping
from urllib import error, request
from urllib.parse import urlsplit

from .utils import sha256_file, sha256_text

PROVIDER_ALIASES = {"local": "openai_compatible", "openai-compatible": "openai_compatible"}
STRUCTURED_MODES = {"json_schema", "json_object", "prompt_only"}
SCHEMA_DIALECTS = {"llama_cpp", "openai"}


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_sync: bool = True
    supports_remote_batch: bool = False
    supports_json_schema: bool = True
    supports_json_object: bool = True
    supports_usage: bool = True
    supports_seed: bool = True
    supports_token_count: bool = True


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "gemini"
    backend: str = "google-genai"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 300.0
    concurrency: int = 1
    structured_output: str = "json_schema"
    schema_dialect: str = "llama_cpp"
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    context_window: int | None = None
    model_path: str | None = None
    model_sha256: str | None = None
    model_size_bytes: int | None = None
    quantization: str | None = None
    cpu_gpu_mode: str | None = None
    server_version: str | None = None
    server_commit: str | None = None
    count_tokens_when_missing: bool = True
    max_attempts: int = 4
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @classmethod
    def from_environment(cls, provider: str | None = None) -> "ProviderConfig":
        raw_provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).strip().lower()
        selected = PROVIDER_ALIASES.get(raw_provider, raw_provider)
        if selected not in {"gemini", "openai_compatible"}:
            raise ValueError(f"LLM_PROVIDER inválido: {raw_provider!r}")
        if selected == "gemini":
            return cls(
                provider="gemini",
                backend="google-genai",
                capabilities=ProviderCapabilities(supports_remote_batch=True),
            )

        backend = os.environ.get("LOCAL_LLM_BACKEND", "llama.cpp").strip() or "llama.cpp"
        base_url = _normalize_base_url(os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080"))
        model = os.environ.get("LOCAL_LLM_MODEL", "local-model").strip() or "local-model"
        mode = os.environ.get("LOCAL_LLM_STRUCTURED_OUTPUT", "json_schema").strip().lower()
        dialect_default = "llama_cpp" if backend.casefold() in {"llama.cpp", "llama_cpp", "llamacpp"} else "openai"
        dialect = os.environ.get("LOCAL_LLM_SCHEMA_DIALECT", dialect_default).strip().lower()
        if mode not in STRUCTURED_MODES:
            raise ValueError(f"LOCAL_LLM_STRUCTURED_OUTPUT inválido: {mode!r}")
        if dialect not in SCHEMA_DIALECTS:
            raise ValueError(f"LOCAL_LLM_SCHEMA_DIALECT inválido: {dialect!r}")
        timeout = _float_env("LOCAL_LLM_TIMEOUT_SECONDS", 300.0, minimum=0.001)
        concurrency = _int_env("LOCAL_LLM_CONCURRENCY", 1, minimum=1)
        context = _int_env("LOCAL_LLM_CONTEXT_WINDOW", 8192, minimum=512)
        top_p = _float_env("LOCAL_LLM_TOP_P", 1.0, minimum=0.0, maximum=1.0)
        attempts = _int_env("LOCAL_LLM_MAX_ATTEMPTS", 4, minimum=1)
        stop = tuple(x for x in os.environ.get("LOCAL_LLM_STOP", "").split("||") if x)
        model_path_raw = os.environ.get("LLAMA_MODEL_PATH", "").strip()
        model_path = str(Path(model_path_raw).expanduser().resolve()) if model_path_raw else None
        model_size = Path(model_path).stat().st_size if model_path and Path(model_path).is_file() else None
        supplied_hash = os.environ.get("LOCAL_LLM_MODEL_SHA256", "").strip() or None
        # Hash at configuration time (once per process/run), never per request.
        # This makes resume conservative even when a GGUF is replaced under the
        # same filename and model alias.
        should_hash = _bool_env("LOCAL_LLM_HASH_MODEL", True)
        model_hash = supplied_hash or (sha256_file(Path(model_path)) if should_hash and model_path and Path(model_path).is_file() else None)
        quantization = os.environ.get("LOCAL_LLM_QUANTIZATION", "").strip() or _quantization_from_path(model_path)
        gpu_layers = os.environ.get("LLAMA_GPU_LAYERS", "").strip()
        mode_name = "cpu" if gpu_layers == "0" else os.environ.get("LOCAL_LLM_COMPUTE_MODE", "unknown").strip()
        caps = ProviderCapabilities(
            supports_remote_batch=False,
            supports_json_schema=mode == "json_schema",
            supports_json_object=mode in {"json_schema", "json_object"},
            supports_usage=True,
            supports_seed=True,
            supports_token_count=_bool_env("LOCAL_LLM_COUNT_TOKENS_WHEN_MISSING", True),
        )
        return cls(
            provider="openai_compatible", backend=backend, base_url=base_url, model=model,
            api_key=os.environ.get("LOCAL_LLM_API_KEY") or None, timeout_seconds=timeout,
            concurrency=concurrency, structured_output=mode, schema_dialect=dialect,
            top_p=top_p, stop=stop, context_window=context, model_path=model_path,
            model_sha256=model_hash, model_size_bytes=model_size, quantization=quantization,
            cpu_gpu_mode=mode_name or "unknown", count_tokens_when_missing=caps.supports_token_count,
            server_version=os.environ.get("LOCAL_LLM_SERVER_VERSION", "").strip() or None,
            server_commit=os.environ.get("LOCAL_LLM_SERVER_COMMIT", "").strip() or None,
            max_attempts=attempts, capabilities=caps,
        )

    @property
    def fingerprint(self) -> str:
        return sha256_text(json.dumps(self.public_metadata(), ensure_ascii=False, sort_keys=True))

    def public_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("api_key", None)
        data["stop"] = list(self.stop)
        data["base_url"] = self.base_url
        data["model_file"] = Path(self.model_path).name if self.model_path else None
        data["generation_endpoint"] = "/v1/chat/completions" if self.provider == "openai_compatible" else None
        return data


class ProviderError(RuntimeError):
    retryable = False
    global_error = False

    def __init__(self, message: str, *, response: Any = None):
        super().__init__(message)
        self.response = response


class ProviderTransportError(ProviderError):
    retryable = True


class ProviderHTTPError(ProviderError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retryable = status in {408, 409, 425, 429, 500, 502, 503, 504}
        self.global_error = status in {401, 403}


class InvalidProviderResponse(ProviderError):
    retryable = True


class OpenAICompatibleClient:
    """Small stdlib HTTP client exposing the SDK shape expected by existing stages."""

    def __init__(self, config: ProviderConfig):
        self.provider_config = config
        self.models = _OpenAICompatibleModels(self)
        self.runtime_metadata: dict[str, Any] = {}

    def close(self) -> None:
        return None

    def probe(self) -> dict[str, Any]:
        health = self._json_request("GET", "/health")
        models = self._json_request("GET", "/v1/models")
        first = (models.get("data") or [{}])[0] if isinstance(models, Mapping) else {}
        self.runtime_metadata = {"health": health, "server_model": first}
        return self.runtime_metadata

    def _json_request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.provider_config.base_url is not None
        url = self.provider_config.base_url + endpoint
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.provider_config.api_key:
            headers["Authorization"] = f"Bearer {self.provider_config.api_key}"
        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.provider_config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderHTTPError(exc.code, detail[:2000]) from exc
        except (error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise ProviderTransportError(f"connection/timeout calling {url}: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidProviderResponse(f"HTTP response is not JSON: {raw[:500]!r}") from exc
        if not isinstance(parsed, dict):
            raise InvalidProviderResponse(f"HTTP response must be an object, got {type(parsed).__name__}")
        if parsed.get("error"):
            raise ProviderError(f"model error: {parsed['error']}")
        return parsed


class _OpenAICompatibleModels:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def generate_content(self, **params: Any) -> dict[str, Any]:
        cfg = self.client.provider_config
        generation = params.get("config", {})
        messages = []
        system = generation.get("system_instruction", "")
        if system:
            messages.append({"role": "system", "content": system})
        for content in params.get("contents", []):
            text = "".join(str(part.get("text", "")) for part in content.get("parts", []))
            messages.append({"role": content.get("role", "user"), "content": text})
        payload: dict[str, Any] = {
            "model": params.get("model") or cfg.model,
            "messages": messages,
            "temperature": generation.get("temperature", 0.0),
            "top_p": cfg.top_p,
            "max_tokens": generation.get("max_output_tokens"),
            "seed": generation.get("seed"),
            "stream": False,
        }
        if cfg.stop:
            payload["stop"] = list(cfg.stop)
        schema = generation.get("response_json_schema")
        if cfg.structured_output == "json_schema":
            payload["response_format"] = _schema_response_format(schema, cfg.schema_dialect)
        elif cfg.structured_output == "json_object":
            payload["response_format"] = _json_object_response_format(schema, cfg.schema_dialect)
        if cfg.structured_output in {"json_object", "prompt_only"} and schema:
            schema_instruction = "Return only one JSON object matching this schema exactly:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n\n" + schema_instruction
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
        raw = self.client._json_request("POST", "/v1/chat/completions", payload)
        response = _canonical_response(raw, cfg)
        if response["provider_metadata"]["stop_reason"] != "STOP":
            response["provider_metadata"]["response_status"] = "truncated"
            raise InvalidProviderResponse(
                "incomplete structured response: "
                f"finish_reason={response['provider_metadata']['stop_reason']}",
                response=response,
            )
        text = response["text"]
        try:
            parsed = _strict_json_object(text)
            if schema:
                _validate_schema(parsed, schema)
        except (ValueError, json.JSONDecodeError) as exc:
            response["provider_metadata"]["parsing_status"] = "invalid"
            raise InvalidProviderResponse(f"invalid structured response: {exc}", response=response) from exc
        response["parsed"] = parsed
        response["provider_metadata"]["parsing_status"] = "valid"
        if response["usage_metadata"].get("prompt_token_count") is None and cfg.count_tokens_when_missing:
            counted = self._count_tokens(payload)
            if counted is not None:
                response["usage_metadata"]["prompt_token_count"] = counted
                response["provider_metadata"]["input_token_source"] = "server_tokenized"
        return response

    def _count_tokens(self, payload: dict[str, Any]) -> int | None:
        try:
            result = self.client._json_request("POST", "/v1/chat/completions/input_tokens", payload)
        except ProviderError:
            return None
        value = result.get("input_tokens")
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _canonical_response(raw: dict[str, Any], cfg: ProviderConfig) -> dict[str, Any]:
    choices = raw.get("choices") or []
    if not choices:
        raise InvalidProviderResponse("empty choices in OpenAI-compatible response", response=raw)
    first = choices[0]
    message = first.get("message") or {}
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise InvalidProviderResponse("empty response content", response=raw)
    finish = str(first.get("finish_reason") or "").lower()
    stop_reason = "STOP" if finish in {"", "stop"} else "MAX_TOKENS" if finish in {"length", "max_tokens"} else finish.upper()
    usage = raw.get("usage") or {}
    timings = raw.get("timings") or {}
    provider_metadata = {
        "provider": cfg.provider, "backend": cfg.backend, "base_url": cfg.base_url,
        "structured_output": cfg.structured_output, "schema_dialect": cfg.schema_dialect,
        "response_status": "succeeded", "parsing_status": "pending",
        "stop_reason": stop_reason,
        "input_token_source": "provider_reported" if usage.get("prompt_tokens") is not None else "unknown",
        "output_token_source": "provider_reported" if usage.get("completion_tokens") is not None else "unknown",
        "prompt_tokens_per_second": _number(timings, "prompt_per_second", "prompt_tokens_per_second"),
        "generation_tokens_per_second": _number(timings, "predicted_per_second", "generation_tokens_per_second"),
        "prompt_ms": _number(timings, "prompt_ms"), "generation_ms": _number(timings, "predicted_ms", "generation_ms"),
    }
    return {
        "text": text.strip(), "parsed": None, "response_id": raw.get("id", ""),
        "model_version": raw.get("model", cfg.model or ""),
        "candidates": [{"finish_reason": stop_reason, "finish_message": "", "content": {"parts": [{"text": text}]}}],
        "usage_metadata": {
            "prompt_token_count": usage.get("prompt_tokens"),
            "candidates_token_count": usage.get("completion_tokens"),
            "total_token_count": usage.get("total_tokens"),
            "thoughts_token_count": None, "cached_content_token_count": _cached_tokens(usage),
        },
        "provider_metadata": provider_metadata, "openai_response": raw,
    }


def _schema_response_format(schema: dict[str, Any], dialect: str) -> dict[str, Any]:
    if dialect == "llama_cpp":
        return {"type": "json_schema", "schema": schema}
    return {"type": "json_schema", "json_schema": {"name": "pipeline_response", "strict": True, "schema": schema}}


def _json_object_response_format(schema: dict[str, Any] | None, dialect: str) -> dict[str, Any]:
    response_format: dict[str, Any] = {"type": "json_object"}
    if dialect == "llama_cpp" and schema is not None:
        response_format["schema"] = schema
    return response_format


def _strict_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}: lines = lines[1:]
        if lines and lines[-1].strip() == "```": lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict): raise ValueError("structured response must be a JSON object")
    return value


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict): raise ValueError(f"{path} must be object")
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing: raise ValueError(f"{path} missing required fields: {missing}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(schema.get("properties", {}))
            if extras: raise ValueError(f"{path} has unknown fields: {sorted(extras)}")
        for key, item in value.items():
            if key in schema.get("properties", {}): _validate_schema(item, schema["properties"][key], f"{path}.{key}")
    elif expected == "array":
        if not isinstance(value, list): raise ValueError(f"{path} must be array")
        for index, item in enumerate(value): _validate_schema(item, schema.get("items", {}), f"{path}[{index}]")
    elif expected == "string" and not isinstance(value, str): raise ValueError(f"{path} must be string")
    elif expected == "boolean" and not isinstance(value, bool): raise ValueError(f"{path} must be boolean")
    if "enum" in schema and value not in schema["enum"]: raise ValueError(f"{path} has invalid enum value {value!r}")


def _cached_tokens(usage: Mapping[str, Any]) -> Any:
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") if isinstance(details, Mapping) else None


def _number(mapping: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool): return float(value)
    return None


def _normalize_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if url.endswith("/v1"): url = url[:-3]
    if not re.match(r"^https?://", url): raise ValueError("LOCAL_LLM_BASE_URL deve usar http:// ou https://")
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        raise ValueError("LOCAL_LLM_BASE_URL não pode conter credenciais; use LOCAL_LLM_API_KEY")
    return url


def _quantization_from_path(path: str | None) -> str | None:
    if not path: return None
    match = re.search(r"(?:^|[-_.])(Q\d(?:_[A-Z0-9]+)+)(?:[-_.]|$)", Path(path).name, re.I)
    return match.group(1).upper() if match else None


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None: return default
    if value.strip().lower() in {"1", "true", "yes", "on"}: return True
    if value.strip().lower() in {"0", "false", "no", "off"}: return False
    raise ValueError(f"{name} deve ser booleano")


def _int_env(name: str, default: int, *, minimum: int) -> int:
    try: value = int(os.environ.get(name, default))
    except ValueError as exc: raise ValueError(f"{name} deve ser inteiro") from exc
    if value < minimum: raise ValueError(f"{name} deve ser >= {minimum}")
    return value


def _float_env(name: str, default: float, *, minimum: float, maximum: float | None = None) -> float:
    try: value = float(os.environ.get(name, default))
    except ValueError as exc: raise ValueError(f"{name} deve ser numérico") from exc
    if value < minimum or (maximum is not None and value > maximum): raise ValueError(f"{name} fora do intervalo permitido")
    return value
