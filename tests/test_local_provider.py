from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import socket
from urllib import error

import pytest

from llm_pipeline.cli import _validate_resume_metadata, build_parser, main
from llm_pipeline.client import call_sync, parse_response_payload
from llm_pipeline.data import load_patterns
from llm_pipeline.providers import (
    InvalidProviderResponse, OpenAICompatibleClient, ProviderConfig,
    ProviderHTTPError, ProviderTransportError,
)
from llm_pipeline.requests import stage1_request_params, stage2_request_params
from llm_pipeline.stages import run_stage1_sync
from llm_pipeline.telemetry import observed_sync, read_calls, summarize_calls
from test_gemini_stages import make_catalog, make_prepared, stage1_json


class FakeHTTPResponse:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode()

    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self): return self.data


class FakeTransport:
    def __init__(self, monkeypatch, responses=()):
        self.responses = list(responses)
        self.requests = []
        monkeypatch.setattr("llm_pipeline.providers.request.urlopen", self.urlopen)

    def urlopen(self, req, timeout=None):
        payload = json.loads(req.data or b"{}")
        self.requests.append((req.full_url, payload, dict(req.headers), timeout))
        if req.full_url.endswith("/health"):
            return FakeHTTPResponse({"status": "ok"})
        if req.full_url.endswith("/v1/models"):
            return FakeHTTPResponse({"data": [{"id": "llama-test", "owned_by": "llamacpp", "meta": {"n_ctx_train": 8192, "n_params": 8_000_000_000}}]})
        if req.full_url.endswith("/v1/chat/completions/input_tokens"):
            return FakeHTTPResponse({"input_tokens": 11})
        status, body, failure = self.responses.pop(0)
        if failure == "timeout":
            raise socket.timeout("timed out")
        if failure == "connection":
            raise error.URLError("connection refused")
        if status >= 400:
            raise error.HTTPError(req.full_url, status, "error", {}, BytesIO(json.dumps(body).encode()))
        return FakeHTTPResponse(body)


@pytest.fixture
def local_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama-test")
    monkeypatch.setenv("LOCAL_LLM_BACKEND", "llama.cpp")
    monkeypatch.setenv("LOCAL_LLM_STRUCTURED_OUTPUT", "json_schema")
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_MODEL_PATH", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL_SHA256", raising=False)
    monkeypatch.delenv("LOCAL_LLM_HASH_MODEL", raising=False)


def completion(content, *, usage=True, finish="stop"):
    payload = {"id": "chat-1", "model": "llama-test", "choices": [{"finish_reason": finish, "message": {"content": content}}], "timings": {"prompt_per_second": 50.0, "predicted_per_second": 5.0}}
    if usage:
        payload["usage"] = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    return payload


def local_config(**changes):
    return replace(ProviderConfig.from_environment("local"), base_url="http://local.test", timeout_seconds=1, **changes)


def test_provider_selection_and_local_defaults(local_env, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:9999/v1/")
    local = ProviderConfig.from_environment()
    assert local.provider == "openai_compatible" and local.base_url == "http://127.0.0.1:9999"
    assert local.model == "llama-test" and local.api_key is None
    assert local.concurrency == 1 and not local.capabilities.supports_remote_batch
    assert ProviderConfig.from_environment("gemini").provider == "gemini"


def test_base_url_credentials_are_rejected(local_env, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://user:secret@127.0.0.1:8080")
    with pytest.raises(ValueError, match="não pode conter credenciais"):
        ProviderConfig.from_environment()


def test_local_model_is_hashed_once_for_resume_fingerprint(local_env, monkeypatch, tmp_path):
    model = tmp_path / "llama-8b-Q4_K_M.gguf"
    model.write_bytes(b"small fake gguf for provenance test")
    monkeypatch.setenv("LLAMA_MODEL_PATH", str(model))
    config = ProviderConfig.from_environment()
    assert config.model_sha256
    assert config.model_size_bytes == model.stat().st_size
    assert config.quantization == "Q4_K_M"
    assert config.public_metadata()["model_file"] == model.name


def test_stage1_request_uses_llama_cpp_json_schema(local_env, monkeypatch, tmp_path):
    transport = FakeTransport(monkeypatch, [(200, completion(stage1_json()), None)])
    catalog, prepared = make_catalog(tmp_path), make_prepared()
    params = stage1_request_params(prepared, catalog, "llama-test", 0, 256, 42, "minimal")
    response = OpenAICompatibleClient(local_config()).models.generate_content(**params)
    payload, metadata = parse_response_payload(response)
    sent = transport.requests[0][1]
    assert sent["response_format"]["type"] == "json_schema" and "schema" in sent["response_format"]
    assert sent["temperature"] == 0 and sent["seed"] == 42
    assert payload["candidates"][0]["pattern"] == "Oracle"
    assert metadata["usage"]["total_tokens"] == 30


def test_json_object_llama_cpp_payload_includes_schema_and_text_instruction(local_env, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_STRUCTURED_OUTPUT", "json_object")
    monkeypatch.setenv("LOCAL_LLM_SCHEMA_DIALECT", "llama_cpp")
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": False,
    }
    transport = FakeTransport(monkeypatch, [(200, completion('{"status":"ok"}'), None)])
    params = {
        "model": "llama-test",
        "contents": [{"role": "user", "parts": [{"text": "Return status ok."}]}],
        "config": {"system_instruction": "Follow the request.", "response_json_schema": schema},
    }

    OpenAICompatibleClient(local_config()).models.generate_content(**params)

    sent = transport.requests[0][1]
    assert sent["response_format"]["type"] == "json_object"
    assert sent["response_format"]["schema"] == schema
    assert json.dumps(schema, ensure_ascii=False, separators=(",", ":")) in sent["messages"][0]["content"]


def test_json_object_openai_dialect_payload_remains_unchanged(local_env, monkeypatch):
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    transport = FakeTransport(monkeypatch, [(200, completion('{"status":"ok"}'), None)])
    params = {"model": "llama-test", "contents": [], "config": {"response_json_schema": schema}}

    OpenAICompatibleClient(
        local_config(structured_output="json_object", schema_dialect="openai")
    ).models.generate_content(**params)

    assert transport.requests[0][1]["response_format"] == {"type": "json_object"}


def test_stage2_request_and_public_schema_preserved(local_env, monkeypatch, tmp_path):
    body = json.dumps({"verdict": "yes", "mechanism_match": True, "scope_match": True, "focus_match": True, "evidence_text": "Fetch price off-chain and submit it to the contract.", "evidence_location": ["body"], "justification": "Matches.", "adoption_status": "proposed_or_planned", "false_friend_detected": "no", "pattern_challenge_categories": ["none_explicit"], "confidence": "high", "alternative_pattern": "", "overlap_with": []})
    transport = FakeTransport(monkeypatch, [(200, completion(body), None)])
    catalog, prepared = make_catalog(tmp_path), make_prepared()
    params = stage2_request_params(prepared, "Oracle", catalog, "llama-test", 0, 256, 42, "low")
    payload, _ = parse_response_payload(OpenAICompatibleClient(local_config()).models.generate_content(**params))
    assert payload["verdict"] == "yes"
    schema = transport.requests[0][1]["response_format"]["schema"]
    assert set(schema["required"]) == set(params["config"]["response_json_schema"]["required"])


@pytest.mark.parametrize("content", ["not json", "", "before {\"status\":\"ok\"} after"])
def test_invalid_or_empty_response_is_rejected(local_env, monkeypatch, content):
    FakeTransport(monkeypatch, [(200, completion(content), None)])
    params = {"model": "llama-test", "contents": [{"role": "user", "parts": [{"text": "x"}]}], "config": {"response_json_schema": {"type": "object"}}}
    with pytest.raises(InvalidProviderResponse):
        OpenAICompatibleClient(local_config()).models.generate_content(**params)


def test_missing_required_field_is_rejected(local_env, monkeypatch):
    FakeTransport(monkeypatch, [(200, completion("{}"), None)])
    params = {"model": "llama-test", "contents": [], "config": {"response_json_schema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}
    with pytest.raises(InvalidProviderResponse, match="missing required"):
        OpenAICompatibleClient(local_config()).models.generate_content(**params)


def test_field_outside_schema_is_rejected_in_json_object_mode(local_env, monkeypatch):
    FakeTransport(monkeypatch, [(200, completion('{"status":"ok","unexpected":true}'), None)])
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    }
    params = {"model": "llama-test", "contents": [], "config": {"response_json_schema": schema}}

    with pytest.raises(InvalidProviderResponse, match="unknown fields: \\['unexpected'\\]"):
        OpenAICompatibleClient(
            local_config(structured_output="json_object", schema_dialect="llama_cpp")
        ).models.generate_content(**params)


def test_http_500_timeout_and_connection_refused(local_env, monkeypatch):
    transport = FakeTransport(monkeypatch, [(500, {"error": "boom"}, None), (200, {}, "timeout"), (200, {}, "connection")])
    client = OpenAICompatibleClient(local_config())
    with pytest.raises(ProviderHTTPError) as exc:
        client._json_request("POST", "/v1/chat/completions", {})
    assert exc.value.retryable
    with pytest.raises(ProviderTransportError): client._json_request("POST", "/v1/chat/completions", {})
    with pytest.raises(ProviderTransportError): client._json_request("POST", "/v1/chat/completions", {})
    assert len(transport.requests) == 3


def test_invalid_response_retry_and_telemetry(local_env, monkeypatch, tmp_path):
    FakeTransport(monkeypatch, [(200, completion("bad"), None), (200, completion(stage1_json()), None)])
    monkeypatch.setattr("llm_pipeline.client.time.sleep", lambda _: None)
    catalog, prepared = make_catalog(tmp_path), make_prepared()
    params = stage1_request_params(prepared, catalog, "llama-test", 0, 256, 42, "minimal")
    result = observed_sync(call_sync, OpenAICompatibleClient(local_config()), params, tmp_path, prepared, "stage1")
    assert parse_response_payload(result)[0]["candidates"]
    rows = read_calls(tmp_path)
    assert len(rows) == 2 and rows[1]["attempt_number"] == 2 and rows[0]["parsing_status"] == "invalid"
    summary = summarize_calls(tmp_path)
    assert summary["retries"] == 1 and (tmp_path / "local_llm_summary.json").exists()


def test_usage_absent_uses_server_tokenizer(local_env, monkeypatch):
    FakeTransport(monkeypatch, [(200, completion("{}", usage=False), None)])
    params = {"model": "llama-test", "contents": [], "config": {"response_json_schema": {"type": "object"}}}
    response = OpenAICompatibleClient(local_config()).models.generate_content(**params)
    assert response["usage_metadata"]["prompt_token_count"] == 11
    assert response["usage_metadata"]["total_token_count"] is None
    assert response["provider_metadata"]["input_token_source"] == "server_tokenized"


def test_truncated_response_is_retried_and_rejected(local_env, monkeypatch):
    FakeTransport(monkeypatch, [
        (200, completion("{}", finish="length"), None),
        (200, completion("{}", finish="length"), None),
    ])
    monkeypatch.setattr("llm_pipeline.client.time.sleep", lambda _: None)
    params = {"model": "llama-test", "contents": [], "config": {"response_json_schema": {"type": "object"}}}
    with pytest.raises(InvalidProviderResponse, match="incomplete structured response"):
        call_sync(OpenAICompatibleClient(local_config(max_attempts=2)), params)


def test_probe_model_metadata(local_env, monkeypatch):
    FakeTransport(monkeypatch)
    metadata = OpenAICompatibleClient(local_config()).probe()
    assert metadata["health"]["status"] == "ok"
    assert metadata["server_model"]["meta"]["n_ctx_train"] == 8192


def test_resume_rejects_provider_and_fingerprint_change(local_env, tmp_path):
    root = Path(__file__).resolve().parents[1]
    patterns = root / "blockchain_patterns_keywords_v3.csv"
    input_path = tmp_path / "input.csv"
    input_path.write_text("repository,issue_number,issue_title,issue_body\nr,1,t,b\n")
    run_dir = tmp_path / "run"; run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(json.dumps({"provider": "gemini", "provider_fingerprint": "old"}))
    args = build_parser().parse_args(["run", "--input", str(input_path), "--patterns", str(patterns), "--provider", "local"])
    args.provider_config = ProviderConfig.from_environment("local")
    with pytest.raises(ValueError, match="provider"):
        _validate_resume_metadata(run_dir, input_path, patterns, load_patterns(patterns), args)
    config = ProviderConfig.from_environment("local")
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "provider": config.provider, "backend": config.backend,
        "provider_fingerprint": config.fingerprint, "model_fingerprint": "different-model",
    }))
    with pytest.raises(ValueError, match="model_fingerprint"):
        _validate_resume_metadata(run_dir, input_path, patterns, load_patterns(patterns), args)


def test_local_batch_fails_explicitly(local_env, capsys):
    assert main(["run", "--input", "unused.csv", "--provider", "local", "--mode", "batch"]) == 1
    assert "não suporta remote batch" in capsys.readouterr().err


def test_local_stage_resume_skips_completed_result(local_env, monkeypatch, tmp_path):
    transport = FakeTransport(monkeypatch, [(200, completion(stage1_json()), None)])
    catalog, prepared = make_catalog(tmp_path), make_prepared()
    client = OpenAICompatibleClient(local_config())
    first = run_stage1_sync(client, [prepared], catalog, "llama-test", 0, 256, 42, "minimal", tmp_path)
    second = run_stage1_sync(client, [prepared], catalog, "llama-test", 0, 256, 42, "minimal", tmp_path)
    assert len(first) == len(second) == 1
    assert len([r for r in transport.requests if r[0].endswith("/v1/chat/completions")]) == 1


def test_local_optimized_cli_end_to_end_and_resume(local_env, monkeypatch, tmp_path):
    from test_token_optimization import compact_payload
    from test_gemini_stages import stage2_json

    root = Path(__file__).resolve().parents[1]
    catalog = load_patterns(root / "blockchain_patterns_keywords_v3.csv")
    transport = FakeTransport(monkeypatch, [
        (200, completion(compact_payload(stage1_json(), catalog, 1)), None),
        (200, completion(compact_payload(stage2_json(), catalog, 2)), None),
    ])
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://local.test")
    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "repository,issue_number,issue_title,issue_body,type,state,labels,concatenated_comments\n"
        "repo,1,External price feed,Fetch price off-chain and submit it to the contract.,Issue,open,,\n"
    )
    args = ["run", "--input", str(input_path), "--patterns", str(root / "blockchain_patterns_keywords_v3.csv"),
            "--provider", "local", "--profile", "optimized", "--mode", "sync",
            "--output-dir", str(tmp_path), "--run-id", "local-run"]
    assert main(args) == 0
    run = tmp_path / "local-run"
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["provider"] == "openai_compatible" and metadata["backend"] == "llama.cpp"
    assert (run / "provider_runtime.json").exists() and (run / "local_llm_summary.json").exists()
    assert "yes" in (run / "issue_results.csv").read_text()
    calls_before = len([r for r in transport.requests if r[0].endswith("/v1/chat/completions")])
    assert main(args) == 0
    assert len([r for r in transport.requests if r[0].endswith("/v1/chat/completions")]) == calls_before
