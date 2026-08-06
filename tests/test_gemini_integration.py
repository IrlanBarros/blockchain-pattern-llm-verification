from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from llm_pipeline.client import chunk_inline_requests, parse_response_payload
from llm_pipeline.data import load_patterns, prepare_issue
from llm_pipeline.requests import stage1_request_params, to_inline_batch_request


def _catalog(tmp_path: Path):
    path = tmp_path / "patterns.csv"
    pd.DataFrame(
        [
            {
                "pattern": "Oracle",
                "category": "Data exchange",
                "subcategory": "Data exchange",
                "description": "Componente off-chain injeta dados externos na blockchain.",
            }
        ]
    ).to_csv(path, index=False)
    return load_patterns(path)


def _prepared():
    row = pd.Series(
        {
            "repository": "repo",
            "issue_number": "1",
            "issue_title": "External price feed",
            "issue_body": "Fetch price off-chain and submit it to the contract.",
            "concatenated_comments": "",
            "type": "Issue",
            "labels": "feature",
            "state": "open",
        }
    )
    return prepare_issue(row, 12_000)


def test_stage1_request_uses_gemini_generate_content_shape(tmp_path: Path):
    params = stage1_request_params(
        _prepared(),
        _catalog(tmp_path),
        "gemini-3.1-flash-lite",
        1.0,
        2048,
        0,
        "minimal",
    )
    assert params["model"] == "gemini-3.1-flash-lite"
    assert params["contents"][0]["role"] == "user"
    config = params["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"]["type"] == "object"
    assert config["thinking_config"] == {"thinking_level": "minimal"}
    assert config["seed"] == 0

    # Valida a forma contra os tipos do SDK oficial sem fazer chamada de rede.
    from google.genai import types

    types.GenerateContentConfig(**config)


def test_inline_batch_request_preserves_custom_id(tmp_path: Path):
    params = stage1_request_params(
        _prepared(),
        _catalog(tmp_path),
        "gemini-3.1-flash-lite",
        1.0,
        2048,
        0,
        "minimal",
    )
    request = to_inline_batch_request("s1_abc", params)
    assert "model" not in request
    assert request["metadata"] == {"custom_id": "s1_abc"}

    from google.genai import types

    types.InlinedRequest(**request)


def test_parse_gemini_response_payload_and_usage():
    response = SimpleNamespace(
        text='{"candidates": []}',
        parsed=None,
        response_id="response-1",
        model_version="gemini-3.1-flash-lite",
        prompt_feedback=None,
        candidates=[
            SimpleNamespace(
                finish_reason=SimpleNamespace(value="STOP"),
                finish_message=None,
                content=None,
            )
        ],
        usage_metadata=SimpleNamespace(
            model_dump=lambda mode="json": {
                "prompt_token_count": 100,
                "candidates_token_count": 20,
                "total_token_count": 125,
                "thoughts_token_count": 5,
                "cached_content_token_count": 0,
            }
        ),
    )
    payload, metadata = parse_response_payload(response)
    assert payload == {"candidates": []}
    assert metadata["message_id"] == "response-1"
    assert metadata["usage"]["input_tokens"] == 100
    assert metadata["usage"]["thoughts_tokens"] == 5


def test_parse_gemini_response_rejects_max_tokens():
    response = SimpleNamespace(
        text='{"candidates": []}',
        parsed=None,
        response_id="response-1",
        model_version="gemini-3.1-flash-lite",
        prompt_feedback=None,
        candidates=[
            SimpleNamespace(
                finish_reason=SimpleNamespace(value="MAX_TOKENS"),
                finish_message="output truncated",
                content=None,
            )
        ],
        usage_metadata=None,
    )
    with pytest.raises(ValueError, match="MAX_TOKENS"):
        parse_response_payload(response)


def test_batch_chunking_respects_count_and_bytes():
    requests = [
        {"metadata": {"custom_id": f"r{i}"}, "contents": "x" * 20, "config": {}}
        for i in range(5)
    ]
    chunks = list(chunk_inline_requests(requests, max_items=2, max_bytes=10_000))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]

    with pytest.raises(ValueError, match="excede sozinha"):
        list(chunk_inline_requests(requests[:1], max_items=2, max_bytes=10))
