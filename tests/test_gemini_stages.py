from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llm_pipeline.data import load_patterns, prepare_issue
from llm_pipeline.stages import run_stage1_batch, run_stage1_sync, run_stage2_sync, stage2_pairs_from_stage1


class EnumValue:
    def __init__(self, value: str):
        self.value = value


class FakeUsage:
    def model_dump(self, mode: str = "json"):
        return {
            "prompt_token_count": 100,
            "candidates_token_count": 20,
            "total_token_count": 125,
            "thoughts_token_count": 5,
        }


def response_for(text: str, model: str):
    return SimpleNamespace(
        text=text,
        parsed=None,
        response_id="resp-1",
        model_version=model,
        prompt_feedback=None,
        candidates=[
            SimpleNamespace(
                finish_reason=EnumValue("STOP"),
                finish_message=None,
                content=None,
            )
        ],
        usage_metadata=FakeUsage(),
    )


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **params):
        self.calls.append(params)
        return self.responses.pop(0)


class FakeSyncClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class FakeBatches:
    def __init__(self, inline_response):
        self.inline_response = inline_response
        self.created_src = None

    def create(self, *, model, src, config):
        self.created_src = src
        return SimpleNamespace(name="batches/test", model=model, state=EnumValue("JOB_STATE_QUEUED"))

    def get(self, *, name):
        return SimpleNamespace(
            name=name,
            state=EnumValue("JOB_STATE_SUCCEEDED"),
            completion_stats=SimpleNamespace(
                model_dump=lambda mode="json": {"successful_count": 1, "failed_count": 0}
            ),
            error=None,
            dest=SimpleNamespace(inlined_responses=[self.inline_response]),
        )


class FakeBatchClient:
    def __init__(self, inline_response):
        self.batches = FakeBatches(inline_response)


def make_catalog(tmp_path: Path):
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


def make_prepared():
    return prepare_issue(
        pd.Series(
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
        ),
        12_000,
    )


def stage1_json():
    return """{
      "issue_summary": "Integração de preço externo.",
      "issue_activity_type": "feature_implementation",
      "issue_challenge_categories": ["implementation_complexity"],
      "context_status": "sufficient",
      "candidates": [{
        "pattern": "Oracle",
        "evidence_text": "Fetch price off-chain and submit it to the contract.",
        "evidence_location": "body",
        "rationale": "Descreve dado externo enviado ao contrato.",
        "confidence": "high"
      }]
    }"""


def stage2_json():
    return """{
      "verdict": "yes",
      "evidence_text": "Fetch price off-chain and submit it to the contract.",
      "evidence_location": ["body"],
      "justification": "O fluxo externo para on-chain corresponde ao Oracle.",
      "adoption_status": "proposed_or_planned",
      "false_friend_detected": "no",
      "pattern_challenge_categories": ["implementation_complexity"],
      "confidence": "high",
      "alternative_pattern": "",
      "overlap_with": []
    }"""


def test_sync_stages_end_to_end_with_fake_gemini(tmp_path: Path):
    catalog = make_catalog(tmp_path)
    prepared = make_prepared()
    client = FakeSyncClient(
        [
            response_for(stage1_json(), "gemini-3.1-flash-lite"),
            response_for(stage2_json(), "gemini-3.5-flash"),
        ]
    )

    s1 = run_stage1_sync(
        client,
        [prepared],
        catalog,
        "gemini-3.1-flash-lite",
        1.0,
        2048,
        0,
        "minimal",
        tmp_path,
    )
    assert s1.loc[0, "request_status"] == "succeeded"
    assert s1.loc[0, "candidate_count"] == 1
    assert s1.loc[0, "thoughts_tokens"] == 5

    pairs = stage2_pairs_from_stage1(s1)
    s2 = run_stage2_sync(
        client,
        {(prepared.repository, prepared.issue_number): prepared},
        pairs,
        catalog,
        "gemini-3.5-flash",
        1.0,
        2048,
        0,
        "low",
        tmp_path,
    )
    assert s2.loc[0, "request_status"] == "succeeded"
    assert s2.loc[0, "verdict"] == "yes"


def test_stage1_inline_batch_uses_metadata_and_parses_result(tmp_path: Path):
    catalog = make_catalog(tmp_path)
    prepared = make_prepared()
    inline = SimpleNamespace(
        metadata={"custom_id": prepared.custom_id_stage1},
        error=None,
        response=response_for(stage1_json(), "gemini-3.1-flash-lite"),
        model_dump=lambda mode="json": {"metadata": {"custom_id": prepared.custom_id_stage1}},
    )
    client = FakeBatchClient(inline)
    result = run_stage1_batch(
        client,
        [prepared],
        catalog,
        "gemini-3.1-flash-lite",
        1.0,
        2048,
        0,
        "minimal",
        250,
        18_000_000,
        0,
        tmp_path,
    )
    assert result.loc[0, "request_status"] == "succeeded"
    assert result.loc[0, "candidate_count"] == 1
    assert client.batches.created_src[0]["metadata"]["custom_id"] == prepared.custom_id_stage1
