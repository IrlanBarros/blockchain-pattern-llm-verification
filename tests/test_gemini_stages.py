from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llm_pipeline.data import load_patterns, prepare_issue
from llm_pipeline.checkpoint import CheckpointIntegrityError, ResultStore
from llm_pipeline.stages import (
    pipeline_integrity_report,
    run_stage1_batch,
    run_stage1_sync,
    run_stage2_sync,
    stage2_pairs_from_stage1,
)
from llm_pipeline.utils import stable_custom_id


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


def test_stage2_resume_regression_72_existing_51_only_calls_21(tmp_path: Path):
    """Regression for the reported 72 -> 51 interruption -> resume case."""
    catalog = make_catalog(tmp_path)
    prepared = []
    for number in range(1, 73):
        item = make_prepared()
        item = item.__class__(
            repository="repo",
            issue_number=str(number),
            issue_key=f"repo#{number}",
            custom_id_stage1=stable_custom_id("s1", "repo", str(number)),
            artifact_text=item.artifact_text,
            original_char_count=item.original_char_count,
            included_char_count=item.included_char_count,
            input_truncated=item.input_truncated,
            artifact_type=item.artifact_type,
        )
        prepared.append(item)
    stage1 = pd.DataFrame(
        [
            {"repository": item.repository, "issue_number": item.issue_number, "request_status": "succeeded", "candidates": '["Oracle"]'}
            for item in prepared
        ]
    )
    pairs = stage2_pairs_from_stage1(stage1)
    assert len(pairs) == 72

    existing = pd.DataFrame(
        [
            {
                "repository": "repo", "issue_number": str(number), "pattern": "Oracle",
                "custom_id": stable_custom_id("s2", "repo", str(number), "Oracle"),
                "request_status": "succeeded", "verdict": "no",
            }
            for number in range(1, 52)
        ]
    )
    existing.to_csv(tmp_path / "stage2_results.csv", index=False)
    client = FakeSyncClient([response_for(stage2_json(), "gemini-3.5-flash") for _ in range(21)])
    result = run_stage2_sync(
        client, {(item.repository, item.issue_number): item for item in prepared}, pairs, catalog,
        "gemini-3.5-flash", 1.0, 2048, 0, "low", tmp_path,
    )
    assert len(client.models.calls) == 21
    assert len(result) == 72
    assert not result.duplicated(["repository", "issue_number", "pattern"]).any()
    integrity = pipeline_integrity_report(prepared, stage1, result)
    assert integrity["expected_stage2_pairs"] == 72
    assert integrity["completed_stage2_pairs"] == 72
    assert integrity["status"] == "OK"


def test_completed_run_is_idempotent_and_uses_no_new_calls(tmp_path: Path):
    catalog = make_catalog(tmp_path)
    prepared = make_prepared()
    first_client = FakeSyncClient(
        [response_for(stage1_json(), "gemini-3.1-flash-lite"), response_for(stage2_json(), "gemini-3.5-flash")]
    )
    stage1 = run_stage1_sync(first_client, [prepared], catalog, "gemini-3.1-flash-lite", 1, 2048, 0, "minimal", tmp_path)
    pairs = stage2_pairs_from_stage1(stage1)
    stage2 = run_stage2_sync(first_client, {(prepared.repository, prepared.issue_number): prepared}, pairs, catalog, "gemini-3.5-flash", 1, 2048, 0, "low", tmp_path)
    resumed_client = FakeSyncClient([])
    resumed_stage1 = run_stage1_sync(resumed_client, [prepared], catalog, "gemini-3.1-flash-lite", 1, 2048, 0, "minimal", tmp_path)
    resumed_stage2 = run_stage2_sync(resumed_client, {(prepared.repository, prepared.issue_number): prepared}, pairs, catalog, "gemini-3.5-flash", 1, 2048, 0, "low", tmp_path)
    assert len(resumed_client.models.calls) == 0
    assert len(resumed_stage1) == len(stage1) == 1
    assert len(resumed_stage2) == len(stage2) == 1


def test_duplicate_persisted_pair_is_rejected_instead_of_silently_chosen(tmp_path: Path):
    pd.DataFrame(
        [
            {"repository": "one", "issue_number": "7", "pattern": "Oracle", "request_status": "succeeded", "verdict": "no"},
            {"repository": "one", "issue_number": "7", "pattern": "Oracle", "request_status": "succeeded", "verdict": "yes"},
        ]
    ).to_csv(tmp_path / "stage2_results.csv", index=False)
    try:
        ResultStore(tmp_path / "stage2_results.csv", stage="stage2")
    except CheckpointIntegrityError:
        pass
    else:
        raise AssertionError("artefato duplicado não foi rejeitado")


def test_legacy_stage1_csv_is_reused_without_api_call(tmp_path: Path):
    """Old results without request_status remain usable when structurally valid."""
    catalog = make_catalog(tmp_path)
    prepared = make_prepared()
    pd.DataFrame(
        [{"repository": "repo", "issue_number": "1", "candidates": '["Oracle"]', "candidate_count": 1}]
    ).to_csv(tmp_path / "stage1_results.csv", index=False)
    result = run_stage1_sync(
        FakeSyncClient([]), [prepared], catalog, "gemini-3.1-flash-lite", 1, 2048, 0, "minimal", tmp_path
    )
    assert len(result) == 1
    assert result.loc[0, "request_status"] == "succeeded"


def test_invalid_checkpoint_is_rejected_instead_of_reset(tmp_path: Path):
    (tmp_path / "stage1_checkpoint.json").write_text("not json", encoding="utf-8")
    try:
        run_stage1_sync(
            FakeSyncClient([]), [], make_catalog(tmp_path), "gemini-3.1-flash-lite", 1, 2048, 0, "minimal", tmp_path
        )
    except CheckpointIntegrityError:
        pass
    else:
        raise AssertionError("checkpoint inválido não foi rejeitado")


def test_keyboard_interrupt_preserves_prior_stage1_result(tmp_path: Path):
    catalog = make_catalog(tmp_path)
    first, second = make_prepared(), make_prepared()
    second = second.__class__(
        repository="other", issue_number="1", issue_key="other#1", custom_id_stage1=stable_custom_id("s1", "other", "1"),
        artifact_text=second.artifact_text, original_char_count=second.original_char_count,
        included_char_count=second.included_char_count, input_truncated=second.input_truncated, artifact_type=second.artifact_type,
    )

    class InterruptingModels:
        def __init__(self): self.calls = 0
        def generate_content(self, **params):
            self.calls += 1
            if self.calls == 1:
                return response_for(stage1_json(), "gemini-3.1-flash-lite")
            raise KeyboardInterrupt()

    interrupted = SimpleNamespace(models=InterruptingModels())
    try:
        run_stage1_sync(interrupted, [first, second], catalog, "gemini-3.1-flash-lite", 1, 2048, 0, "minimal", tmp_path)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("interrupção deveria propagar")
    assert len(ResultStore(tmp_path / "stage1_results.csv", stage="stage1").completed_ids()) == 1
    resumed = FakeSyncClient([response_for(stage1_json(), "gemini-3.1-flash-lite")])
    result = run_stage1_sync(resumed, [first, second], catalog, "gemini-3.1-flash-lite", 1, 2048, 0, "minimal", tmp_path)
    assert len(resumed.models.calls) == 1
    assert len(result) == 2
