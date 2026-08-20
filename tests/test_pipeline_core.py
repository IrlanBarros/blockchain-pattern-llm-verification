from pathlib import Path

import pandas as pd
import pytest

from llm_pipeline.aggregation import aggregate_issue_results
from llm_pipeline.data import load_patterns, prepare_issue, validate_input_dataframe
from llm_pipeline.schemas import normalize_stage2, stage1_schema


def minimal_catalog(tmp_path: Path) -> Path:
    path = tmp_path / "patterns.csv"
    pd.DataFrame(
        [
            {
                "pattern": "Oracle",
                "category": "On/off-chain interaction",
                "subcategory": "Data exchange",
                "description": "Dados externos entram on-chain por componente off-chain.",
            },
            {
                "pattern": "Proxy contract",
                "category": "On-chain - Smart contract",
                "subcategory": "Contract management",
                "description": "Proxy encaminha chamadas para implementação substituível.",
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_validate_requires_composite_key_uniqueness():
    df = pd.DataFrame(
        [
            {"repository": "a", "issue_number": 1, "issue_title": "x", "issue_body": "y"},
            {"repository": "a", "issue_number": 1, "issue_title": "z", "issue_body": "w"},
        ]
    )
    with pytest.raises(ValueError, match="chave composta"):
        validate_input_dataframe(df)


def test_same_issue_number_different_repo_is_allowed():
    df = pd.DataFrame(
        [
            {"repository": "a", "issue_number": 1, "issue_title": "x", "issue_body": "y"},
            {"repository": "b", "issue_number": 1, "issue_title": "z", "issue_body": "w"},
        ]
    )
    out = validate_input_dataframe(df)
    assert out["issue_key"].tolist() == ["a#1", "b#1"]


def test_prepare_issue_marks_truncation_and_keeps_tail():
    row = pd.Series(
        {
            "repository": "r",
            "issue_number": "7",
            "issue_title": "title",
            "issue_body": "A" * 1000 + "BODY_TAIL",
            "concatenated_comments": "B" * 1000 + "COMMENT_TAIL",
            "type": "Issue",
            "labels": "bug",
            "state": "open",
        }
    )
    prepared = prepare_issue(row, 700)
    assert prepared.input_truncated is True
    assert "BODY_TAIL" in prepared.artifact_text
    assert "COMMENT_TAIL" in prepared.artifact_text
    assert "caracteres omitidos" in prepared.artifact_text


def test_schema_restricts_pattern_names(tmp_path: Path):
    catalog = load_patterns(minimal_catalog(tmp_path))
    schema = stage1_schema(catalog.names)
    enum = schema["properties"]["candidates"]["items"]["properties"]["pattern"]["enum"]
    assert enum == ["Oracle", "Proxy contract"]


def test_stage2_normalization_rejects_inconsistent_yes(tmp_path: Path):
    catalog = load_patterns(minimal_catalog(tmp_path))
    payload = {
        "verdict": "yes",
        "evidence_text": "oracle",
        "evidence_location": ["body"],
        "justification": "x",
        "adoption_status": "not_related",
        "false_friend_detected": "no",
        "pattern_challenge_categories": ["none_explicit"],
        "confidence": "high",
        "alternative_pattern": "",
        "overlap_with": [],
    }
    with pytest.raises(ValueError, match="Inconsistência"):
        normalize_stage2(payload, catalog)


def test_aggregate_issue_results_priority():
    s1 = pd.DataFrame(
        [
            {
                "repository": "r",
                "issue_number": "1",
                "issue_key": "r#1",
                "request_status": "succeeded",
                "context_status": "sufficient",
                "candidate_count": 2,
                "issue_activity_type": "bug_report",
                "issue_challenge_categories": "reliability_or_availability",
                "issue_summary": "summary",
                "input_truncated": "no",
            }
        ]
    )
    s2 = pd.DataFrame(
        [
            {
                "repository": "r",
                "issue_number": "1",
                "pattern": "Oracle",
                "request_status": "succeeded",
                "verdict": "uncertain",
                "adoption_status": "conceptual_discussion",
                "false_friend_detected": "no",
            },
            {
                "repository": "r",
                "issue_number": "1",
                "pattern": "Proxy contract",
                "request_status": "succeeded",
                "verdict": "yes",
                "adoption_status": "problem_with_implementation",
                "false_friend_detected": "no",
            },
        ]
    )
    out = aggregate_issue_results(s1, s2).iloc[0]
    assert out["relevant_to_pattern_study"] == "yes"
    assert out["patterns_confirmed"] == "Proxy contract"
    assert out["patterns_uncertain"] == "Oracle"
    assert out["adoption_status"] == "multiple"
    assert out["stage1_candidate_count"] == 2
    assert out["stage2_pair_count"] == 2
    assert out["pipeline_error_count"] == 0
    assert out["issue_summary"] == "summary"
    assert out["challenge_categories"] == "reliability_or_availability"
    assert out["annotator_id"] == ""
    assert out["human_notes"] == ""
    assert out["annotation_date"] == ""
