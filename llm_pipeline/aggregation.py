"""Agregação determinística dos pares no nível da issue."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import clean_scalar

FINAL_ISSUE_COLUMNS = [
    "issue_id",
    "repository_full_name",
    "issue_number",
    "created_at",
    "closed_at",
    "labels",
    "repository_category",
    "repository_category_canonical",
    "data_source",
    "issue_title",
    "issue_body",
    "concatenated_comments",
    "comments_count",
    "comments_count_filtered",
    "commits_count",
    "raw_text_sha256",
    "sample_group",
    "selection_trigger",
    "issue_summary",
    "stage1_candidate_count",
    "stage2_pair_count",
    "pipeline_error_count",
    "input_truncated",
    "patterns_present",
    "evidence_text",
    "evidence_location",
    "confidence",
    "insufficient_context",
    "false_friend_detected",
    "overlapping_patterns",
    "issue_activity_type",
    "challenge_categories",
    "adoption_status",
    "relevant_to_pattern_study",
    "annotator_id",
    "human_notes",
    "annotation_date",
    "type",
    "state",
]


def _join_unique(values: list[str]) -> str:
    out: list[str] = []
    for value in values:
        cleaned = clean_scalar(value)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return "|".join(out)


def _aggregate_confidence(values: list[str]) -> str:
    normalized = [clean_scalar(value).lower() for value in values if clean_scalar(value)]
    if not normalized:
        return ""
    if "low" in normalized:
        return "low"
    if "medium" in normalized:
        return "medium"
    if "high" in normalized:
        return "high"
    return normalized[0]


def aggregate_issue_results(
    stage1_df: pd.DataFrame,
    stage2_df: pd.DataFrame,
    input_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    stage2_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not stage2_df.empty:
        for row in stage2_df.to_dict("records"):
            key = (clean_scalar(row["repository"]), clean_scalar(row["issue_number"]))
            stage2_groups.setdefault(key, []).append(row)

    input_groups: dict[tuple[str, str], dict[str, Any]] = {}
    if input_df is not None and not input_df.empty:
        for row in input_df.to_dict("records"):
            key = (clean_scalar(row.get("repository", "")), clean_scalar(row.get("issue_number", "")))
            input_groups[key] = row

    output: list[dict[str, Any]] = []
    for s1 in stage1_df.to_dict("records"):
        key = (clean_scalar(s1["repository"]), clean_scalar(s1["issue_number"]))
        source_row = input_groups.get(key, {})
        pairs = [row for row in stage2_groups.get(key, []) if row.get("request_status") == "succeeded"]
        failed_pairs = [row for row in stage2_groups.get(key, []) if row.get("request_status") != "succeeded"]

        yes_patterns = [row["pattern"] for row in pairs if row.get("verdict") == "yes"]
        uncertain_patterns = [row["pattern"] for row in pairs if row.get("verdict") == "uncertain"]
        insufficient_patterns = [
            row["pattern"] for row in pairs if row.get("verdict") == "insufficient_context"
        ]
        non_no_patterns = yes_patterns + uncertain_patterns + insufficient_patterns

        if s1.get("request_status") != "succeeded" or failed_pairs:
            relevance = "pipeline_error"
        elif yes_patterns:
            relevance = "yes"
        elif uncertain_patterns:
            relevance = "uncertain"
        elif insufficient_patterns or s1.get("context_status") == "insufficient_context":
            relevance = "insufficient_context"
        else:
            relevance = "no"

        statuses = []
        for row in pairs:
            if row.get("verdict") != "no" and row.get("adoption_status"):
                if row["adoption_status"] not in statuses:
                    statuses.append(row["adoption_status"])
        if len(statuses) == 1:
            adoption_status = statuses[0]
        elif len(statuses) > 1:
            adoption_status = "multiple"
        elif relevance == "insufficient_context":
            adoption_status = "insufficient_context"
        else:
            adoption_status = "not_related"

        ff_values = [row.get("false_friend_detected") for row in pairs]
        if "yes" in ff_values:
            false_friend = "yes"
        elif "uncertain" in ff_values:
            false_friend = "uncertain"
        else:
            false_friend = "no"

        relevant_pairs = [row for row in pairs if row.get("verdict") != "no"]
        evidence_text = _join_unique([clean_scalar(row.get("evidence_text", "")) for row in relevant_pairs])
        evidence_location = _join_unique(
            [clean_scalar(row.get("evidence_location", "")) for row in relevant_pairs]
        )
        overlapping_patterns = _join_unique(
            [clean_scalar(row.get("overlap_with", "")) for row in relevant_pairs]
        )
        confidence = _aggregate_confidence([clean_scalar(row.get("confidence", "")) for row in relevant_pairs])

        repository_full_name = clean_scalar(
            source_row.get("repository_full_name", source_row.get("repository", s1["repository"]))
        )
        repository_category = clean_scalar(source_row.get("repository_category", ""))
        repository_category_canonical = clean_scalar(
            source_row.get("repository_category_canonical", repository_category)
        )
        challenge_categories = clean_scalar(s1.get("issue_challenge_categories", ""))

        output.append(
            {
                "issue_id": clean_scalar(source_row.get("issue_id", "")),
                "repository_full_name": repository_full_name,
                "issue_number": s1["issue_number"],
                "created_at": clean_scalar(source_row.get("created_at", "")),
                "closed_at": clean_scalar(source_row.get("closed_at", "")),
                "labels": clean_scalar(source_row.get("labels", "")),
                "repository_category": repository_category,
                "repository_category_canonical": repository_category_canonical,
                "data_source": clean_scalar(source_row.get("data_source", "")),
                "issue_title": clean_scalar(source_row.get("issue_title", "")),
                "issue_body": clean_scalar(source_row.get("issue_body", "")),
                "concatenated_comments": clean_scalar(source_row.get("concatenated_comments", "")),
                "comments_count": clean_scalar(source_row.get("comments_count", "")),
                "comments_count_filtered": clean_scalar(source_row.get("comments_count_filtered", "")),
                "commits_count": clean_scalar(source_row.get("commits_count", "")),
                "raw_text_sha256": clean_scalar(source_row.get("raw_text_sha256", "")),
                "sample_group": clean_scalar(source_row.get("sample_group", "")),
                "selection_trigger": clean_scalar(source_row.get("selection_trigger", "")),
                "issue_summary": s1.get("issue_summary", ""),
                "stage1_candidate_count": s1.get("candidate_count", 0),
                "stage2_pair_count": len(pairs),
                "pipeline_error_count": (1 if s1.get("request_status") != "succeeded" else 0)
                + len(failed_pairs),
                "input_truncated": s1.get("input_truncated", "no"),
                "patterns_present": "|".join(non_no_patterns),
                "evidence_text": evidence_text,
                "evidence_location": evidence_location,
                "confidence": confidence,
                "insufficient_context": "yes" if relevance == "insufficient_context" else "no",
                "false_friend_detected": false_friend,
                "overlapping_patterns": overlapping_patterns,
                "issue_activity_type": s1.get("issue_activity_type", ""),
                "challenge_categories": challenge_categories,
                "adoption_status": adoption_status,
                "relevant_to_pattern_study": relevance,
                # Campos reservados para anotação humana posterior.
                "annotator_id": "",
                "human_notes": "",
                "annotation_date": "",
                "type": clean_scalar(source_row.get("type", "")),
                "state": clean_scalar(source_row.get("state", "")),
                "repository": s1["repository"],
                "issue_key": s1.get("issue_key", ""),
                "patterns_confirmed": "|".join(yes_patterns),
                "patterns_uncertain": "|".join(uncertain_patterns),
                "patterns_insufficient_context": "|".join(insufficient_patterns),
                "issue_challenge_categories": s1.get("issue_challenge_categories", ""),
            }
        )

    out = pd.DataFrame(output)
    for column in FINAL_ISSUE_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    ordered = FINAL_ISSUE_COLUMNS + [col for col in out.columns if col not in FINAL_ISSUE_COLUMNS]
    return out[ordered]
