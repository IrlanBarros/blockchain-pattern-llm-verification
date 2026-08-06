"""Agregação determinística dos pares no nível da issue."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import clean_scalar

def aggregate_issue_results(stage1_df: pd.DataFrame, stage2_df: pd.DataFrame) -> pd.DataFrame:
    stage2_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not stage2_df.empty:
        for row in stage2_df.to_dict("records"):
            key = (clean_scalar(row["repository"]), clean_scalar(row["issue_number"]))
            stage2_groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for s1 in stage1_df.to_dict("records"):
        key = (clean_scalar(s1["repository"]), clean_scalar(s1["issue_number"]))
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

        output.append(
            {
                "repository": s1["repository"],
                "issue_number": s1["issue_number"],
                "issue_key": s1.get("issue_key", ""),
                "relevant_to_pattern_study": relevance,
                "patterns_present": "|".join(non_no_patterns),
                "patterns_confirmed": "|".join(yes_patterns),
                "patterns_uncertain": "|".join(uncertain_patterns),
                "patterns_insufficient_context": "|".join(insufficient_patterns),
                "adoption_status": adoption_status,
                "issue_activity_type": s1.get("issue_activity_type", ""),
                "issue_challenge_categories": s1.get("issue_challenge_categories", ""),
                "false_friend_detected": false_friend,
                "insufficient_context": "yes" if relevance == "insufficient_context" else "no",
                "issue_summary": s1.get("issue_summary", ""),
                "stage1_candidate_count": s1.get("candidate_count", 0),
                "stage2_pair_count": len(pairs),
                "pipeline_error_count": (1 if s1.get("request_status") != "succeeded" else 0)
                + len(failed_pairs),
                "input_truncated": s1.get("input_truncated", "no"),
            }
        )
    return pd.DataFrame(output)
