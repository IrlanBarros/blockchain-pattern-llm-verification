"""Issue-level evaluation helpers shared by reports and tests.

These functions are deliberately read-only: they normalize persisted artifacts,
validate keys, and measure final issue decisions.  They do not run either LLM
stage and do not reinterpret Stage 1 candidates as final positives.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from evaluate_pipeline import binary_metrics
from .human_annotations import human_component
from .normalization import clean_token

ISSUE_KEY = ["repository", "issue_number"]
HUMAN_LABELS = {"yes", "no", "uncertain", "insufficient_context"}


def _normalized_keys(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    out = frame.copy()
    if "repository" not in out and "repository_full_name" in out:
        out["repository"] = out["repository_full_name"]
    missing = set(ISSUE_KEY) - set(out.columns)
    if missing:
        raise ValueError(f"{label} sem colunas de chave: {sorted(missing)}")
    out["repository"] = out["repository"].map(clean_token)
    out["issue_number"] = (
        out["issue_number"].map(clean_token).str.replace(r"\.0+$", "", regex=True)
    )
    blank = (out["repository"] == "") | (out["issue_number"] == "")
    duplicated = out.duplicated(ISSUE_KEY, keep=False)
    if blank.any():
        raise ValueError(f"{label} contém chave de issue vazia")
    if duplicated.any():
        examples = out.loc[duplicated, ISSUE_KEY].drop_duplicates().to_dict("records")
        raise ValueError(f"{label} contém chaves de issue duplicadas: {examples[:10]}")
    return out


def human_issue_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract the annotator decision while preserving provenance-prefixed cells."""
    out = _normalized_keys(frame, label="human_issues")
    if "relevant_to_pattern_study" not in out:
        raise ValueError("human_issues sem relevant_to_pattern_study")
    out["human_label"] = out["relevant_to_pattern_study"].map(human_component)
    unknown = sorted(set(out["human_label"]) - HUMAN_LABELS)
    if unknown:
        raise ValueError(f"human_issues contém labels ausentes/desconhecidos: {unknown}")
    return out


def final_issue_predictions(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Read the persisted post-aggregation decision (`yes` is the only positive)."""
    out = _normalized_keys(frame, label=label)
    if "relevant_to_pattern_study" not in out:
        raise ValueError(f"{label} sem relevant_to_pattern_study")
    out["predicted_relevant"] = out["relevant_to_pattern_study"].eq("yes")
    return out


def issue_level_binary(
    human: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Measure final issue decisions; uncertain/context cases stay visible but excluded."""
    human = human_issue_reference(human)
    predictions = final_issue_predictions(predictions, label="issue_results")
    merged = human.merge(
        predictions[ISSUE_KEY + ["predicted_relevant"]],
        on=ISSUE_KEY,
        how="left",
        validate="one_to_one",
    )
    if merged["predicted_relevant"].isna().any():
        missing = merged.loc[merged["predicted_relevant"].isna(), ISSUE_KEY].to_dict("records")
        raise ValueError(f"issue_results não cobre todas as issues humanas: {missing[:10]}")
    merged["predicted_relevant"] = merged["predicted_relevant"].astype(bool)
    merged["predicted_binary"] = merged["predicted_relevant"].map({True: "yes", False: "no"})

    def classification(row: pd.Series) -> str:
        if row.human_label == "uncertain":
            return "EXCLUDED_UNCERTAIN"
        if row.human_label == "insufficient_context":
            return "EXCLUDED_INSUFFICIENT_CONTEXT"
        if row.human_label == "yes":
            return "TP" if row.predicted_relevant else "FN"
        return "FP" if row.predicted_relevant else "TN"

    merged["classification"] = merged.apply(classification, axis=1)
    binary = merged[merged["human_label"].isin(["yes", "no"])]
    metrics = binary_metrics(
        binary["human_label"].tolist(), binary["predicted_binary"].tolist()
    )
    counts = human["human_label"].value_counts().to_dict()
    metrics.update(
        {
            "total_issues": int(len(human)),
            "human_positive": int(counts.get("yes", 0)),
            "human_negative": int(counts.get("no", 0)),
            "excluded_uncertain": int(counts.get("uncertain", 0)),
            "excluded_insufficient_context": int(counts.get("insufficient_context", 0)),
            "predicted_positive": int(binary["predicted_relevant"].sum()),
            "predicted_negative": int((~binary["predicted_relevant"]).sum()),
            "predicted_positive_all_labels": int(merged["predicted_relevant"].sum()),
            "decision_rule": (
                "predicted_relevant=true iff persisted post-aggregation "
                "relevant_to_pattern_study equals yes"
            ),
        }
    )
    return metrics, merged
