#!/usr/bin/env python3
"""Avalia Stage 1 e Stage 2 contra anotações humanas/adjudicadas.

Antes das métricas, todos os CSVs passam pela mesma normalização estrutural
conservadora do pipeline. Arquivos originais não são modificados; mudanças de
Unicode, espaços, cabeçalhos, enums e nomes canônicos são registradas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from llm_pipeline.config import VERDICTS
from llm_pipeline.data import load_patterns_with_report
from llm_pipeline.normalization import (
    CanonicalLookup,
    NormalizationReport,
    canonicalize_multivalue,
    clean_token,
    merge_reports,
    normalize_dataframe_cells,
    read_csv_robust,
    resolve_repository_columns,
)
from llm_pipeline.utils import write_json

KEY = ["repository", "issue_number"]
PAIR_KEY = ["repository", "issue_number", "pattern"]
RELEVANCE_VALUES = ["yes", "no", "uncertain", "insufficient_context"]


def split_patterns(value: Any) -> list[str]:
    """Compatibilidade pública: separa JSON-list ou valores com |."""
    from llm_pipeline.normalization import split_multivalue

    return split_multivalue(value)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label} sem colunas obrigatórias: {sorted(missing)}")


def read_normalized(
    path: Path,
    *,
    source: str,
    token_columns: list[str],
) -> tuple[pd.DataFrame, NormalizationReport]:
    df, report = read_csv_robust(path, source=source)
    df = resolve_repository_columns(df, report, require=True)
    df = normalize_dataframe_cells(
        df,
        token_columns=[column for column in token_columns if column in df.columns],
        report=report,
    )
    return df, report


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = resolve_repository_columns(df, require=True)
    for col in KEY:
        if col in out.columns:
            out[col] = out[col].map(clean_token)
    return out


def normalize_human_issues(
    df: pd.DataFrame,
    *,
    pattern_lookup: CanonicalLookup,
    report: NormalizationReport,
) -> pd.DataFrame:
    require_columns(df, KEY + ["relevant_to_pattern_study", "patterns_present"], "human_issues")
    out = normalize_key_columns(df)
    relevance_lookup = CanonicalLookup(RELEVANCE_VALUES, label="relevant_to_pattern_study")
    relevance: list[str] = []
    patterns: list[str] = []
    for pos, row in out.iterrows():
        relevance.append(
            relevance_lookup.canonicalize(
                row["relevant_to_pattern_study"],
                report=report,
                row=pos + 2,
                column="relevant_to_pattern_study",
            )
        )
        names = canonicalize_multivalue(
            row.get("patterns_present", ""),
            pattern_lookup,
            report=report,
            row=pos + 2,
            column="patterns_present",
        )
        patterns.append("|".join(names))
    out["relevant_to_pattern_study"] = relevance
    out["patterns_present"] = patterns
    return out


def normalize_human_pairs(
    df: pd.DataFrame,
    *,
    pattern_lookup: CanonicalLookup,
    report: NormalizationReport,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PAIR_KEY + ["human_verdict"])
    require_columns(df, PAIR_KEY + ["human_verdict"], "human_pairs")
    out = normalize_key_columns(df)
    verdict_lookup = CanonicalLookup(VERDICTS, label="human_verdict")
    patterns: list[str] = []
    verdicts: list[str] = []
    for pos, row in out.iterrows():
        patterns.append(
            pattern_lookup.canonicalize(
                row["pattern"], report=report, row=pos + 2, column="pattern"
            )
        )
        verdicts.append(
            verdict_lookup.canonicalize(
                row["human_verdict"],
                report=report,
                row=pos + 2,
                column="human_verdict",
            )
        )
    out["pattern"] = patterns
    out["human_verdict"] = verdicts
    duplicated = out.duplicated(PAIR_KEY, keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, PAIR_KEY].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"human_pairs contém pares duplicados após normalização: {examples}")
    return out


def human_pairs_from_files(human_issues: pd.DataFrame, human_pairs: pd.DataFrame) -> pd.DataFrame:
    generated_rows: list[dict[str, str]] = []
    existing_keys = set()
    if not human_pairs.empty:
        existing_keys = set(map(tuple, human_pairs[PAIR_KEY].itertuples(index=False, name=None)))

    for row in human_issues.to_dict("records"):
        relevance = row["relevant_to_pattern_study"]
        patterns = [item for item in str(row.get("patterns_present", "")).split("|") if item]
        for pattern in patterns:
            key = (row["repository"], row["issue_number"], pattern)
            if key in existing_keys:
                continue
            fallback = relevance if relevance in {"yes", "uncertain", "insufficient_context"} else ""
            if fallback:
                generated_rows.append(
                    {
                        "repository": key[0],
                        "issue_number": key[1],
                        "pattern": key[2],
                        "human_verdict": fallback,
                        "label_source": "inferred_from_issue_file",
                    }
                )

    if not human_pairs.empty:
        human_pairs = human_pairs.copy()
        human_pairs["label_source"] = "pair_file"
    frames = [df for df in [human_pairs, pd.DataFrame(generated_rows)] if not df.empty]
    if not frames:
        return pd.DataFrame(columns=PAIR_KEY + ["human_verdict", "label_source"])
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.drop_duplicates(PAIR_KEY, keep="first")


def expand_stage1(
    stage1: pd.DataFrame,
    *,
    pattern_lookup: CanonicalLookup,
    report: NormalizationReport,
) -> pd.DataFrame:
    require_columns(stage1, KEY + ["candidates"], "stage1_results")
    stage1 = normalize_key_columns(stage1)
    rows: list[dict[str, str]] = []
    for pos, row in stage1.iterrows():
        patterns = canonicalize_multivalue(
            row.get("candidates", ""),
            pattern_lookup,
            report=report,
            row=pos + 2,
            column="candidates",
        )
        for pattern in patterns:
            rows.append(
                {
                    "repository": row["repository"],
                    "issue_number": row["issue_number"],
                    "pattern": pattern,
                }
            )
    return pd.DataFrame(rows, columns=PAIR_KEY).drop_duplicates()


def normalize_stage2_results(
    stage2: pd.DataFrame,
    *,
    pattern_lookup: CanonicalLookup,
    report: NormalizationReport,
) -> pd.DataFrame:
    require_columns(stage2, PAIR_KEY + ["verdict"], "stage2_results")
    out = normalize_key_columns(stage2)
    if "request_status" in out.columns:
        out["request_status"] = out["request_status"].map(clean_token)
        ok_mask = out["request_status"].map(lambda x: x.casefold() == "succeeded")
    else:
        ok_mask = pd.Series(True, index=out.index)

    verdict_lookup = CanonicalLookup(VERDICTS, label="verdict")
    for pos in out.index:
        out.at[pos, "pattern"] = pattern_lookup.canonicalize(
            out.at[pos, "pattern"], report=report, row=int(pos) + 2, column="pattern"
        )
        if ok_mask.loc[pos]:
            out.at[pos, "verdict"] = verdict_lookup.canonicalize(
                out.at[pos, "verdict"], report=report, row=int(pos) + 2, column="verdict"
            )
        else:
            out.at[pos, "verdict"] = clean_token(out.at[pos, "verdict"])

    duplicated = out.loc[ok_mask].duplicated(PAIR_KEY, keep=False)
    if duplicated.any():
        examples = out.loc[ok_mask].loc[duplicated, PAIR_KEY].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"stage2_results contém pares duplicados após normalização: {examples}")
    return out.loc[ok_mask].copy()


def safe_div(num: int | float, den: int | float) -> float | None:
    return float(num / den) if den else None


def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    tp = sum(t == "yes" and p == "yes" for t, p in zip(y_true, y_pred))
    tn = sum(t == "no" and p == "no" for t, p in zip(y_true, y_pred))
    fp = sum(t == "no" and p == "yes" for t, p in zip(y_true, y_pred))
    fn = sum(t == "yes" and p == "no" for t, p in zip(y_true, y_pred))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "n": len(y_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(args: argparse.Namespace) -> int:
    catalog, catalog_report = load_patterns_with_report(Path(args.patterns))
    pattern_lookup = CanonicalLookup(catalog.names, label="pattern_catalog")

    human_issues_raw, human_issues_report = read_normalized(
        Path(args.human_issues),
        source="human_issues",
        token_columns=KEY + ["relevant_to_pattern_study", "patterns_present"],
    )
    human_pairs_file = Path(args.human_pairs)
    if human_pairs_file.exists():
        human_pairs_raw, human_pairs_report = read_normalized(
            human_pairs_file,
            source="human_pairs",
            token_columns=PAIR_KEY + ["human_verdict"],
        )
    else:
        human_pairs_raw = pd.DataFrame()
        human_pairs_report = NormalizationReport(source="human_pairs_missing")
        human_pairs_report.warnings.append("Arquivo de pares humanos não encontrado; usando inferências controladas.")

    stage1_raw, stage1_report = read_normalized(
        Path(args.stage1),
        source="stage1_results",
        token_columns=KEY + ["candidates", "request_status"],
    )
    stage2_raw, stage2_report = read_normalized(
        Path(args.stage2),
        source="stage2_results",
        token_columns=PAIR_KEY + ["verdict", "request_status"],
    )

    human_issues = normalize_human_issues(
        human_issues_raw, pattern_lookup=pattern_lookup, report=human_issues_report
    )
    human_pairs = normalize_human_pairs(
        human_pairs_raw, pattern_lookup=pattern_lookup, report=human_pairs_report
    )
    human_pair_norm = human_pairs_from_files(human_issues, human_pairs)
    stage1_pairs = expand_stage1(
        stage1_raw, pattern_lookup=pattern_lookup, report=stage1_report
    )
    stage2_ok = normalize_stage2_results(
        stage2_raw, pattern_lookup=pattern_lookup, report=stage2_report
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    human_issues.to_csv(output_dir / "human_issues_clean.csv", index=False)
    human_pairs.to_csv(output_dir / "human_pairs_clean.csv", index=False)
    stage1_raw.to_csv(output_dir / "stage1_clean.csv", index=False)
    stage2_ok.to_csv(output_dir / "stage2_clean.csv", index=False)

    reports = (
        catalog_report,
        human_issues_report,
        human_pairs_report,
        stage1_report,
        stage2_report,
    )
    normalization = merge_reports(*reports)
    write_json(output_dir / "evaluation_normalization_report.json", normalization)
    change_columns = [
        "source",
        "scope",
        "row",
        "column",
        "reason",
        "before_preview",
        "after_preview",
        "before_sha256",
        "after_sha256",
    ]
    changes = [change for report in reports for change in report.changes]
    pd.DataFrame(changes, columns=change_columns).to_csv(
        output_dir / "evaluation_normalization_changes.csv", index=False
    )

    human_positive = human_pair_norm[
        human_pair_norm["human_verdict"].isin(["yes", "uncertain", "insufficient_context"])
    ].copy()
    s1_eval = human_positive.merge(stage1_pairs.assign(stage1_candidate="yes"), on=PAIR_KEY, how="left")
    s1_eval["stage1_candidate"] = s1_eval["stage1_candidate"].fillna("no")
    false_negatives = s1_eval[s1_eval["stage1_candidate"] == "no"].copy()
    false_negatives.to_csv(output_dir / "stage1_false_negatives.csv", index=False)

    stage1_counts = stage1_raw.copy()
    stage1_counts["candidate_count_eval"] = [
        len(canonicalize_multivalue(value, pattern_lookup)) for value in stage1_counts["candidates"]
    ]
    stage1_metrics = {
        "human_positive_pairs": len(human_positive),
        "candidate_hits": int((s1_eval["stage1_candidate"] == "yes").sum()),
        "candidate_recall": safe_div(
            int((s1_eval["stage1_candidate"] == "yes").sum()), len(human_positive)
        ),
        "false_negative_count": len(false_negatives),
        "mean_candidates_per_case": float(stage1_counts["candidate_count_eval"].mean())
        if len(stage1_counts)
        else None,
        "median_candidates_per_case": float(stage1_counts["candidate_count_eval"].median())
        if len(stage1_counts)
        else None,
        "issues_with_zero_candidates": int((stage1_counts["candidate_count_eval"] == 0).sum()),
    }

    labeled = stage2_ok.merge(
        human_pair_norm[PAIR_KEY + ["human_verdict", "label_source"]],
        on=PAIR_KEY,
        how="left",
    )
    unlabeled = labeled[labeled["human_verdict"].isna() | (labeled["human_verdict"] == "")].copy()
    unlabeled.to_csv(output_dir / "pairs_requiring_human_review.csv", index=False)
    labeled_only = labeled[~labeled.index.isin(unlabeled.index)].copy()
    labeled_only.to_csv(output_dir / "stage2_labeled_comparison.csv", index=False)

    binary = labeled_only[
        labeled_only["human_verdict"].isin(["yes", "no"])
        & labeled_only["verdict"].isin(["yes", "no"])
    ].copy()
    metrics_binary = binary_metrics(binary["human_verdict"].tolist(), binary["verdict"].tolist())

    confusion = pd.crosstab(
        labeled_only["human_verdict"],
        labeled_only["verdict"],
        rownames=["human"],
        colnames=["llm"],
        dropna=False,
    )
    confusion.to_csv(output_dir / "stage2_confusion_matrix.csv")

    stage2_metrics = {
        "generated_pairs": len(stage2_ok),
        "labeled_pairs": len(labeled_only),
        "unlabeled_pairs": len(unlabeled),
        "coverage_for_metrics": safe_div(len(labeled_only), len(stage2_ok)),
        "binary_yes_no": metrics_binary,
        "llm_verdict_counts": stage2_ok["verdict"].value_counts(dropna=False).to_dict(),
        "human_verdict_counts_on_labeled": labeled_only["human_verdict"]
        .value_counts(dropna=False)
        .to_dict(),
        "uncertain_rate_llm": safe_div(int((stage2_ok["verdict"] == "uncertain").sum()), len(stage2_ok)),
        "insufficient_context_rate_llm": safe_div(
            int((stage2_ok["verdict"] == "insufficient_context").sum()), len(stage2_ok)
        ),
    }

    report = {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "normalization_change_count": normalization["total_changes"],
        "warnings": [
            "As métricas dos 30 casos são diagnósticas, não estimativas finais.",
            "Pares sem rótulo humano foram excluídos e exportados para revisão."
            if len(unlabeled)
            else "Todos os pares gerados possuem rótulo humano.",
        ],
    }
    write_json(output_dir / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia o pipeline contra anotações humanas.")
    parser.add_argument("--human-issues", required=True)
    parser.add_argument("--human-pairs", required=True)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--stage2", required=True)
    parser.add_argument("--patterns", default="blockchain_patterns_keywords_v3.csv")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return evaluate(args)
    except Exception as exc:
        print(f"ERRO: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
