"""Leitura, normalização, validação e preparação dos artefatos de entrada."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from .config import OPTIONAL_INPUT_COLUMNS, REQUIRED_INPUT_COLUMNS, REQUIRED_PATTERN_COLUMNS
from .models import PatternCatalog, PreparedIssue
from .normalization import (
    CanonicalLookup,
    NormalizationReport,
    canonical_display,
    clean_text,
    clean_token,
    normalize_dataframe_cells,
    normalize_dataframe_columns,
    read_csv_robust,
    resolve_repository_columns,
)
from .utils import stable_custom_id

_INPUT_TOKEN_COLUMNS = [
    "repository",
    "repository_full_name",
    "repository_category",
    "issue_number",
    "type",
    "state",
    "html_url",
    "sample_group",
    "selection_trigger",
]

_PATTERN_TOKEN_COLUMNS = ["pattern", "category", "subcategory"]


def _normalize_issue_number(value: str) -> str:
    value = clean_token(value)
    if re.fullmatch(r"\d+\.0+", value):
        return value.split(".", 1)[0]
    return value


def validate_input_dataframe(
    df: pd.DataFrame,
    report: NormalizationReport | None = None,
) -> pd.DataFrame:
    normalized = normalize_dataframe_columns(df, report)
    normalized = resolve_repository_columns(normalized, report, require=True)
    missing = REQUIRED_INPUT_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError(f"CSV de entrada sem colunas obrigatórias: {sorted(missing)}")

    normalized = normalized.copy()
    for column, default in OPTIONAL_INPUT_COLUMNS.items():
        if column not in normalized.columns:
            normalized[column] = default

    normalized = normalize_dataframe_cells(
        normalized,
        token_columns=[column for column in _INPUT_TOKEN_COLUMNS if column in normalized.columns],
        report=report,
    )

    old_numbers = normalized["issue_number"].copy()
    normalized["issue_number"] = normalized["issue_number"].map(_normalize_issue_number)
    if report is not None:
        for pos, (before, after) in enumerate(zip(old_numbers, normalized["issue_number"])):
            if before != after:
                report.add_change(
                    scope="cell",
                    before=before,
                    after=after,
                    reason="normalized_integer_issue_number",
                    row=pos + 2,
                    column="issue_number",
                )

    blank_key = (normalized["repository"] == "") | (normalized["issue_number"] == "")
    if blank_key.any():
        rows = (normalized.index[blank_key] + 2).tolist()[:10]
        raise ValueError(f"Há repository/issue_number vazio nas linhas CSV: {rows}")

    duplicated = normalized.duplicated(subset=["repository", "issue_number"], keep=False)
    if duplicated.any():
        examples = (
            normalized.loc[duplicated, ["repository", "issue_number"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "A chave composta (repository, issue_number) deve ser única após a normalização. "
            f"Exemplos duplicados: {examples}"
        )

    normalized["issue_key"] = normalized["repository"] + "#" + normalized["issue_number"]
    return normalized


def load_input(path: Path) -> tuple[pd.DataFrame, NormalizationReport]:
    df, report = read_csv_robust(path, source="input_csv")
    return validate_input_dataframe(df, report), report


def load_patterns_with_report(path: Path) -> tuple[PatternCatalog, NormalizationReport]:
    df, report = read_csv_robust(path, source="pattern_catalog")
    missing = REQUIRED_PATTERN_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Catálogo sem colunas obrigatórias: {sorted(missing)}")

    df = normalize_dataframe_cells(
        df,
        token_columns=[column for column in _PATTERN_TOKEN_COLUMNS if column in df.columns],
        report=report,
    )
    original_patterns = df["pattern"].copy()
    df["pattern"] = df["pattern"].map(canonical_display)
    for pos, (before, after) in enumerate(zip(original_patterns, df["pattern"])):
        if report is not None and before != after:
            report.add_change(
                scope="cell",
                before=before,
                after=after,
                reason="normalized_pattern_display",
                row=pos + 2,
                column="pattern",
            )

    if (df["pattern"] == "").any() or (df["description"] == "").any():
        raise ValueError("Catálogo contém pattern ou description vazio.")
    if df["pattern"].duplicated().any():
        duplicates = df.loc[df["pattern"].duplicated(keep=False), "pattern"].tolist()
        raise ValueError(f"Nomes de pattern duplicados no catálogo: {duplicates}")

    names = df["pattern"].tolist()
    # Também rejeita colisões como A-B vs A_B. Isso impede canonicalização
    # silenciosamente ambígua em arquivos humanos ou respostas da LLM.
    CanonicalLookup(names, label="pattern_catalog")

    by_name: dict[str, dict[str, str]] = {}
    compact_lines: list[str] = []
    full_lines: list[str] = []
    for row in df.to_dict("records"):
        name = row["pattern"]
        subcategory = row["subcategory"] or row["category"]
        description = clean_text(row["description"])
        by_name[name] = {
            "pattern": name,
            "category": row["category"],
            "subcategory": row["subcategory"],
            "description": description,
        }
        compact_desc = re.sub(r"\s+", " ", description).strip()
        if len(compact_desc) > 260:
            compact_desc = compact_desc[:257].rstrip() + "..."
        compact_lines.append(f"- {name} [{subcategory}]: {compact_desc}")
        full_lines.append(f"- {name} [{subcategory}]: {description}")

    return (
        PatternCatalog(
            dataframe=df,
            names=names,
            by_name=by_name,
            compact_catalog="\n".join(compact_lines),
            full_catalog="\n".join(full_lines),
        ),
        report,
    )


def load_patterns(path: Path) -> PatternCatalog:
    catalog, _ = load_patterns_with_report(path)
    return catalog


def head_tail(text: str, budget: int) -> str:
    text = text.strip()
    if len(text) <= budget:
        return text
    if budget <= 80:
        return text[:budget]
    marker = f"\n[... {len(text) - budget} caracteres omitidos ...]\n"
    usable = max(1, budget - len(marker))
    head = math.ceil(usable * 0.65)
    tail = usable - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def prepare_issue(row: pd.Series, max_chars: int) -> PreparedIssue:
    repository = clean_token(row["repository"])
    issue_number = clean_token(row["issue_number"])
    title = clean_text(row["issue_title"])
    body = clean_text(row["issue_body"])
    comments = clean_text(row.get("concatenated_comments", ""))
    artifact_type = clean_token(row.get("type", ""))
    labels = clean_text(row.get("labels", ""))
    state = clean_token(row.get("state", ""))

    original_sections = {
        "title": title,
        "body": body,
        "comments": comments,
        "type": artifact_type,
        "labels": labels,
        "state": state,
    }
    original_char_count = sum(len(v) for v in original_sections.values())

    fixed = (
        f"<repository>{repository}</repository>\n"
        f"<issue_number>{issue_number}</issue_number>\n"
        f"<artifact_type>{artifact_type}</artifact_type>\n"
        f"<state>{state}</state>\n"
        f"<labels>{labels}</labels>\n"
        f"<title>{title}</title>\n"
    )
    closing_overhead = len("<body></body>\n<comments></comments>")
    variable_budget = max(200, max_chars - len(fixed) - closing_overhead)

    body_weight = 0.65 if comments else 1.0
    body_budget = max(100, int(variable_budget * body_weight))
    comments_budget = max(0, variable_budget - body_budget)

    body_prepared = head_tail(body, body_budget)
    comments_prepared = head_tail(comments, comments_budget) if comments_budget else ""

    artifact_text = fixed + f"<body>{body_prepared}</body>\n" + f"<comments>{comments_prepared}</comments>"
    included_char_count = len(artifact_text)
    input_truncated = body_prepared != body or comments_prepared != comments
    issue_key = f"{repository}#{issue_number}"

    return PreparedIssue(
        repository=repository,
        issue_number=issue_number,
        issue_key=issue_key,
        custom_id_stage1=stable_custom_id("s1", repository, issue_number),
        artifact_text=artifact_text,
        original_char_count=original_char_count,
        included_char_count=included_char_count,
        input_truncated=input_truncated,
    )
