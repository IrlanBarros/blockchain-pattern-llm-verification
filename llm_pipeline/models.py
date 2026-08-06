"""Estruturas de dados internas do pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PreparedIssue:
    repository: str
    issue_number: str
    issue_key: str
    custom_id_stage1: str
    artifact_text: str
    original_char_count: int
    included_char_count: int
    input_truncated: bool


@dataclass(frozen=True)
class PatternCatalog:
    dataframe: pd.DataFrame
    names: list[str]
    by_name: dict[str, dict[str, str]]
    compact_catalog: str
    full_catalog: str
