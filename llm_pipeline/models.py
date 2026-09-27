"""Estruturas de dados internas do pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    # P6: artifact_type tracks whether the artefact is a pull_request or issue so that
    # evidence_location can be mapped to "pull_request_description" instead of "body".
    artifact_type: str = ""
    source_sections: dict[str, str] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    selected_spans: dict[str, Any] = field(default_factory=dict)
    optimization: Any = None


@dataclass(frozen=True)
class PatternCatalog:
    dataframe: pd.DataFrame
    names: list[str]
    by_name: dict[str, dict[str, str]]
    compact_catalog: str
    full_catalog: str
