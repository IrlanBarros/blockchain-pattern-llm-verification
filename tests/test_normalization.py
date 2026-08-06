from pathlib import Path

import pandas as pd
import pytest

from llm_pipeline.data import load_input, load_patterns, validate_input_dataframe
from llm_pipeline.normalization import (
    CanonicalLookup,
    canonicalize_multivalue,
    clean_text,
    clean_token,
    normalize_dataframe_columns,
    read_csv_robust,
)
from llm_pipeline.schemas import normalize_enum, normalize_stage1


def test_headers_spaces_case_hyphen_bom_and_invisible_are_normalized():
    df = pd.DataFrame(
        [[" repo ", " 12 ", " title ", " body "]],
        columns=[" Repository\u200b ", "Issue Number", "\ufeffISSUE-TITLE", " issue body "],
    )
    out = validate_input_dataframe(df)
    assert ["repository", "issue_number", "issue_title", "issue_body"] == [
        column for column in out.columns if column in {"repository", "issue_number", "issue_title", "issue_body"}
    ]
    assert out.loc[0, "repository"] == "repo"
    assert out.loc[0, "issue_number"] == "12"


def test_duplicate_columns_after_normalization_are_rejected():
    df = pd.DataFrame([["a", "b", "1", "t", "x"]], columns=[
        "repository", " Repository ", "issue_number", "issue_title", "issue_body"
    ])
    with pytest.raises(ValueError, match="tornam-se iguais"):
        normalize_dataframe_columns(df)


def test_unicode_spaces_and_invisible_characters_are_removed_conservatively():
    assert clean_token("\u200b  Role\u00a0  Based\tControl  ") == "Role Based Control"
    assert clean_text("\ufefffirst\r\nsecond\u200b") == "first\nsecond"


def test_structural_pattern_variants_are_canonicalized_without_synonyms():
    lookup = CanonicalLookup(["Role-based control", "Proxy contract"], label="patterns")
    assert lookup.canonicalize(" role_based_control ") == "Role-based control"
    assert lookup.canonicalize("ROLE  BASED   CONTROL") == "Role-based control"
    assert lookup.canonicalize("role–based control") == "Role-based control"
    with pytest.raises(ValueError, match="não adivinha sinônimos"):
        lookup.canonicalize("RBAC")


def test_structural_collision_is_rejected():
    with pytest.raises(ValueError, match="ambíguos"):
        CanonicalLookup(["A-B", "A_B"], label="patterns")


def test_multivalue_normalizes_separator_spaces_case_and_duplicates():
    lookup = CanonicalLookup(["Oracle", "Proxy contract"], label="patterns")
    values = canonicalize_multivalue(" oracle | Proxy_contract | ORACLE ", lookup)
    assert values == ["Oracle", "Proxy contract"]


def test_enum_accepts_space_hyphen_underscore_variants():
    assert normalize_enum(" BUG REPORT ", ["bug_report", "bug_fix"], "activity") == "bug_report"
    assert normalize_enum("bug-report", ["bug_report", "bug_fix"], "activity") == "bug_report"


def test_read_csv_robust_accepts_utf8_bom_and_semicolon(tmp_path: Path):
    path = tmp_path / "smoke.csv"
    path.write_bytes(
        "\ufeff Repository ;Issue Number;Issue Title;Issue Body\n repo ;7;Title;Body\n".encode("utf-8")
    )
    df, report = read_csv_robust(path)
    assert report.encoding == "utf-8-sig"
    assert report.delimiter == ";"
    assert df.columns.tolist() == ["repository", "issue_number", "issue_title", "issue_body"]


def test_load_input_preserves_strings_and_normalizes_excel_integer(tmp_path: Path):
    path = tmp_path / "smoke.csv"
    path.write_text(
        "repository,issue_number,issue_title,issue_body\nrepo,15.0,title,body\n",
        encoding="utf-8",
    )
    df, report = load_input(path)
    assert df.loc[0, "issue_number"] == "15"
    assert any(change["reason"] == "normalized_integer_issue_number" for change in report.changes)


def test_full_catalog_has_no_structural_name_collisions():
    catalog = load_patterns(Path(__file__).parents[1] / "blockchain_patterns_keywords_v3.csv")
    lookup = CanonicalLookup(catalog.names, label="pattern_catalog")
    assert lookup.canonicalize("proxy_contract") == "Proxy contract"


def test_stage1_payload_canonicalizes_pattern_and_categories(tmp_path: Path):
    catalog_path = tmp_path / "patterns.csv"
    pd.DataFrame([
        {
            "pattern": "Proxy contract",
            "category": "Contract management",
            "subcategory": "Contract management",
            "description": "Encaminha chamadas.",
        }
    ]).to_csv(catalog_path, index=False)
    catalog = load_patterns(catalog_path)
    payload = {
        "issue_summary": " x ",
        "issue_activity_type": "BUG REPORT",
        "issue_challenge_categories": [" none explicit "],
        "context_status": "sufficient",
        "candidates": [{
            "pattern": "proxy_contract",
            "evidence_text": " evidence ",
            "evidence_location": "BODY",
            "rationale": " rationale ",
            "confidence": "HIGH",
        }],
    }
    out = normalize_stage1(payload, catalog)
    assert out["issue_activity_type"] == "bug_report"
    assert out["issue_challenge_categories"] == ["none_explicit"]
    assert out["candidates"][0]["pattern"] == "Proxy contract"


def test_smoke_schema_repository_full_name_is_accepted_and_category_preserved():
    df = pd.DataFrame([
        {
            "repository_full_name": " owner/repo ",
            "repository_category": " DeFi ",
            "issue_number": "15.0",
            "issue_title": "Title",
            "issue_body": "Body",
        }
    ])
    out = validate_input_dataframe(df)
    assert out.loc[0, "repository"] == "owner/repo"
    assert out.loc[0, "repository_full_name"] == "owner/repo"
    assert out.loc[0, "repository_category"] == "DeFi"
    assert out.loc[0, "issue_number"] == "15"
    assert out.loc[0, "issue_key"] == "owner/repo#15"


def test_repository_and_repository_full_name_conflict_is_rejected():
    df = pd.DataFrame([
        {
            "repository": "owner/repo-a",
            "repository_full_name": "owner/repo-b",
            "issue_number": "1",
            "issue_title": "Title",
            "issue_body": "Body",
        }
    ])
    with pytest.raises(ValueError, match="divergem"):
        validate_input_dataframe(df)
