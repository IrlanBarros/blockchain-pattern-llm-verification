import json
from pathlib import Path

import pytest

from llm_pipeline.cli import _validate_resume_metadata, build_parser
from llm_pipeline.data import load_patterns


def test_cli_accepts_validate_command():
    parser = build_parser()
    args = parser.parse_args(["validate", "--input", "issues.csv"])
    assert args.command == "validate"
    assert args.patterns == "blockchain_patterns_keywords_v3.csv"


def test_cli_accepts_dry_run_mode():
    parser = build_parser()
    args = parser.parse_args(["run", "--input", "issues.csv", "--mode", "dry-run"])
    assert args.mode == "dry-run"


def test_resume_rejects_run_created_with_different_prompt_hash(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    patterns = project_root / "blockchain_patterns_keywords_v3.csv"
    input_path = tmp_path / "input.csv"
    input_path.write_text("repository,issue_number,issue_title,issue_body\nr,1,t,b\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"stage2_rules_sha256": "hash-from-older-methodology"}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["run", "--input", str(input_path), "--patterns", str(patterns), "--run-id", "run"]
    )
    with pytest.raises(ValueError, match="stage2_rules_sha256"):
        _validate_resume_metadata(run_dir, input_path, patterns, load_patterns(patterns), args)
