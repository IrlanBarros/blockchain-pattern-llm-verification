from llm_pipeline.cli import build_parser


def test_cli_accepts_validate_command():
    parser = build_parser()
    args = parser.parse_args(["validate", "--input", "issues.csv"])
    assert args.command == "validate"
    assert args.patterns == "blockchain_patterns_keywords_v3.csv"


def test_cli_accepts_dry_run_mode():
    parser = build_parser()
    args = parser.parse_args(["run", "--input", "issues.csv", "--mode", "dry-run"])
    assert args.mode == "dry-run"
