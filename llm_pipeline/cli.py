"""Interface de linha de comando e orquestração do pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .aggregation import aggregate_issue_results
from .artifacts import build_dry_run_artifacts, save_manifest
from .checkpoint import CheckpointIntegrityError, ResultStore
from .client import close_client, create_client
from .config import (
    DEFAULT_BATCH_MAX_BYTES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_POLL_SECONDS,
    DEFAULT_SEED,
    DEFAULT_STAGE1_MODEL,
    DEFAULT_STAGE1_THINKING_LEVEL,
    DEFAULT_STAGE2_MODEL,
    DEFAULT_STAGE2_THINKING_LEVEL,
    DEFAULT_TEMPERATURE,
    THINKING_LEVELS,
)
from .data import load_input, load_patterns_with_report, prepare_issue
from .normalization import merge_reports
from .stages import (
    run_stage1_batch,
    run_stage1_sync,
    run_stage2_batch,
    run_stage2_sync,
    pipeline_integrity_report,
    stage2_pairs_from_stage1,
)
from .utils import sha256_file, stable_custom_id, utc_run_id, write_dataframe_csv, write_json


def _empty_stage2_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "repository",
            "issue_number",
            "issue_key",
            "custom_id",
            "pattern",
            "request_status",
            "error",
            "verdict",
            "mechanism_match",
            "scope_match",
            "focus_match",
            "evidence_text",
            "evidence_location",
            "justification",
            "adoption_status",
            "false_friend_detected",
            "pattern_challenge_categories",
            "confidence",
            "alternative_pattern",
            "overlap_with",
            "input_truncated",
            "message_id",
            "response_model",
            "stop_reason",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "thoughts_tokens",
            "cached_input_tokens",
        ]
    )


def _validate_resume_metadata(run_dir: Path, input_path: Path, patterns_path: Path, args: argparse.Namespace) -> None:
    """Reject accidental reuse of a run with a different methodological input."""
    metadata_path = run_dir / "run_metadata.json"
    known_artifacts = {"stage1_results.csv", "stage2_results.csv", "stage1_checkpoint.json", "stage2_checkpoint.json"}
    if not metadata_path.exists():
        if not any((run_dir / name).exists() for name in known_artifacts):
            raise FileExistsError(f"Diretório existente sem artefatos de run reconhecíveis: {run_dir}")
        print("[resume] run legado sem metadata completa; validando CSVs existentes.")
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"run_metadata inválido em {metadata_path}: {exc}") from exc
    expected = {
        "input_sha256": sha256_file(input_path),
        "patterns_sha256": sha256_file(patterns_path),
        "max_input_chars": args.max_input_chars,
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "temperature": args.temperature,
        "seed": args.seed,
        "stage1_thinking_level": args.stage1_thinking_level,
        "stage2_thinking_level": args.stage2_thinking_level,
        "stage1_max_tokens": args.stage1_max_tokens,
        "stage2_max_tokens": args.stage2_max_tokens,
        "limit": args.limit,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if key in metadata and metadata[key] != value}
    if mismatches:
        raise ValueError(f"Não é seguro retomar {run_dir}: parâmetros/metadados divergentes: {mismatches}")


def run_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    patterns_path = Path(args.patterns)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not patterns_path.exists():
        raise FileNotFoundError(patterns_path)

    df, input_report = load_input(input_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit deve ser maior que zero")
        df = df.head(args.limit).copy()
    catalog, pattern_report = load_patterns_with_report(patterns_path)
    prepared_issues = [prepare_issue(row, args.max_input_chars) for _, row in df.iterrows()]
    prepared_by_key = {(x.repository, x.issue_number): x for x in prepared_issues}

    run_id = args.run_id or utc_run_id()
    run_dir = Path(args.output_dir) / run_id
    existing_run = run_dir.exists() and any(run_dir.iterdir())
    if existing_run:
        _validate_resume_metadata(run_dir, input_path, patterns_path, args)
        print(f"[resume] reutilizando run existente: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_manifest(
            run_dir, input_path, patterns_path, df, prepared_issues, catalog, args, input_report, pattern_report,
        )

    if args.mode == "dry-run":
        build_dry_run_artifacts(prepared_issues, catalog, args, run_dir)
        return 0

    client = None

    def get_client():
        nonlocal client
        if client is None:
            client = create_client()
        return client

    try:
        stage1_existing = ResultStore(run_dir / "stage1_results.csv", stage="stage1")
        stage1_pending = len({item.custom_id_stage1 for item in prepared_issues} - stage1_existing.completed_ids())
        if args.mode == "sync":
            stage1_df = run_stage1_sync(
                get_client() if stage1_pending else None,
                prepared_issues,
                catalog,
                args.stage1_model,
                args.temperature,
                args.stage1_max_tokens,
                args.seed,
                args.stage1_thinking_level,
                run_dir,
            )
        else:
            stage1_df = run_stage1_batch(
                get_client() if stage1_pending else None,
                prepared_issues,
                catalog,
                args.stage1_model,
                args.temperature,
                args.stage1_max_tokens,
                args.seed,
                args.stage1_thinking_level,
                args.batch_size,
                args.batch_max_bytes,
                args.poll_seconds,
                run_dir,
            )

        pairs = stage2_pairs_from_stage1(stage1_df)
        write_json(
            run_dir / "stage2_pair_manifest.json",
            [
                {
                    "repository": repository,
                    "issue_number": issue_number,
                    "pattern": pattern,
                    "custom_id": stable_custom_id("s2", repository, issue_number, pattern),
                }
                for repository, issue_number, pattern in pairs
            ],
        )

        stage2_existing = ResultStore(run_dir / "stage2_results.csv", stage="stage2")
        expected_stage2_ids = {stable_custom_id("s2", repository, issue_number, pattern) for repository, issue_number, pattern in pairs}
        stage2_pending = len(expected_stage2_ids - stage2_existing.completed_ids())
        if not pairs:
            stage2_df = stage2_existing.dataframe.copy()
            if stage2_df.empty:
                stage2_existing.ensure_exists(_empty_stage2_dataframe().columns.tolist())
                stage2_df = stage2_existing.dataframe.copy()
        elif args.mode == "sync":
            stage2_df = run_stage2_sync(
                get_client() if stage2_pending else None,
                prepared_by_key,
                pairs,
                catalog,
                args.stage2_model,
                args.temperature,
                args.stage2_max_tokens,
                args.seed,
                args.stage2_thinking_level,
                run_dir,
            )
        else:
            stage2_df = run_stage2_batch(
                get_client() if stage2_pending else None,
                prepared_by_key,
                pairs,
                catalog,
                args.stage2_model,
                args.temperature,
                args.stage2_max_tokens,
                args.seed,
                args.stage2_thinking_level,
                args.batch_size,
                args.batch_max_bytes,
                args.poll_seconds,
                run_dir,
            )
    finally:
        if client is not None:
            close_client(client)

    issue_df = aggregate_issue_results(stage1_df, stage2_df, input_df=df)
    write_dataframe_csv(run_dir / "issue_results.csv", issue_df)
    integrity = pipeline_integrity_report(prepared_issues, stage1_df, stage2_df)
    write_json(run_dir / "integrity_report.json", integrity)
    print(
        "[validation] "
        f"expected_stage2_pairs={integrity['expected_stage2_pairs']} "
        f"completed_stage2_pairs={integrity['completed_stage2_pairs']} "
        f"missing={len(integrity['missing_stage2_pairs'])} "
        f"duplicates={len(integrity['duplicate_stage2_pairs'])} "
        f"status={integrity['status']}"
    )

    candidate_counts = pd.to_numeric(
        stage1_df.get("candidate_count", pd.Series(0, index=stage1_df.index)), errors="coerce"
    ).fillna(0)
    summary = {
        "run_dir": str(run_dir.resolve()),
        "issues": len(stage1_df),
        "stage1_succeeded": int((stage1_df["request_status"] == "succeeded").sum()),
        "stage1_errored": int((stage1_df["request_status"] != "succeeded").sum()),
        "issues_with_candidates": int((candidate_counts > 0).sum()),
        "stage2_pairs": integrity["expected_stage2_pairs"],
        "stage2_succeeded": int((stage2_df["request_status"] == "succeeded").sum())
        if not stage2_df.empty
        else 0,
        "stage2_errored": int((stage2_df["request_status"] != "succeeded").sum())
        if not stage2_df.empty
        else 0,
        "verdict_counts": stage2_df["verdict"].value_counts(dropna=False).to_dict()
        if not stage2_df.empty
        else {},
        "relevance_counts": issue_df["relevant_to_pattern_study"]
        .value_counts(dropna=False)
        .to_dict(),
        "integrity": integrity,
    }
    write_json(run_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def validate_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    patterns_path = Path(args.patterns)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not patterns_path.exists():
        raise FileNotFoundError(patterns_path)
    df, input_report = load_input(input_path)
    catalog, pattern_report = load_patterns_with_report(patterns_path)
    prepared = [prepare_issue(row, args.max_input_chars) for _, row in df.iterrows()]
    normalization = merge_reports(input_report, pattern_report)
    report = {
        "input_rows": len(df),
        "pattern_count": len(catalog.names),
        "input_sha256": sha256_file(input_path),
        "patterns_sha256": sha256_file(patterns_path),
        "input_encoding": input_report.encoding,
        "input_delimiter": input_report.delimiter,
        "normalization_change_count": normalization["total_changes"],
        "normalization_summaries": [input_report.summary(), pattern_report.summary()],
        "truncated_rows": sum(item.input_truncated for item in prepared),
        "max_original_chars": max((item.original_char_count for item in prepared), default=0),
        "max_included_chars": max((item.included_char_count for item in prepared), default=0),
        "duplicate_keys": 0,
        "status": "ok",
    }
    if args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_dir / "input_clean.csv", index=False)
        catalog.dataframe.to_csv(report_dir / "pattern_catalog_clean.csv", index=False)
        write_json(report_dir / "normalization_report.json", normalization)
        changes = [change for item in (input_report, pattern_report) for change in item.changes]
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
        pd.DataFrame(changes, columns=change_columns).to_csv(
            report_dir / "normalization_changes.csv", index=False
        )
        report["report_dir"] = str(report_dir.resolve())
        write_json(report_dir / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline Gemini Stage 1 + Stage 2 alinhado ao Manual de Anotação Humana v0.2."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Normaliza, valida CSVs e estima truncamento sem chamar API."
    )
    validate.add_argument("--input", required=True)
    validate.add_argument("--patterns", default="blockchain_patterns_keywords_v3.csv")
    validate.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    validate.add_argument(
        "--report-dir",
        default="",
        help="Opcional: salva CSVs limpos e relatório detalhado de normalização.",
    )
    validate.set_defaults(func=validate_command)

    run = subparsers.add_parser("run", help="Executa o pipeline Gemini ou gera dry-run.")
    run.add_argument("--input", required=True)
    run.add_argument("--patterns", default="blockchain_patterns_keywords_v3.csv")
    run.add_argument("--mode", choices=["sync", "batch", "dry-run"], default="sync")
    run.add_argument("--output-dir", default="outputs/runs")
    run.add_argument("--run-id", default="")
    run.add_argument("--limit", type=int)
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="Legado: um --run-id existente agora é retomado automaticamente após validação.",
    )
    run.add_argument("--stage1-model", default=DEFAULT_STAGE1_MODEL)
    run.add_argument("--stage2-model", default=DEFAULT_STAGE2_MODEL)
    run.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument(
        "--stage1-thinking-level",
        choices=THINKING_LEVELS,
        default=DEFAULT_STAGE1_THINKING_LEVEL,
    )
    run.add_argument(
        "--stage2-thinking-level",
        choices=THINKING_LEVELS,
        default=DEFAULT_STAGE2_THINKING_LEVEL,
    )
    run.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    run.add_argument("--stage1-max-tokens", type=int, default=2048)
    run.add_argument("--stage2-max-tokens", type=int, default=2048)
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--batch-max-bytes", type=int, default=DEFAULT_BATCH_MAX_BYTES)
    run.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    run.set_defaults(func=run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print(
            "Execução interrompida pelo usuário. Resultados concluídos já persistidos foram preservados; "
            "execute novamente o mesmo comando para retomar.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
