"""Persistência dos artefatos de auditoria e metadados da execução."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import os
import sys
from pathlib import Path

import pandas as pd

from .client import import_genai
from .config import API_FAMILY, CATALOG_VERSION, MANUAL_VERSION, PIPELINE_VERSION, PROVIDER
from .models import PatternCatalog, PreparedIssue
from .normalization import NormalizationReport, merge_reports
from .prompts import COMMON_METHOD_RULES, STAGE1_RULES, STAGE2_RULES
from .requests import stage1_request_params
from .schemas import stage1_schema, stage2_schema
from .utils import append_jsonl, json_dumps, sha256_file, sha256_text, utc_now_iso, write_json


def build_dry_run_artifacts(
    prepared_issues: list[PreparedIssue],
    catalog: PatternCatalog,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    stage1_requests_path = run_dir / "stage1_requests.jsonl"
    # Dry-run requests are derived artifacts: replace atomically, never append duplicates.
    temporary = stage1_requests_path.with_suffix('.jsonl.tmp')
    temporary.write_text('', encoding='utf-8')
    for item in prepared_issues:
        append_jsonl(
            temporary,
            {
                "custom_id": item.custom_id_stage1,
                "params": stage1_request_params(
                    item,
                    catalog,
                    args.stage1_model,
                    args.temperature,
                    args.stage1_max_tokens,
                    args.seed,
                    args.stage1_thinking_level,
                ),
            },
        )
    os.replace(temporary, stage1_requests_path)
    print(f"Dry-run concluído. Requisições Stage 1: {stage1_requests_path}")


def save_manifest(
    run_dir: Path,
    input_path: Path,
    patterns_path: Path,
    df: pd.DataFrame,
    prepared_issues: list[PreparedIssue],
    catalog: PatternCatalog,
    args: argparse.Namespace,
    input_report: NormalizationReport,
    pattern_report: NormalizationReport,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, run_dir / "input_raw_snapshot.csv")
    df.to_csv(run_dir / "input_clean_snapshot.csv", index=False)
    shutil.copy2(patterns_path, run_dir / "pattern_catalog_raw_snapshot.csv")
    catalog.dataframe.to_csv(run_dir / "pattern_catalog_clean_snapshot.csv", index=False)
    normalization = merge_reports(input_report, pattern_report)
    write_json(run_dir / "normalization_report.json", normalization)
    changes = [change for report in (input_report, pattern_report) for change in report.changes]
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
        run_dir / "normalization_changes.csv", index=False
    )
    (run_dir / "pattern_catalog_full.txt").write_text(catalog.full_catalog, encoding="utf-8")
    (run_dir / "pattern_catalog_compact.txt").write_text(
        catalog.compact_catalog, encoding="utf-8"
    )

    prepared_manifest = [
        {
            "repository": item.repository,
            "issue_number": item.issue_number,
            "issue_key": item.issue_key,
            "custom_id_stage1": item.custom_id_stage1,
            "original_char_count": item.original_char_count,
            "included_char_count": item.included_char_count,
            "input_truncated": item.input_truncated,
            "artifact_sha256": sha256_text(item.artifact_text),
        }
        for item in prepared_issues
    ]
    write_json(run_dir / "request_manifest.json", prepared_manifest)

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "provider": PROVIDER,
        "api_family": API_FAMILY,
        "manual_version": MANUAL_VERSION,
        "catalog_version": CATALOG_VERSION,
        "created_at_utc": utc_now_iso(),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "patterns_path": str(patterns_path.resolve()),
        "patterns_sha256": sha256_file(patterns_path),
        "input_rows": len(df),
        "normalization_change_count": normalization["total_changes"],
        "input_encoding": input_report.encoding,
        "input_delimiter": input_report.delimiter,
        "patterns_encoding": pattern_report.encoding,
        "patterns_delimiter": pattern_report.delimiter,
        "pattern_count": len(catalog.names),
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "temperature": args.temperature,
        "seed": args.seed,
        "stage1_thinking_level": args.stage1_thinking_level,
        "stage2_thinking_level": args.stage2_thinking_level,
        "stage1_max_tokens": args.stage1_max_tokens,
        "stage2_max_tokens": args.stage2_max_tokens,
        "max_input_chars": args.max_input_chars,
        "mode": args.mode,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "batch_max_bytes": args.batch_max_bytes,
        "poll_seconds": args.poll_seconds,
        "stage1_schema_sha256": sha256_text(json_dumps(stage1_schema(catalog.names))),
        "stage2_schema_sha256": sha256_text(json_dumps(stage2_schema(catalog.names))),
        "common_rules_sha256": sha256_text(COMMON_METHOD_RULES),
        "stage1_rules_sha256": sha256_text(STAGE1_RULES),
        "stage2_rules_sha256": sha256_text(STAGE2_RULES),
        "python_version": sys.version,
    }
    metadata['profile'] = getattr(args, 'profile', 'legacy')
    metadata['quantization'] = 'provider_managed_not_exposed'
    try:
        metadata['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        metadata['working_tree_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata['commit_hash'] = None
    from .optimization import OptimizationConfig
    from .optimization_runtime import optimization_metadata
    metadata.update(optimization_metadata(OptimizationConfig.from_environment(enabled=metadata['profile']=='optimized'),catalog))
    try:
        genai = import_genai()
        metadata["google_genai_sdk_version"] = getattr(genai, "__version__", "unknown")
    except RuntimeError:
        metadata["google_genai_sdk_version"] = "not_installed"
    write_json(run_dir / "run_metadata.json", metadata)
