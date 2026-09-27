"""Execução dos Stages 1 e 2 com Gemini em modo síncrono ou batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .checkpoint import Checkpoint, CheckpointIntegrityError, ResultStore, resumable_batch, mark_batch_consumed
from .client import (
    batch_custom_id,
    batch_error_text,
    batch_generated_response,
    batch_inlined_responses,
    batch_state_name,
    call_batch_create,
    call_sync,
    chunk_inline_requests,
    is_global_api_error,
    parse_response_payload,
    wait_for_batch,
)
from .confidence import compute_stage2_confidence
from .optimization_runtime import decode_stage, prepare_stage2_contexts, audit_request
from .telemetry import observed_sync, record_call
from .evidence import is_literal_match
from .lexical import detect_false_friend_signals
from .models import PatternCatalog, PreparedIssue
from .requests import stage1_request_params, stage2_request_params, to_inline_batch_request
from .schemas import normalize_stage1, normalize_stage2
from .utils import append_jsonl, clean_scalar, json_dumps, object_to_dict, stable_custom_id, write_json


def _usage(metadata: dict[str, Any], key: str) -> Any:
    return (metadata.get("usage") or {}).get(key, "")


def flatten_stage1_result(
    prepared: PreparedIssue,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    request_status: str = "succeeded",
    error: str = "",
) -> dict[str, Any]:
    candidates = payload.get("candidates", [])
    return {
        "repository": prepared.repository,
        "issue_number": prepared.issue_number,
        "issue_key": prepared.issue_key,
        "custom_id": prepared.custom_id_stage1,
        "request_status": request_status,
        "error": error,
        "issue_summary": payload.get("issue_summary", ""),
        "issue_activity_type": payload.get("issue_activity_type", ""),
        "issue_challenge_categories": "|".join(payload.get("issue_challenge_categories", [])),
        "context_status": payload.get("context_status", ""),
        "candidates": json.dumps([c["pattern"] for c in candidates], ensure_ascii=False),
        "candidate_details": json.dumps(candidates, ensure_ascii=False),
        "candidate_count": len(candidates),
        "input_truncated": "yes" if prepared.input_truncated else "no",
        "original_char_count": prepared.original_char_count,
        "included_char_count": prepared.included_char_count,
        "message_id": metadata.get("message_id", ""),
        "response_model": metadata.get("response_model", ""),
        "stop_reason": metadata.get("stop_reason", ""),
        "input_tokens": _usage(metadata, "input_tokens"),
        "output_tokens": _usage(metadata, "output_tokens"),
        "total_tokens": _usage(metadata, "total_tokens"),
        "thoughts_tokens": _usage(metadata, "thoughts_tokens"),
        "cached_input_tokens": _usage(metadata, "cached_input_tokens"),
    }


def flatten_stage2_result(
    prepared: PreparedIssue,
    pattern: str,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    request_status: str = "succeeded",
    error: str = "",
) -> dict[str, Any]:
    return {
        "repository": prepared.repository,
        "issue_number": prepared.issue_number,
        "issue_key": prepared.issue_key,
        "custom_id": stable_custom_id("s2", prepared.repository, prepared.issue_number, pattern),
        "pattern": pattern,
        "request_status": request_status,
        "error": error,
        "verdict": payload.get("verdict", ""),
        "mechanism_match": payload.get("mechanism_match", ""),
        "scope_match": payload.get("scope_match", ""),
        "focus_match": payload.get("focus_match", ""),
        "evidence_text": payload.get("evidence_text", ""),
        "evidence_location": "|".join(payload.get("evidence_location", [])),
        "justification": payload.get("justification", ""),
        "adoption_status": payload.get("adoption_status", ""),
        "false_friend_detected": payload.get("false_friend_detected", ""),
        "pattern_challenge_categories": "|".join(
            payload.get("pattern_challenge_categories", [])
        ),
        "confidence": payload.get("confidence", ""),
        "alternative_pattern": payload.get("alternative_pattern", ""),
        "overlap_with": "|".join(payload.get("overlap_with", [])),
        "input_truncated": "yes" if prepared.input_truncated else "no",
        "message_id": metadata.get("message_id", ""),
        "response_model": metadata.get("response_model", ""),
        "stop_reason": metadata.get("stop_reason", ""),
        "input_tokens": _usage(metadata, "input_tokens"),
        "output_tokens": _usage(metadata, "output_tokens"),
        "total_tokens": _usage(metadata, "total_tokens"),
        "thoughts_tokens": _usage(metadata, "thoughts_tokens"),
        "cached_input_tokens": _usage(metadata, "cached_input_tokens"),
    }


def run_stage1_sync(
    client: Any,
    prepared_issues: list[PreparedIssue],
    catalog: PatternCatalog,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    thinking_level: str,
    run_dir: Path,
) -> pd.DataFrame:
    checkpoint = Checkpoint(run_dir / "stage1_checkpoint.json", stage="stage1", run_id=run_dir.name)
    store = ResultStore(run_dir / "stage1_results.csv", stage="stage1")
    completed = store.completed_ids()
    checkpoint.reconcile_completed(sorted(completed))
    print(f"[stage1] total={len(prepared_issues)} completed={len(completed)} pending={len(prepared_issues) - len(completed)}")
    raw_path = run_dir / "stage1_raw.jsonl"
    for idx, prepared in enumerate(prepared_issues, 1):
        # The validated CSV, not the JSON index, determines completion.
        if prepared.custom_id_stage1 in completed:
            print(f"[stage1 sync] {idx}/{len(prepared_issues)} {prepared.issue_key} [skip: resultado persistido]")
            continue

        print(f"[stage1 sync] {idx}/{len(prepared_issues)} {prepared.issue_key}")
        params = stage1_request_params(
            prepared,
            catalog,
            model,
            temperature,
            max_tokens,
            seed,
            thinking_level,
        )
        audit_request(run_dir, prepared, params, 'stage1')
        try:
            response = observed_sync(call_sync, client, params, run_dir, prepared, 'stage1')
            raw = object_to_dict(response)
            append_jsonl(raw_path, {"custom_id": prepared.custom_id_stage1, "response": raw})
            payload, metadata = parse_response_payload(response)
            normalized = normalize_stage1(decode_stage(payload, catalog, prepared, 1), catalog)

            # P2: Detect lexical false friends for candidate patterns.
            candidate_patterns = [c["pattern"] for c in normalized.get("candidates", [])]
            false_friend_signals = detect_false_friend_signals(prepared.artifact_text, candidate_patterns)
            # Annotate candidates with false friend signals for aggregation reporting.
            for candidate in normalized["candidates"]:
                signals_for_pattern = [s for s in false_friend_signals if s.pattern == candidate["pattern"]]
                if signals_for_pattern:
                    candidate["_lexical_false_friend_signals"] = [
                        {"reason": s.reason} for s in signals_for_pattern
                    ]

            # Persist the complete semantic result before marking its index done.
            store.upsert(flatten_stage1_result(prepared, normalized, metadata))
            checkpoint.mark_completed(prepared.custom_id_stage1)
            completed.add(prepared.custom_id_stage1)
        except (CheckpointIntegrityError, OSError):
            # Storage/integrity failures are operational failures of the run,
            # never a semantic "errored" LLM result to be overwritten later.
            raise
        except Exception as exc:
            append_jsonl(
                raw_path,
                {
                    "custom_id": prepared.custom_id_stage1,
                    "error": str(exc),
                    "issue_key": prepared.issue_key,
                },
            )
            store.upsert(flatten_stage1_result(prepared, {}, {}, "errored", str(exc)))
            checkpoint.mark_failed(prepared.custom_id_stage1)
            if is_global_api_error(exc):
                raise
    return store.dataframe.copy()


def stage2_pairs_from_stage1(stage1_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    required = {"repository", "issue_number", "candidates"}
    missing = required - set(stage1_df.columns)
    if missing:
        raise CheckpointIntegrityError(f"Stage 1 sem colunas necessárias para reconstruir pares: {sorted(missing)}")
    successful = stage1_df.get("request_status", pd.Series("succeeded", index=stage1_df.index)) == "succeeded"
    completed = stage1_df.loc[successful].copy()
    if completed.duplicated(["repository", "issue_number"], keep=False).any():
        raise CheckpointIntegrityError("Stage 1 contém issues concluídas duplicadas")
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in completed.to_dict("records"):
        try:
            candidates = json.loads(row.get("candidates") or "[]")
        except json.JSONDecodeError as exc:
            raise CheckpointIntegrityError(f"Stage 1 contém candidates inválido para {row.get('repository')}#{row.get('issue_number')}") from exc
        if not isinstance(candidates, list) or not all(isinstance(pattern, str) and pattern.strip() for pattern in candidates):
            raise CheckpointIntegrityError("Stage 1 contém lista de candidates inválida")
        for pattern in candidates:
            pair = (clean_scalar(row["repository"]), clean_scalar(row["issue_number"]), clean_scalar(pattern))
            if pair in seen:
                raise CheckpointIntegrityError(f"Stage 1 produz par Stage 2 duplicado: {pair}")
            seen.add(pair)
            pairs.append(pair)
    return pairs


def pipeline_integrity_report(
    prepared_issues: list[PreparedIssue], stage1_df: pd.DataFrame, stage2_df: pd.DataFrame
) -> dict[str, Any]:
    """Compare the explicit Stage 2 pair set with valid persisted results."""
    input_keys = {(item.repository, item.issue_number) for item in prepared_issues}
    successful_stage1 = stage1_df.loc[
        stage1_df.get("request_status", pd.Series("succeeded", index=stage1_df.index)) == "succeeded"
    ] if not stage1_df.empty else pd.DataFrame()
    completed_stage1 = {
        (clean_scalar(row["repository"]), clean_scalar(row["issue_number"]))
        for row in successful_stage1.to_dict("records")
    }
    expected = set(stage2_pairs_from_stage1(stage1_df)) if not stage1_df.empty else set()
    if stage2_df.empty:
        completed, failed, duplicates = set(), set(), []
    else:
        key_columns = ["repository", "issue_number", "pattern"]
        missing = set(key_columns) - set(stage2_df.columns)
        if missing:
            raise CheckpointIntegrityError(f"Stage 2 sem colunas-chave: {sorted(missing)}")
        statuses = stage2_df.get("request_status", pd.Series("succeeded", index=stage2_df.index))
        succeeded = stage2_df.loc[statuses == "succeeded"]
        duplicated = succeeded.duplicated(key_columns, keep=False)
        duplicates = succeeded.loc[duplicated, key_columns].drop_duplicates().to_dict("records")
        if duplicates:
            raise CheckpointIntegrityError(f"Stage 2 contém pares concluídos duplicados: {duplicates[:10]}")
        completed = {
            tuple(clean_scalar(row[column]) for column in key_columns)
            for row in succeeded.to_dict("records")
        }
        failed = {
            tuple(clean_scalar(row[column]) for column in key_columns)
            for row in stage2_df.loc[statuses != "succeeded"].to_dict("records")
        }
    missing_pairs = expected - completed
    extra_pairs = completed - expected
    report = {
        "total_stage1_issues": len(input_keys),
        "completed_stage1_issues": len(completed_stage1 & input_keys),
        "issues_with_candidates": sum(1 for repository, issue_number in completed_stage1 if any(pair[:2] == (repository, issue_number) for pair in expected)),
        "total_candidates": len(expected),
        "expected_stage2_pairs": len(expected),
        "completed_stage2_pairs": len(completed & expected),
        "missing_stage2_pairs": [dict(zip(["repository", "issue_number", "pattern"], pair)) for pair in sorted(missing_pairs)],
        "extra_stage2_pairs": [dict(zip(["repository", "issue_number", "pattern"], pair)) for pair in sorted(extra_pairs)],
        "duplicate_stage2_pairs": duplicates,
        "failed_stage2_pairs": len(failed & expected),
    }
    report["stage1_complete"] = report["completed_stage1_issues"] == report["total_stage1_issues"]
    report["stage2_complete"] = not missing_pairs and not extra_pairs and not duplicates
    report["status"] = "OK" if report["stage1_complete"] and report["stage2_complete"] else "FAILED"
    return report


def run_stage2_sync(
    client: Any,
    prepared_by_key: dict[tuple[str, str], PreparedIssue],
    pairs: list[tuple[str, str, str]],
    catalog: PatternCatalog,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    thinking_level: str,
    run_dir: Path,
) -> pd.DataFrame:
    select_stage2 = prepare_stage2_contexts(prepared_by_key, catalog, run_dir)
    checkpoint = Checkpoint(run_dir / "stage2_checkpoint.json", stage="stage2", run_id=run_dir.name)
    store = ResultStore(run_dir / "stage2_results.csv", stage="stage2")
    completed = store.completed_ids()
    expected_ids = {stable_custom_id("s2", repository, issue_number, pattern) for repository, issue_number, pattern in pairs}
    unknown = store.completed_ids() - expected_ids
    if unknown:
        print(f"[stage2] aviso: {len(unknown)} resultados concluídos não pertencem aos pares atuais")
    checkpoint.reconcile_completed(sorted(completed))
    print(f"[stage2] expected={len(pairs)} completed={len(expected_ids & completed)} pending={len(expected_ids - completed)}")
    raw_path = run_dir / "stage2_raw.jsonl"
    for idx, (repository, issue_number, pattern) in enumerate(pairs, 1):
        prepared = prepared_by_key[(repository, issue_number)]
        custom_id = stable_custom_id("s2", repository, issue_number, pattern)

        if custom_id in completed:
            print(f"[stage2 sync] {idx}/{len(pairs)} {prepared.issue_key} / {pattern} [skip: resultado persistido]")
            continue

        print(f"[stage2 sync] {idx}/{len(pairs)} {prepared.issue_key} / {pattern}")
        prepared = select_stage2(prepared, pattern)
        params = stage2_request_params(
            prepared,
            pattern,
            catalog,
            model,
            temperature,
            max_tokens,
            seed,
            thinking_level,
        )
        audit_request(run_dir, prepared, params, 'stage2', pattern)
        try:
            response = observed_sync(call_sync, client, params, run_dir, prepared, 'stage2', pattern)
            raw = object_to_dict(response)
            append_jsonl(raw_path, {"custom_id": custom_id, "response": raw})
            payload, metadata = parse_response_payload(response)
            normalized = normalize_stage2(decode_stage(payload, catalog, prepared, 2), catalog)

            # P5: Verify evidence_text is a literal substring of the artifact.
            evidence = normalized.get("evidence_text", "")
            if evidence and not is_literal_match(evidence, prepared.artifact_text):
                print(
                    f"[warning P5] {custom_id}: evidence_text não é substring literal: {evidence[:80]!r}",
                    file=sys.stderr,
                )

            # P6: If artifact is a PR and evidence_location contains "body", replace with "pull_request_description".
            if prepared.artifact_type and prepared.artifact_type.lower() == "pull_request":
                locations = normalized.get("evidence_location", [])
                normalized["evidence_location"] = [
                    "pull_request_description" if loc == "body" else loc
                    for loc in locations
                ]

            # P1: Override confidence with deterministic computation based on evidence and validations.
            computed_confidence = compute_stage2_confidence(normalized)
            normalized["confidence"] = computed_confidence

            store.upsert(flatten_stage2_result(prepared, pattern, normalized, metadata))
            checkpoint.mark_completed(custom_id)
            completed.add(custom_id)
        except (CheckpointIntegrityError, OSError):
            raise
        except Exception as exc:
            append_jsonl(
                raw_path,
                {
                    "custom_id": custom_id,
                    "error": str(exc),
                    "issue_key": prepared.issue_key,
                    "pattern": pattern,
                },
            )
            store.upsert(flatten_stage2_result(prepared, pattern, {}, {}, "errored", str(exc)))
            checkpoint.mark_failed(custom_id)
            if is_global_api_error(exc):
                raise
    return store.dataframe.copy()


def _record_failed_stage1_chunk(
    rows: list[dict[str, Any]],
    manifest: dict[str, PreparedIssue],
    request_chunk: list[dict[str, Any]],
    error: str,
    run_dir: Path,
    model: str,
) -> None:
    for request in request_chunk:
        custom_id = clean_scalar((request.get("metadata") or {}).get("custom_id"))
        prepared = manifest[custom_id]
        record_call(run_dir, prepared, {**request, 'model':model}, stage='stage1', error=error,
                    timing_source='batch_submission_or_polling_failure_unknown_usage')
        rows.append(flatten_stage1_result(prepared, {}, {}, "errored", error))


def _record_failed_stage2_chunk(
    rows: list[dict[str, Any]],
    manifest: dict[str, tuple[PreparedIssue, str]],
    request_chunk: list[dict[str, Any]],
    error: str,
    run_dir: Path,
    model: str,
) -> None:
    for request in request_chunk:
        custom_id = clean_scalar((request.get("metadata") or {}).get("custom_id"))
        prepared, pattern = manifest[custom_id]
        record_call(run_dir, prepared, {**request, 'model':model}, stage='stage2', pattern=pattern,
                    error=error, timing_source='batch_submission_or_polling_failure_unknown_usage')
        rows.append(flatten_stage2_result(prepared, pattern, {}, {}, "errored", error))


class _DurableRows(list[dict[str, Any]]):
    """Compatibility collector that persists every batch response immediately."""

    def __init__(self, store: ResultStore, checkpoint: Checkpoint) -> None:
        super().__init__()
        self.store = store
        self.checkpoint = checkpoint

    def append(self, row: dict[str, Any]) -> None:
        self.store.upsert(row)
        custom_id = clean_scalar(row.get("custom_id"))
        if custom_id:
            if row.get("request_status") == "succeeded":
                self.checkpoint.mark_completed(custom_id)
            else:
                self.checkpoint.mark_failed(custom_id)
        super().append(row)


def run_stage1_batch(
    client: Any,
    prepared_issues: list[PreparedIssue],
    catalog: PatternCatalog,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    thinking_level: str,
    batch_size: int,
    batch_max_bytes: int,
    poll_seconds: int,
    run_dir: Path,
) -> pd.DataFrame:
    checkpoint = Checkpoint(run_dir / "stage1_checkpoint.json", stage="stage1", run_id=run_dir.name)
    store = ResultStore(run_dir / "stage1_results.csv", stage="stage1")
    completed = store.completed_ids()
    checkpoint.reconcile_completed(sorted(completed))
    pending_issues = [item for item in prepared_issues if item.custom_id_stage1 not in completed]
    print(f"[stage1] total={len(prepared_issues)} completed={len(completed)} pending={len(pending_issues)}")
    manifest = {item.custom_id_stage1: item for item in pending_issues}
    rows: list[dict[str, Any]] = _DurableRows(store, checkpoint)
    raw_path = run_dir / "stage1_raw.jsonl"
    batch_meta: list[dict[str, Any]] = []

    requests = []
    for item in pending_issues:
        params = stage1_request_params(
            item,
            catalog,
            model,
            temperature,
            max_tokens,
            seed,
            thinking_level,
        )
        audit_request(run_dir, item, params, 'stage1')
        requests.append(to_inline_batch_request(item.custom_id_stage1, params))

    chunks = list(
        chunk_inline_requests(requests, max_items=batch_size, max_bytes=batch_max_bytes)
    )
    for batch_index, request_chunk in enumerate(chunks, 1):
        display_name = f"stage1-{run_dir.name}-{batch_index}"
        print(f"[stage1 batch] enviando lote {batch_index}/{len(chunks)} com {len(request_chunk)} requests")
        try:
            batch_name, created, final_batch = resumable_batch(
                client, model=model, requests=request_chunk, display_name=display_name,
                run_dir=run_dir, stage='stage1', create=call_batch_create,
                wait=wait_for_batch, poll_seconds=poll_seconds)
            batch_meta.append(
                {
                    "batch_index": batch_index,
                    "created": object_to_dict(created),
                    "final": object_to_dict(final_batch),
                }
            )
            write_json(run_dir / "stage1_batches.json", batch_meta)
        except (CheckpointIntegrityError, OSError):
            raise
        except Exception as exc:
            error = f"Falha no job batch: {exc}"
            append_jsonl(raw_path, {"batch_index": batch_index, "error": error})
            _record_failed_stage1_chunk(rows, manifest, request_chunk, error, run_dir, model)
            if is_global_api_error(exc):
                raise
            continue

        state = batch_state_name(final_batch)
        if state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            error = f"Batch terminou com state={state}: {json_dumps(object_to_dict(getattr(final_batch, 'error', {})))}"
            _record_failed_stage1_chunk(rows, manifest, request_chunk, error, run_dir, model)
            mark_batch_consumed(run_dir, 'stage1', batch_name)
            continue

        expected_ids = [clean_scalar((req.get("metadata") or {}).get("custom_id")) for req in request_chunk]
        returned_ids: set[str] = set()
        responses = batch_inlined_responses(final_batch)
        original_order = created.get('_request_order', expected_ids)
        for response_index, inline_response in enumerate(responses):
            # Reused jobs may cover a superset of pending IDs. Positional fallback
            # is safe only against the original complete job order.
            fallback_id = original_order[response_index] if len(responses) == len(original_order) else ""
            custom_id = batch_custom_id(inline_response) or fallback_id
            append_jsonl(
                raw_path,
                {
                    "batch_name": clean_scalar(getattr(final_batch, "name", "")),
                    "custom_id": custom_id,
                    "inline_response": object_to_dict(inline_response),
                },
            )
            if not custom_id or custom_id not in expected_ids or custom_id in returned_ids:
                continue
            returned_ids.add(custom_id)
            prepared = manifest[custom_id]
            api_error = batch_error_text(inline_response)
            response = batch_generated_response(inline_response)
            request = next(r for r in request_chunk if r['metadata']['custom_id'] == custom_id)
            record_call(run_dir, prepared, {**request, 'model': model}, stage='stage1',
                        response=response, error=api_error, batch_name=batch_name,
                        timing_source='batch_per_request_unavailable')
            if api_error or response is None:
                rows.append(
                    flatten_stage1_result(
                        prepared,
                        {},
                        {},
                        "errored",
                        api_error or "Batch sem GenerateContentResponse",
                    )
                )
                continue
            try:
                payload, metadata = parse_response_payload(response)
                normalized = normalize_stage1(decode_stage(payload, catalog, prepared, 1), catalog)

                # P2: Detect lexical false friends for candidate patterns (batch mode).
                candidate_patterns = [c["pattern"] for c in normalized.get("candidates", [])]
                false_friend_signals = detect_false_friend_signals(prepared.artifact_text, candidate_patterns)
                for candidate in normalized["candidates"]:
                    signals_for_pattern = [s for s in false_friend_signals if s.pattern == candidate["pattern"]]
                    if signals_for_pattern:
                        candidate["_lexical_false_friend_signals"] = [
                            {"reason": s.reason} for s in signals_for_pattern
                        ]

                rows.append(flatten_stage1_result(prepared, normalized, metadata))
            except (CheckpointIntegrityError, OSError):
                raise
            except Exception as exc:
                rows.append(flatten_stage1_result(prepared, {}, {}, "errored", str(exc)))

        for custom_id in expected_ids:
            if custom_id not in returned_ids:
                prepared = manifest[custom_id]
                request = next(r for r in request_chunk if r['metadata']['custom_id'] == custom_id)
                record_call(run_dir, prepared, {**request,'model':model}, stage='stage1',
                            error='Missing batch response; usage unknown', batch_name=batch_name,
                            timing_source='batch_per_request_unavailable')
                rows.append(
                    flatten_stage1_result(
                        prepared,
                        {},
                        {},
                        "errored",
                        "Resposta ausente no resultado inline do Batch API",
                    )
                )
        mark_batch_consumed(run_dir, 'stage1', batch_name)
    return store.dataframe.copy()


def run_stage2_batch(
    client: Any,
    prepared_by_key: dict[tuple[str, str], PreparedIssue],
    pairs: list[tuple[str, str, str]],
    catalog: PatternCatalog,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    thinking_level: str,
    batch_size: int,
    batch_max_bytes: int,
    poll_seconds: int,
    run_dir: Path,
) -> pd.DataFrame:
    select_stage2 = prepare_stage2_contexts(prepared_by_key, catalog, run_dir)
    checkpoint = Checkpoint(run_dir / "stage2_checkpoint.json", stage="stage2", run_id=run_dir.name)
    store = ResultStore(run_dir / "stage2_results.csv", stage="stage2")
    completed = store.completed_ids()
    checkpoint.reconcile_completed(sorted(completed))
    manifest: dict[str, tuple[PreparedIssue, str]] = {}
    requests: list[dict[str, Any]] = []
    for repository, issue_number, pattern in pairs:
        prepared = prepared_by_key[(repository, issue_number)]
        custom_id = stable_custom_id("s2", repository, issue_number, pattern)
        if custom_id in completed:
            continue
        prepared = select_stage2(prepared, pattern)
        manifest[custom_id] = (prepared, pattern)
        params = stage2_request_params(
            prepared,
            pattern,
            catalog,
            model,
            temperature,
            max_tokens,
            seed,
            thinking_level,
        )
        audit_request(run_dir, prepared, params, 'stage2', pattern)
        requests.append(to_inline_batch_request(custom_id, params))

    print(f"[stage2] expected={len(pairs)} completed={len(completed)} pending={len(requests)}")
    rows: list[dict[str, Any]] = _DurableRows(store, checkpoint)
    raw_path = run_dir / "stage2_raw.jsonl"
    batch_meta: list[dict[str, Any]] = []
    chunks = list(
        chunk_inline_requests(requests, max_items=batch_size, max_bytes=batch_max_bytes)
    )

    for batch_index, request_chunk in enumerate(chunks, 1):
        display_name = f"stage2-{run_dir.name}-{batch_index}"
        print(f"[stage2 batch] enviando lote {batch_index}/{len(chunks)} com {len(request_chunk)} requests")
        try:
            batch_name, created, final_batch = resumable_batch(
                client, model=model, requests=request_chunk, display_name=display_name,
                run_dir=run_dir, stage='stage2', create=call_batch_create,
                wait=wait_for_batch, poll_seconds=poll_seconds)
            batch_meta.append(
                {
                    "batch_index": batch_index,
                    "created": object_to_dict(created),
                    "final": object_to_dict(final_batch),
                }
            )
            write_json(run_dir / "stage2_batches.json", batch_meta)
        except (CheckpointIntegrityError, OSError):
            raise
        except Exception as exc:
            error = f"Falha no job batch: {exc}"
            append_jsonl(raw_path, {"batch_index": batch_index, "error": error})
            _record_failed_stage2_chunk(rows, manifest, request_chunk, error, run_dir, model)
            if is_global_api_error(exc):
                raise
            continue

        state = batch_state_name(final_batch)
        if state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            error = f"Batch terminou com state={state}: {json_dumps(object_to_dict(getattr(final_batch, 'error', {})))}"
            _record_failed_stage2_chunk(rows, manifest, request_chunk, error, run_dir, model)
            mark_batch_consumed(run_dir, 'stage2', batch_name)
            continue

        expected_ids = [clean_scalar((req.get("metadata") or {}).get("custom_id")) for req in request_chunk]
        returned_ids: set[str] = set()
        responses = batch_inlined_responses(final_batch)
        original_order = created.get('_request_order', expected_ids)
        for response_index, inline_response in enumerate(responses):
            # Reused jobs may cover a superset of pending IDs. Positional fallback
            # is safe only against the original complete job order.
            fallback_id = original_order[response_index] if len(responses) == len(original_order) else ""
            custom_id = batch_custom_id(inline_response) or fallback_id
            append_jsonl(
                raw_path,
                {
                    "batch_name": clean_scalar(getattr(final_batch, "name", "")),
                    "custom_id": custom_id,
                    "inline_response": object_to_dict(inline_response),
                },
            )
            if not custom_id or custom_id not in expected_ids or custom_id in returned_ids:
                continue
            returned_ids.add(custom_id)
            prepared, pattern = manifest[custom_id]
            api_error = batch_error_text(inline_response)
            response = batch_generated_response(inline_response)
            request = next(r for r in request_chunk if r['metadata']['custom_id'] == custom_id)
            record_call(run_dir, prepared, {**request, 'model': model}, stage='stage2', pattern=pattern,
                        response=response, error=api_error, batch_name=batch_name,
                        timing_source='batch_per_request_unavailable')
            if api_error or response is None:
                rows.append(
                    flatten_stage2_result(
                        prepared,
                        pattern,
                        {},
                        {},
                        "errored",
                        api_error or "Batch sem GenerateContentResponse",
                    )
                )
                continue
            try:
                payload, metadata = parse_response_payload(response)
                normalized = normalize_stage2(decode_stage(payload, catalog, prepared, 2), catalog)

                # P5: Verify evidence_text is a literal substring of the artifact.
                evidence = normalized.get("evidence_text", "")
                if evidence and not is_literal_match(evidence, prepared.artifact_text):
                    print(
                        f"[warning P5] {custom_id}: evidence_text não é substring literal: {evidence[:80]!r}",
                        file=sys.stderr,
                    )

                # P6: If artifact is a PR and evidence_location contains "body", replace with "pull_request_description".
                if prepared.artifact_type and prepared.artifact_type.lower() == "pull_request":
                    locations = normalized.get("evidence_location", [])
                    normalized["evidence_location"] = [
                        "pull_request_description" if loc == "body" else loc
                        for loc in locations
                    ]

                # P1: Override confidence with deterministic computation based on evidence and validations.
                computed_confidence = compute_stage2_confidence(normalized)
                normalized["confidence"] = computed_confidence

                rows.append(flatten_stage2_result(prepared, pattern, normalized, metadata))
            except (CheckpointIntegrityError, OSError):
                raise
            except Exception as exc:
                rows.append(flatten_stage2_result(prepared, pattern, {}, {}, "errored", str(exc)))

        for custom_id in expected_ids:
            if custom_id not in returned_ids:
                prepared, pattern = manifest[custom_id]
                request = next(r for r in request_chunk if r['metadata']['custom_id'] == custom_id)
                record_call(run_dir, prepared, {**request,'model':model}, stage='stage2',pattern=pattern,
                            error='Missing batch response; usage unknown', batch_name=batch_name,
                            timing_source='batch_per_request_unavailable')
                rows.append(
                    flatten_stage2_result(
                        prepared,
                        pattern,
                        {},
                        {},
                        "errored",
                        "Resposta ausente no resultado inline do Batch API",
                    )
                )
        mark_batch_consumed(run_dir, 'stage2', batch_name)
    return store.dataframe.copy()
