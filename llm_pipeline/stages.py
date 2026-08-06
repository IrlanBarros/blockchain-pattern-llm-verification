"""Execução dos Stages 1 e 2 com Gemini em modo síncrono ou batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .client import (
    batch_custom_id,
    batch_error_text,
    batch_generated_response,
    batch_inlined_responses,
    batch_state_name,
    call_batch_create,
    call_sync,
    chunk_inline_requests,
    parse_response_payload,
    wait_for_batch,
)
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
    rows: list[dict[str, Any]] = []
    raw_path = run_dir / "stage1_raw.jsonl"
    for idx, prepared in enumerate(prepared_issues, 1):
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
        try:
            response = call_sync(client, params)
            raw = object_to_dict(response)
            append_jsonl(raw_path, {"custom_id": prepared.custom_id_stage1, "response": raw})
            payload, metadata = parse_response_payload(response)
            normalized = normalize_stage1(payload, catalog)
            rows.append(flatten_stage1_result(prepared, normalized, metadata))
        except Exception as exc:
            append_jsonl(
                raw_path,
                {
                    "custom_id": prepared.custom_id_stage1,
                    "error": str(exc),
                    "issue_key": prepared.issue_key,
                },
            )
            rows.append(flatten_stage1_result(prepared, {}, {}, "errored", str(exc)))
    return pd.DataFrame(rows)


def stage2_pairs_from_stage1(stage1_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for row in stage1_df.to_dict("records"):
        if row.get("request_status") != "succeeded":
            continue
        try:
            candidates = json.loads(row.get("candidates") or "[]")
        except json.JSONDecodeError:
            continue
        for pattern in candidates:
            pairs.append(
                (clean_scalar(row["repository"]), clean_scalar(row["issue_number"]), pattern)
            )
    return pairs


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
    rows: list[dict[str, Any]] = []
    raw_path = run_dir / "stage2_raw.jsonl"
    for idx, (repository, issue_number, pattern) in enumerate(pairs, 1):
        prepared = prepared_by_key[(repository, issue_number)]
        print(f"[stage2 sync] {idx}/{len(pairs)} {prepared.issue_key} / {pattern}")
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
        custom_id = stable_custom_id("s2", repository, issue_number, pattern)
        try:
            response = call_sync(client, params)
            raw = object_to_dict(response)
            append_jsonl(raw_path, {"custom_id": custom_id, "response": raw})
            payload, metadata = parse_response_payload(response)
            normalized = normalize_stage2(payload, catalog)
            rows.append(flatten_stage2_result(prepared, pattern, normalized, metadata))
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
            rows.append(flatten_stage2_result(prepared, pattern, {}, {}, "errored", str(exc)))
    return pd.DataFrame(rows)


def _record_failed_stage1_chunk(
    rows: list[dict[str, Any]],
    manifest: dict[str, PreparedIssue],
    request_chunk: list[dict[str, Any]],
    error: str,
) -> None:
    for request in request_chunk:
        custom_id = clean_scalar((request.get("metadata") or {}).get("custom_id"))
        prepared = manifest[custom_id]
        rows.append(flatten_stage1_result(prepared, {}, {}, "errored", error))


def _record_failed_stage2_chunk(
    rows: list[dict[str, Any]],
    manifest: dict[str, tuple[PreparedIssue, str]],
    request_chunk: list[dict[str, Any]],
    error: str,
) -> None:
    for request in request_chunk:
        custom_id = clean_scalar((request.get("metadata") or {}).get("custom_id"))
        prepared, pattern = manifest[custom_id]
        rows.append(flatten_stage2_result(prepared, pattern, {}, {}, "errored", error))


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
    manifest = {item.custom_id_stage1: item for item in prepared_issues}
    rows: list[dict[str, Any]] = []
    raw_path = run_dir / "stage1_raw.jsonl"
    batch_meta: list[dict[str, Any]] = []

    requests = []
    for item in prepared_issues:
        params = stage1_request_params(
            item,
            catalog,
            model,
            temperature,
            max_tokens,
            seed,
            thinking_level,
        )
        requests.append(to_inline_batch_request(item.custom_id_stage1, params))

    chunks = list(
        chunk_inline_requests(requests, max_items=batch_size, max_bytes=batch_max_bytes)
    )
    for batch_index, request_chunk in enumerate(chunks, 1):
        display_name = f"stage1-{run_dir.name}-{batch_index}"
        print(f"[stage1 batch] enviando lote {batch_index}/{len(chunks)} com {len(request_chunk)} requests")
        try:
            created = call_batch_create(
                client,
                model=model,
                requests=request_chunk,
                display_name=display_name,
            )
            batch_name = clean_scalar(getattr(created, "name", ""))
            if not batch_name:
                raise ValueError("Batch criado sem name")
            final_batch = wait_for_batch(client, batch_name, poll_seconds)
            batch_meta.append(
                {
                    "batch_index": batch_index,
                    "created": object_to_dict(created),
                    "final": object_to_dict(final_batch),
                }
            )
            write_json(run_dir / "stage1_batches.json", batch_meta)
        except Exception as exc:
            error = f"Falha no job batch: {exc}"
            append_jsonl(raw_path, {"batch_index": batch_index, "error": error})
            _record_failed_stage1_chunk(rows, manifest, request_chunk, error)
            continue

        state = batch_state_name(final_batch)
        if state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            error = f"Batch terminou com state={state}: {json_dumps(object_to_dict(getattr(final_batch, 'error', {})))}"
            _record_failed_stage1_chunk(rows, manifest, request_chunk, error)
            continue

        expected_ids = [clean_scalar((req.get("metadata") or {}).get("custom_id")) for req in request_chunk]
        returned_ids: set[str] = set()
        responses = batch_inlined_responses(final_batch)
        for response_index, inline_response in enumerate(responses):
            fallback_id = expected_ids[response_index] if response_index < len(expected_ids) else ""
            custom_id = batch_custom_id(inline_response) or fallback_id
            append_jsonl(
                raw_path,
                {
                    "batch_name": clean_scalar(getattr(final_batch, "name", "")),
                    "custom_id": custom_id,
                    "inline_response": object_to_dict(inline_response),
                },
            )
            if not custom_id or custom_id not in manifest:
                continue
            returned_ids.add(custom_id)
            prepared = manifest[custom_id]
            api_error = batch_error_text(inline_response)
            response = batch_generated_response(inline_response)
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
                normalized = normalize_stage1(payload, catalog)
                rows.append(flatten_stage1_result(prepared, normalized, metadata))
            except Exception as exc:
                rows.append(flatten_stage1_result(prepared, {}, {}, "errored", str(exc)))

        for custom_id in expected_ids:
            if custom_id not in returned_ids:
                prepared = manifest[custom_id]
                rows.append(
                    flatten_stage1_result(
                        prepared,
                        {},
                        {},
                        "errored",
                        "Resposta ausente no resultado inline do Batch API",
                    )
                )
    return pd.DataFrame(rows)


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
    manifest: dict[str, tuple[PreparedIssue, str]] = {}
    requests: list[dict[str, Any]] = []
    for repository, issue_number, pattern in pairs:
        prepared = prepared_by_key[(repository, issue_number)]
        custom_id = stable_custom_id("s2", repository, issue_number, pattern)
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
        requests.append(to_inline_batch_request(custom_id, params))

    rows: list[dict[str, Any]] = []
    raw_path = run_dir / "stage2_raw.jsonl"
    batch_meta: list[dict[str, Any]] = []
    chunks = list(
        chunk_inline_requests(requests, max_items=batch_size, max_bytes=batch_max_bytes)
    )

    for batch_index, request_chunk in enumerate(chunks, 1):
        display_name = f"stage2-{run_dir.name}-{batch_index}"
        print(f"[stage2 batch] enviando lote {batch_index}/{len(chunks)} com {len(request_chunk)} requests")
        try:
            created = call_batch_create(
                client,
                model=model,
                requests=request_chunk,
                display_name=display_name,
            )
            batch_name = clean_scalar(getattr(created, "name", ""))
            if not batch_name:
                raise ValueError("Batch criado sem name")
            final_batch = wait_for_batch(client, batch_name, poll_seconds)
            batch_meta.append(
                {
                    "batch_index": batch_index,
                    "created": object_to_dict(created),
                    "final": object_to_dict(final_batch),
                }
            )
            write_json(run_dir / "stage2_batches.json", batch_meta)
        except Exception as exc:
            error = f"Falha no job batch: {exc}"
            append_jsonl(raw_path, {"batch_index": batch_index, "error": error})
            _record_failed_stage2_chunk(rows, manifest, request_chunk, error)
            continue

        state = batch_state_name(final_batch)
        if state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            error = f"Batch terminou com state={state}: {json_dumps(object_to_dict(getattr(final_batch, 'error', {})))}"
            _record_failed_stage2_chunk(rows, manifest, request_chunk, error)
            continue

        expected_ids = [clean_scalar((req.get("metadata") or {}).get("custom_id")) for req in request_chunk]
        returned_ids: set[str] = set()
        responses = batch_inlined_responses(final_batch)
        for response_index, inline_response in enumerate(responses):
            fallback_id = expected_ids[response_index] if response_index < len(expected_ids) else ""
            custom_id = batch_custom_id(inline_response) or fallback_id
            append_jsonl(
                raw_path,
                {
                    "batch_name": clean_scalar(getattr(final_batch, "name", "")),
                    "custom_id": custom_id,
                    "inline_response": object_to_dict(inline_response),
                },
            )
            if not custom_id or custom_id not in manifest:
                continue
            returned_ids.add(custom_id)
            prepared, pattern = manifest[custom_id]
            api_error = batch_error_text(inline_response)
            response = batch_generated_response(inline_response)
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
                normalized = normalize_stage2(payload, catalog)
                rows.append(flatten_stage2_result(prepared, pattern, normalized, metadata))
            except Exception as exc:
                rows.append(flatten_stage2_result(prepared, pattern, {}, {}, "errored", str(exc)))

        for custom_id in expected_ids:
            if custom_id not in returned_ids:
                prepared, pattern = manifest[custom_id]
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
    return pd.DataFrame(rows)
