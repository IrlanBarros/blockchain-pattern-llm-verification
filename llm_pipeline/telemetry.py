"""Per-attempt usage accounting; estimates never masquerade as API measurements."""
from __future__ import annotations

from contextvars import ContextVar
import json
import math
from pathlib import Path
import time
import uuid

import pandas as pd

from .utils import append_jsonl, object_to_dict, utc_now_iso, write_json

ATTEMPT_OBSERVER = ContextVar('token_attempt_observer', default=None)


def prompt_text(params):
    config = params.get('config', {})
    return (config.get('system_instruction', '') + '\n' +
            json.dumps(config.get('response_json_schema', {}), ensure_ascii=False, separators=(',', ':')) + '\n' +
            '\n'.join(part.get('text', '') for content in params.get('contents', []) for part in content.get('parts', [])))


def estimate_tokens(text):
    """Rough chars/4 estimate, explicitly uncalibrated for Gemini; not a hard token bound."""
    return math.ceil(len(text) / 4)


def record_call(run_dir, prepared, params, *, stage, response=None, error=None,
                elapsed_seconds=None, pattern='', batch_name='', candidate_count=None,
                timing_source='per_attempt_wall_clock'):
    cfg = prepared.optimization
    if cfg and not cfg.token_telemetry_enabled:
        return
    raw = object_to_dict(response) if response is not None else {}
    usage = (raw.get('usage_metadata') or raw.get('usageMetadata') or {}) if isinstance(raw, dict) else {}
    def number(snake, camel):
        value = usage.get(snake, usage.get(camel))
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    text = prompt_text(params)
    if candidate_count is None and stage == 'stage1' and isinstance(raw, dict):
        try:
            payload = raw.get('parsed')
            if not isinstance(payload, dict):
                from .client import _response_text, _parse_json_text
                payload = _parse_json_text(_response_text(response))
            candidates = payload.get('candidates', payload.get('c'))
            if isinstance(candidates, list):
                candidate_count = len(candidates)
        except (ValueError, TypeError, AttributeError):
            pass  # Invalid output still has billable usage; it is not zero candidates.
    row = {'call_id': uuid.uuid4().hex, 'timestamp': utc_now_iso(), 'stage': stage,
           'repository': prepared.repository, 'issue_number': prepared.issue_number, 'pattern': pattern,
           'model': params.get('model'), 'response_model': raw.get('model_version', raw.get('modelVersion')) if isinstance(raw, dict) else None,
           'input_tokens': number('prompt_token_count','promptTokenCount'),
           'output_tokens': number('candidates_token_count','candidatesTokenCount'),
           'total_tokens': number('total_token_count','totalTokenCount'),
           'thoughts_tokens': number('thoughts_token_count','thoughtsTokenCount'),
           'cached_input_tokens': number('cached_content_token_count','cachedContentTokenCount'),
           'candidate_count': candidate_count if candidate_count is not None else (1 if stage == 'stage2' else None),
           'retrieval_candidate_count': prepared.retrieval.get('retrieval_candidate_count'),
           'prompt_chars': len(text), 'estimated_input_tokens': estimate_tokens(text),
           'estimate_method': 'uncalibrated_chars_div_4_including_schema',
           'elapsed_seconds': elapsed_seconds, 'timing_source': timing_source,
           'batch_name': batch_name, 'transport_error': str(error) if error else None}
    # A batch response replayed after interruption must not be billed twice in the ledger.
    if batch_name:
        identity = (batch_name, stage, prepared.repository, prepared.issue_number, pattern)
        for previous in read_calls(run_dir):
            if identity == tuple(previous.get(k) for k in ('batch_name','stage','repository','issue_number','pattern')):
                return
    append_jsonl(Path(run_dir) / 'token_calls.jsonl', row)


def observed_sync(call, client, params, run_dir, prepared, stage, pattern=''):
    recorded = False
    def observer(response, error, elapsed):
        nonlocal recorded
        record_call(run_dir, prepared, params, stage=stage, response=response, error=error,
                    elapsed_seconds=elapsed, pattern=pattern)
        recorded = True
    token = ATTEMPT_OBSERVER.set(observer)
    start = time.perf_counter()
    try:
        response = call(client, params)
        if not recorded:  # Also supports injected/mock backends.
            observer(response, None, time.perf_counter() - start)
        return response
    except BaseException as exc:
        if not recorded:
            observer(None, exc, time.perf_counter() - start)
        raise
    finally:
        ATTEMPT_OBSERVER.reset(token)


def read_calls(run_dir):
    path = Path(run_dir) / 'token_calls.jsonl'
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_calls(run_dir, config=None, baseline_total=None):
    rows = read_calls(run_dir)
    summary = {'calls': len(rows), 'usage_complete': bool(rows) and all(r['total_tokens'] is not None for r in rows)}
    for key in ('input_tokens','output_tokens','total_tokens','thoughts_tokens','cached_input_tokens'):
        values = [r[key] for r in rows if r.get(key) is not None]
        summary['total_' + key.removeprefix('total_')] = sum(values) if values else None
        summary[key + '_measured_calls'] = len(values)
    for stage in ('stage1','stage2'):
        values = [r['input_tokens'] for r in rows if r['stage']==stage and r['input_tokens'] is not None]
        for label, func in [('mean',lambda s:s.mean()),('median',lambda s:s.median()),('p95',lambda s:s.quantile(.95))]:
            summary[f'{label}_input_tokens_{stage}'] = float(func(pd.Series(values))) if values else None
    path = Path(run_dir) / 'stage1_results.csv'
    summary['mean_candidates_per_issue'] = None
    if path.exists():
        frame = pd.read_csv(path)
        values = frame.loc[frame.request_status == 'succeeded', 'candidate_count']
        summary['mean_candidates_per_issue'] = float(values.mean()) if len(values) else None
    summary['estimated_cost_if_api'] = None
    if config and config.input_price_per_million is not None and config.output_price_per_million is not None and summary['usage_complete']:
        # Explicit flat rates; no claim about actual Gemini pricing or cache/batch discounts.
        summary['estimated_cost_if_api'] = ((summary['total_input_tokens'] or 0) * config.input_price_per_million +
            ((summary['total_output_tokens'] or 0) + (summary['total_thoughts_tokens'] or 0)) * config.output_price_per_million) / 1e6
    summary['tokens_saved_vs_baseline'] = None
    summary['percentage_saved_vs_baseline'] = None
    if baseline_total and summary['usage_complete']:
        saved = baseline_total - summary['total_tokens']
        summary.update(tokens_saved_vs_baseline=saved, percentage_saved_vs_baseline=100*saved/baseline_total)
    summary['measured_call_elapsed_seconds'] = sum(r['elapsed_seconds'] for r in rows if r['elapsed_seconds'] is not None)
    sessions = Path(run_dir) / 'execution_sessions.jsonl'
    summary['run_wall_seconds'] = sum(json.loads(line)['elapsed_seconds'] for line in sessions.read_text().splitlines() if line.strip()) if sessions.exists() else None
    summary['timing_note'] = 'Sync: sum of attempt latency, excluding backoff. Batch: per-request latency unavailable.'
    write_json(Path(run_dir) / 'token_summary.json', summary)
    return summary
