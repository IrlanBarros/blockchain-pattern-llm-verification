"""Optimization integration, request provenance and strict resume guards."""
from __future__ import annotations
import json
from pathlib import Path

from .compact_wire import decode_payload
from .context import select_context
from .optimization import OPTIMIZATION_VERSION
from .retrieval import CompactCatalog, HybridRetriever
from .utils import append_jsonl, sha256_file, sha256_text, write_json


def optimization_metadata(config, catalog):
    root = Path(__file__).parent
    modules = sorted(p.name for p in root.glob('*.py')) + ['compact_catalog_v1.json']
    import numpy as np
    return {'numpy_version':np.__version__, 'optimization_version': OPTIMIZATION_VERSION, 'optimization_config': config.to_dict(),
            'optimization_config_sha256': config.fingerprint,
            'compact_catalog_sha256': CompactCatalog(catalog).fingerprint,
            'implementation_sha256': {name: sha256_file(root/name) for name in modules}}


def validate_optimization_resume(run_dir, config, catalog):
    path = run_dir / 'optimization_manifest.json'
    if not path.exists():
        if config.enabled:
            raise ValueError('Cannot resume a legacy run with optimization enabled; choose a new --run-id')
        return
    previous = json.loads(path.read_text())
    current = optimization_metadata(config, catalog)
    if previous['metadata'] != current:
        raise ValueError('Optimization/retrieval configuration, catalog or implementation changed; use a new --run-id')


def save_optimization_manifest(run_dir, config, catalog, prepared):
    manifest = {'metadata': optimization_metadata(config,catalog), 'issues': [
        {'repository': p.repository, 'issue_number': p.issue_number, 'artifact_sha256': sha256_text(p.artifact_text),
         'selected_spans': p.selected_spans, **p.retrieval} for p in prepared]}
    path = run_dir / 'optimization_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError('Deterministic retrieval/context differs from persisted manifest; refusing resume')
    else:
        write_json(path, manifest)
        write_json(run_dir / 'compact_catalog.json', CompactCatalog(catalog).to_dict())


def decode_stage(payload, catalog, prepared, stage):
    if not prepared.optimization or not prepared.optimization.enabled:
        return payload
    return decode_payload(payload, catalog, stage=stage,
               allowed=prepared.retrieval['retrieval_candidates'] if stage==1 else None)


def prepare_stage2_contexts(prepared_by_key, catalog, run_dir):
    """Build a reusable extractor; retain Stage 1 evidence for each candidate."""
    prepared = next(iter(prepared_by_key.values()), None)
    if not prepared or not prepared.optimization or not prepared.optimization.enabled:
        return lambda item, pattern: item
    retriever = HybridRetriever(catalog, prepared.optimization)
    evidence = {}
    path = run_dir / 'stage1_results.csv'
    if path.exists():
        import pandas as pd
        for row in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict('records'):
            if row.get('request_status','succeeded') != 'succeeded':
                continue
            for candidate in json.loads(row.get('candidate_details') or '[]'):
                evidence[(row['repository'],row['issue_number'],candidate['pattern'])] = candidate.get('evidence_text','')
    def select(item, pattern):
        required = (evidence.get((item.repository,item.issue_number,pattern), ''),)
        return select_context(item,retriever,[pattern],stage=2,required=required)
    return select


def audit_request(run_dir, prepared, params, stage, pattern=''):
    append_jsonl(run_dir / 'request_diagnostics.jsonl', {
        'stage':stage,'repository':prepared.repository,'issue_number':prepared.issue_number,'pattern':pattern,
        'request_sha256':sha256_text(json.dumps(params,sort_keys=True,ensure_ascii=False)),
        'artifact_sha256':sha256_text(prepared.artifact_text), 'selected_spans':prepared.selected_spans,
        'input_truncated':prepared.input_truncated,
    })
