#!/usr/bin/env python3
"""Offline paired prompt benchmark. Does not infer or manufacture LLM quality.

Use --pairs-from-stage1 to compare identical historical candidate workloads.
Without it Stage 2 probes every pattern; that is not a pipeline token total.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from llm_pipeline.context import prepare_optimized, select_context
from llm_pipeline.data import load_input, load_patterns
from llm_pipeline.data import prepare_issue
from llm_pipeline.normalization import CanonicalLookup, canonicalize_multivalue
from llm_pipeline.optimization import OptimizationConfig
from llm_pipeline.optimization_runtime import optimization_metadata
from llm_pipeline.requests import stage1_request_params, stage2_request_params
from llm_pipeline.retrieval import HybridRetriever
from llm_pipeline.stages import stage2_pairs_from_stage1
from llm_pipeline.telemetry import estimate_tokens, prompt_text
from llm_pipeline.utils import sha256_file, write_json


def distribution(values):
    if not values:
        return {'n':0, 'sum':None, 'mean':None, 'median':None, 'p95':None, 'min':None, 'max':None}
    s = pd.Series(values, dtype=float)
    return {'n':len(values),'sum':float(s.sum()),'mean':float(s.mean()),'median':float(s.median()),
            'p95':float(s.quantile(.95)),'min':float(s.min()),'max':float(s.max())}


def retrieval_metrics(df, prepared, catalog):
    lookup = CanonicalLookup(catalog.names, label='gold patterns')
    misses, positives = [], 0
    labeled = 0
    index = {(p.repository,p.issue_number):p for p in prepared}
    for row in df.to_dict('records'):
        relevance = row.get('relevant_to_pattern_study', '')
        if relevance:
            if relevance not in {'yes','no','uncertain','insufficient_context'}:
                raise ValueError(f'Unknown human relevance: {relevance!r}')
            labeled += 1
        if relevance != 'yes':
            continue
        names = canonicalize_multivalue(row.get('patterns_present',''), lookup)
        if not names:
            raise ValueError('Human-positive issue without pattern labels cannot measure retrieval recall')
        selected = index[(row['repository'],row['issue_number'])].retrieval['retrieval_candidates']
        for name in names:
            positives += 1
            if name not in selected:
                misses.append({'repository':row['repository'],'issue_number':row['issue_number'],'pattern':name})
    return {'human_labeled_issues':labeled,'human_positive_pairs':positives,
            'recall':(positives-len(misses))/positives if positives else None,
            'missed_positive_count':len(misses) if positives else None,'missed_positives':misses,
            'note':'No human labels: quality unavailable.' if not labeled else
                   'Finite diagnostic sample; absolute misses reported, no population-level recall guarantee.'}


def run(args):
    df,_ = load_input(Path(args.input))
    catalog = load_patterns(Path(args.patterns))
    config = OptimizationConfig.from_environment(enabled=True)
    start = time.perf_counter()
    retriever = HybridRetriever(catalog,config)
    index_seconds = time.perf_counter()-start
    start = time.perf_counter()
    baseline = [prepare_issue(row,12000) for _,row in df.iterrows()]
    baseline_prepare = time.perf_counter()-start
    start = time.perf_counter()
    optimized = [prepare_optimized(row,12000,retriever) for _,row in df.iterrows()]
    optimized_prepare = time.perf_counter()-start
    by_key = {(p.repository,p.issue_number):p for p in optimized}
    if args.pairs_from_stage1:
        stage1 = pd.read_csv(args.pairs_from_stage1,dtype=str,keep_default_na=False)
        stage_keys = set(zip(stage1.repository,stage1.issue_number))
        if stage_keys != set(by_key):
            raise ValueError('Historical Stage 1 must cover exactly the same issue keys')
        # Verify content, not just issue numbers, when an originating run is provided.
        metadata_path = Path(args.pairs_from_stage1).parent/'run_metadata.json'
        if not metadata_path.exists() or json.loads(metadata_path.read_text()).get('input_sha256') != sha256_file(Path(args.input)):
            raise ValueError('Historical workload requires matching input SHA256 in run_metadata.json')
        pairs = stage2_pairs_from_stage1(stage1)
        pair_source = 'fixed_historical_stage1_candidates_not_gold'
    else:
        pairs = [(p.repository,p.issue_number,n) for p in baseline for n in catalog.names]
        pair_source = 'all_patterns_probe_not_actual_pipeline_workload'
    baseline_index = {(p.repository,p.issue_number):p for p in baseline}
    rows = []
    times = {'baseline':baseline_prepare,'optimized':optimized_prepare+index_seconds}
    for profile,items in [('baseline',baseline),('optimized',optimized)]:
        start = time.perf_counter()
        for p in items:
            params = stage1_request_params(p,catalog,'gemini-3.1-flash-lite',1.,2048,0,'minimal')
            text = prompt_text(params)
            rows.append({'profile':profile,'stage':'stage1','repository':p.repository,'issue_number':p.issue_number,
                         'pattern':'','prompt_chars':len(text),'input_tokens_estimate':estimate_tokens(text)})
        for repository,number,pattern in pairs:
            p = baseline_index[(repository,number)] if profile=='baseline' else select_context(by_key[(repository,number)],retriever,[pattern],stage=2)
            params = stage2_request_params(p,pattern,catalog,'gemini-3.5-flash',1.,2048,0,'low')
            text = prompt_text(params)
            rows.append({'profile':profile,'stage':'stage2','repository':repository,'issue_number':number,
                         'pattern':pattern,'prompt_chars':len(text),'input_tokens_estimate':estimate_tokens(text)})
        times[profile] += time.perf_counter()-start
    metrics = {}
    for profile in ('baseline','optimized'):
        metrics[profile] = {stage:distribution([r['input_tokens_estimate'] for r in rows if r['profile']==profile and r['stage']==stage])
                            for stage in ('stage1','stage2')}
        metrics[profile]['local_preparation_seconds'] = times[profile]
    reductions = {}
    for stage in ('stage1','stage2'):
        old,new = metrics['baseline'][stage]['sum'],metrics['optimized'][stage]['sum']
        reductions[stage] = 100*(old-new)/old if old else None
    old = sum(metrics['baseline'][s]['sum'] or 0 for s in ('stage1','stage2'))
    new = sum(metrics['optimized'][s]['sum'] or 0 for s in ('stage1','stage2'))
    quality = retrieval_metrics(df,optimized,catalog)
    diagnostics = [{'repository':p.repository,'issue_number':p.issue_number,
                    'selected_spans':p.selected_spans,'input_truncated':p.input_truncated,**p.retrieval} for p in optimized]
    dropped_historical = [{'repository':r,'issue_number':n,'pattern':p} for r,n,p in pairs
                          if args.pairs_from_stage1 and p not in by_key[(r,n)].retrieval['retrieval_candidates']]
    report = {'mode':'offline_prompt_construction_only','input_sha256':sha256_file(Path(args.input)),
              'issues':len(df), 'patterns_sha256':sha256_file(Path(args.patterns)),
              'estimator':'uncalibrated ceil(Unicode characters/4), including schema; not official Gemini tokens',
              'stage2_pair_source':pair_source,'stage2_pairs':len(pairs),'metrics':metrics,
              'input_reduction_percent':reductions,
              'fixed_workload_input_estimate':{'baseline':old,'optimized':new,'reduction_percent':100*(old-new)/old if old else None},
              'retrieval':{**quality,'candidate_count':distribution([p.retrieval['retrieval_candidate_count'] for p in optimized]),
                           'fallback_issues':sum(p.retrieval['retrieval_fallback_used'] for p in optimized)},
              'historical_llm_candidates_excluded':dropped_historical,
              'quality':{'stage1_candidate_recall':None,'stage2_precision':None,'stage2_recall':None,'stage2_f1':None,
                         'new_llm_false_negatives':None},
              'actual_output_tokens':None,'actual_total_tokens':None,'actual_llm_elapsed_seconds':None,
              'acceptance':'NOT_VALIDATED: live paired inference and populated human annotations required',
              'provenance':optimization_metadata(config,catalog)}
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite benchmark artifacts: {out}')
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'prompt_measurements.csv',index=False)
    write_json(out/'retrieval_diagnostics.json',diagnostics)
    write_json(out/'report.json',report)
    table = ['# Offline token optimization benchmark','',report['acceptance'],'',report['estimator'],'',
             '| Metric | Baseline | Optimized | Reduction |','|---|---:|---:|---:|']
    for stage in ('stage1','stage2'):
        table.append(f"| Mean input estimate {stage} | {metrics['baseline'][stage]['mean']:.2f} | {metrics['optimized'][stage]['mean']:.2f} | {reductions[stage]:.2f}% |")
    table += [f'| Fixed workload input estimate | {old:.0f} | {new:.0f} | {100*(old-new)/old:.2f}% |',
              f"| Local preparation seconds | {times['baseline']:.3f} | {times['optimized']:.3f} | — |",'',
              'Stage 2 workload: '+pair_source+'. Outputs, model latency and stage quality were not measured.',
              f"Human positive pairs: {quality['human_positive_pairs']}; retrieval recall: {quality['recall']}.",
              f'Historical LLM candidates excluded (not human false negatives): {len(dropped_historical)}.',
              'Full diagnostics and provenance are in report.json and retrieval_diagnostics.json.']
    (out/'REPORT.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'provenance','historical_llm_candidates_excluded'}},indent=2))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--patterns',default='blockchain_patterns_keywords_v3.csv')
    parser.add_argument('--pairs-from-stage1')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    return run(args)

if __name__=='__main__':
    raise SystemExit(main())
