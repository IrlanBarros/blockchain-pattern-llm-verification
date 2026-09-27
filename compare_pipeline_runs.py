#!/usr/bin/env python3
"""Compare completed live A/B runs; missing usage/labels never pass acceptance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_pipeline import (binary_metrics, human_pairs_from_files, normalize_human_issues,
                               normalize_human_pairs)
from llm_pipeline.checkpoint import ResultStore
from llm_pipeline.data import load_input, load_patterns, prepare_issue
from llm_pipeline.human_annotations import load_human_gold
from llm_pipeline.normalization import CanonicalLookup, NormalizationReport, read_csv_robust, resolve_repository_columns
from llm_pipeline.stages import pipeline_integrity_report, stage2_pairs_from_stage1
from llm_pipeline.utils import write_json


def compare(args):
    dirs={name:Path(getattr(args,name+'_run')) for name in ('baseline','optimized')}
    metadata={name:json.loads((path/'run_metadata.json').read_text()) for name,path in dirs.items()}
    for key in ('input_sha256','patterns_sha256','stage1_model','stage2_model','temperature','seed',
                'stage1_thinking_level','stage2_thinking_level','stage1_max_tokens','stage2_max_tokens','limit','mode'):
        if metadata['baseline'].get(key)!=metadata['optimized'].get(key):
            raise ValueError(f'Incomparable runs: {key} differs')
    data,_=load_input(dirs['baseline']/'input_clean_snapshot.csv')
    catalog=load_patterns(dirs['baseline']/'pattern_catalog_clean_snapshot.csv')
    prepared=[prepare_issue(row,metadata['baseline']['max_input_chars']) for _,row in data.iterrows()]
    keys=set(zip(data.repository,data.issue_number))
    lookup=CanonicalLookup(catalog.names,label='pattern_catalog')
    labels=pd.DataFrame(columns=['repository','issue_number','pattern','human_verdict'])
    if args.human_issues:
        raw,report=read_csv_robust(Path(args.human_issues),source='human_issues')
        raw=resolve_repository_columns(raw,require=True)
        if 'patterns_confirmed' in raw.columns:
            # Provenance-preserving pilot schema: use the explicit A1-aware adapter.
            gold=load_human_gold(Path(args.human_issues),catalog,
                                 Path(args.human_pairs) if args.human_pairs else None)
            labels=pd.DataFrame([
                {'repository':r,'issue_number':n,'pattern':p,'human_verdict':'yes'}
                for r,n,p in gold.strict_pairs
            ] + [
                {'repository':r,'issue_number':n,'pattern':p,'human_verdict':'uncertain'}
                for r,n,p in gold.uncertain_pairs
            ])
        else:
            # Unannotated cells are unknown; do not turn them into negatives.
            labeled=raw.loc[raw.relevant_to_pattern_study.str.strip()!=''].copy()
            issues=normalize_human_issues(labeled,pattern_lookup=lookup,report=report)
            pairs=pd.DataFrame()
            if args.human_pairs:
                pairs,pair_report=read_csv_robust(Path(args.human_pairs),source='human_pairs')
                pairs=normalize_human_pairs(pairs,pattern_lookup=lookup,report=pair_report)
            labels=human_pairs_from_files(issues,pairs)
    elif args.human_pairs:
        raw,report=read_csv_robust(Path(args.human_pairs),source='human_pairs')
        labels=normalize_human_pairs(raw,pattern_lookup=lookup,report=report)
    labels=labels.loc[[ (r,n) in keys for r,n in zip(labels.repository,labels.issue_number) ]]
    positive={tuple(row) for row in labels.loc[labels.human_verdict=='yes',['repository','issue_number','pattern']].itertuples(index=False,name=None)}
    report={'comparison':'live_runs_only','human_positive_pairs':len(positive),'runs':{}}
    misses={}
    for name,path in dirs.items():
        s1=ResultStore(path/'stage1_results.csv',stage='stage1').dataframe
        s2=ResultStore(path/'stage2_results.csv',stage='stage2').dataframe
        integrity=pipeline_integrity_report(prepared,s1,s2)
        candidates=set(stage2_pairs_from_stage1(s1))
        misses[name]=positive-candidates
        summary_path=path/'token_summary.json'
        tokens=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        verdicts={(r.repository,r.issue_number,r.pattern):r.verdict for r in s2.itertuples() if r.request_status=='succeeded'}
        binary=labels.loc[labels.human_verdict.isin(['yes','no'])]
        actual,strict_predictions=[],[]
        conditional_actual,conditional_predictions=[],[]
        for r in binary.itertuples():
            verdict=verdicts.get((r.repository,r.issue_number,r.pattern),'no')
            actual.append(r.human_verdict)
            strict_predictions.append('yes' if verdict=='yes' else 'no')
            if (r.repository,r.issue_number,r.pattern) in verdicts and verdict in {'yes','no'}:
                conditional_actual.append(r.human_verdict)
                conditional_predictions.append(verdict)
        quality={'stage1_candidate_recall':(len(positive)-len(misses[name]))/len(positive) if positive else None,
                 'stage1_missed_positives':[list(pair) for pair in sorted(misses[name])],
                 'stage2_conditional_binary':binary_metrics(conditional_actual,conditional_predictions),
                 'end_to_end_strict_yes':binary_metrics(actual,strict_predictions),
                 'note':'Conditional metrics only for generated labeled binary pairs. Strict yes includes screened-out pairs and abstentions as not-yes.'}
        if integrity['status']!='OK':
            quality['stage1_candidate_recall']=None
            quality['stage2_conditional_binary']=None
            quality['end_to_end_strict_yes']=None
        manifest=path/'optimization_manifest.json'
        retrieval_recall=None
        retrieval_misses=None
        if manifest.exists() and positive:
            issues=json.loads(manifest.read_text())['issues']
            retrieved={(i['repository'],i['issue_number'],p) for i in issues for p in i['retrieval_candidates']}
            retrieval_misses=[list(pair) for pair in sorted(positive-retrieved)]
            retrieval_recall=(len(positive)-len(retrieval_misses))/len(positive)
        report['runs'][name]={'integrity':integrity,'tokens':tokens,'quality':quality,
                              'retrieval_recall':retrieval_recall,'retrieval_missed_positives':retrieval_misses}
    old,new=[report['runs'][n]['tokens'] for n in ('baseline','optimized')]
    usage_ok=old.get('usage_complete') and new.get('usage_complete')
    savings=100*(old['total_tokens']-new['total_tokens'])/old['total_tokens'] if usage_ok and old.get('total_tokens') else None
    report['tokens_saved_vs_baseline']=old['total_tokens']-new['total_tokens'] if usage_ok else None
    report['percentage_saved_vs_baseline']=savings
    report['new_stage1_false_negatives']=[list(p) for p in sorted(misses['optimized']-misses['baseline'])] if positive else None
    gates={'complete_runs':all(r['integrity']['status']=='OK' for r in report['runs'].values()),
           'measured_token_reduction_ge_60_percent':savings is not None and savings>=60,
           'retrieval_recall_ge_98_percent':report['runs']['optimized']['retrieval_recall'] is not None and report['runs']['optimized']['retrieval_recall']>=.98}
    qualities=[report['runs'][n]['quality'] for n in ('baseline','optimized')]
    recalls=[q['stage1_candidate_recall'] for q in qualities]
    gates['stage1_recall_drop_le_1pp']=all(r is not None for r in recalls) and recalls[0]-recalls[1]<=.01
    for metric in ('precision','recall','f1'):
        values=[(q['end_to_end_strict_yes'] or {}).get(metric) for q in qualities]
        gates['end_to_end_'+metric+'_drop_le_2pp']=all(v is not None for v in values) and values[0]-values[1]<=.02
    for metric in ('precision','recall','f1'):
        values=[(q['stage2_conditional_binary'] or {}).get(metric) for q in qualities]
        gates['stage2_conditional_'+metric+'_drop_le_2pp']=all(v is not None for v in values) and values[0]-values[1]<=.02
    report['acceptance_gates']=gates
    report['acceptance']='PASSED_ON_OBSERVED_SAMPLE_ONLY' if all(gates.values()) else 'NOT_APPROVED'
    report['statistical_note']='Report absolute misses; a finite sample does not establish population recall >=98%.'
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    write_json(out,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if all(gates.values()) else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',required=True)
    p.add_argument('--optimized-run',required=True)
    p.add_argument('--human-issues')
    p.add_argument('--human-pairs')
    p.add_argument('--output',required=True)
    return compare(p.parse_args())

if __name__=='__main__':
    raise SystemExit(main())
