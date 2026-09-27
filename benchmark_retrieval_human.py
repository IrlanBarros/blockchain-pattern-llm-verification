#!/usr/bin/env python3
"""Offline human retrieval benchmark; never calls Gemini or changes annotations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time

import pandas as pd

from llm_pipeline.config import KNOWN_OVERLAP_GROUPS
from llm_pipeline.context import prepare_optimized, select_context
from llm_pipeline.data import load_input, load_patterns, prepare_issue
from llm_pipeline.human_annotations import load_human_gold
from llm_pipeline.neural_embeddings import NeuralBackendUnavailable
from llm_pipeline.optimization import OptimizationConfig
from llm_pipeline.requests import stage1_request_params, stage2_request_params
from llm_pipeline.retrieval import HybridRetriever
from llm_pipeline.stages import stage2_pairs_from_stage1
from llm_pipeline.telemetry import estimate_tokens, prompt_text
from llm_pipeline.utils import sha256_file, write_dataframe_csv, write_json


def distribution(values):
    series = pd.Series(values, dtype=float)
    if series.empty:
        return {'n': 0, 'mean': None, 'median': None, 'p95': None, 'min': None, 'max': None, 'sum': None}
    return {'n': len(series), 'mean': float(series.mean()), 'median': float(series.median()),
            'p95': float(series.quantile(.95)), 'min': float(series.min()),
            'max': float(series.max()), 'sum': float(series.sum())}


def family_labels(pattern):
    labels = [f'F{index:02}' for index, group in enumerate(KNOWN_OVERLAP_GROUPS) if pattern in group]
    return labels or ['unassigned']


def fixed_historical_pairs(path, issue_keys):
    if not path.exists():
        return [], {'available': False, 'path': str(path), 'pairs': 0}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'request_status' in frame:
        frame = frame[frame.request_status == 'succeeded']
    pairs = stage2_pairs_from_stage1(frame)
    pairs = [pair for pair in pairs if pair[:2] in issue_keys]
    return pairs, {'available': True, 'path': str(path), 'sha256': sha256_file(path),
                   'pairs': len(pairs), 'note': 'Diagnostic workload only; not ground truth.'}


def token_metrics(items, retriever, catalog, pairs):
    stage1 = []
    stage2 = []
    for item in items:
        params = stage1_request_params(item, catalog, 'gemini-3.1-flash-lite', 1., 2048, 0, 'minimal')
        stage1.append(estimate_tokens(prompt_text(params)))
    by_key = {(item.repository, item.issue_number): item for item in items}
    for repository, number, pattern in pairs:
        item = by_key[(repository, number)]
        if retriever is not None:
            item = select_context(item, retriever, [pattern], stage=2)
        params = stage2_request_params(item, pattern, catalog, 'gemini-3.5-flash', 1., 2048, 0, 'low')
        stage2.append(estimate_tokens(prompt_text(params)))
    return {'stage1': distribution(stage1), 'stage2_fixed_historical_workload': distribution(stage2),
            'estimated_stage1_input_tokens': sum(stage1),
            'estimated_total_input_tokens': sum(stage1) + sum(stage2),
            '_stage1_by_key': {(item.repository, item.issue_number): value for item, value in zip(items, stage1)}}


def miss_record(row, pattern, item):
    score = item.retrieval['retrieval_scores'][pattern]
    reasons = score['selection_reasons']
    if not score['lexical_matches'] and score['semantic_rank'] > item.optimization.retrieval_max_candidates:
        failure = 'paraphrase_not_captured' if score['semantic'] > 0 else 'semantic_model_failure'
    elif score['semantic'] < item.optimization.retrieval_similarity_threshold:
        failure = 'threshold_too_high'
    elif not family_labels(pattern) or family_labels(pattern) == ['unassigned']:
        failure = 'missing_family_relation'
    else:
        failure = 'other'
    return {
        'repository': item.repository, 'issue_number': item.issue_number, 'human_pattern': pattern,
        'issue_title': row.get('issue_title', ''), 'evidence_text': row.get('evidence_text', ''),
        'lexical_score': 1 if score['lexical_matches'] else 0,
        'lexical_matches': score['lexical_matches'], 'semantic_score': score['semantic'],
        'semantic_rank': score['semantic_rank'], 'lsa_score': score.get('lsa'),
        'shortlist': item.retrieval['retrieval_candidates'], 'family': family_labels(pattern),
        'retrieval_reason': reasons, 'fallback_reason': item.retrieval['retrieval_fallback_reasons'],
        'suspected_failure_mode': failure,
    }


def evaluate(label, config, frame, gold, catalog, pairs, baseline_total):
    started = time.perf_counter()
    retriever = HybridRetriever(catalog, config)
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    items = [prepare_optimized(row, 12000, retriever) for _, row in frame.iterrows()]
    retrieval_seconds = time.perf_counter() - started
    by_key = {(item.repository, item.issue_number): item for item in items}
    row_by_key = {(row['repository'], row['issue_number']): row for row in frame.to_dict('records')}

    def score_pairs(pairs_to_score):
        misses = []
        mechanisms = Counter()
        recovered_by_issue = defaultdict(list)
        for repository, number, pattern in pairs_to_score:
            item = by_key[(repository, number)]
            recovered = pattern in item.retrieval['retrieval_candidates']
            recovered_by_issue[(repository, number)].append(recovered)
            if not recovered:
                misses.append(miss_record(row_by_key[(repository, number)], pattern, item))
                continue
            reasons = item.retrieval['retrieval_scores'][pattern]['selection_reasons']
            lexical = bool(item.retrieval['retrieval_scores'][pattern]['lexical_matches'])
            if not lexical and any(reason.startswith(('semantic_', 'lsa_')) for reason in reasons):
                mechanisms['semantic_only'] += 1
            if reasons == ['family_expansion']:
                mechanisms['family_only'] += 1
        positives = len(pairs_to_score)
        positive_issues = len({(r, n) for r, n, _ in pairs_to_score})
        fully_covered_issues = sum(all(values) for values in recovered_by_issue.values())
        return {'positive_pairs': positives, 'miss_count': len(misses),
                'recall': (positives - len(misses)) / positives if positives else None,
                'positive_issues': positive_issues,
                'issue_level_positive_coverage': fully_covered_issues / positive_issues if positive_issues else None,
                'semantic_only_recoveries': mechanisms['semantic_only'],
                'semantic_only_recovery_rate': mechanisms['semantic_only'] / positives if positives else None,
                'family_only_recoveries': mechanisms['family_only'],
                'family_expansion_recovery_rate': mechanisms['family_only'] / positives if positives else None,
                'misses': misses}

    strict = score_pairs(gold.strict_pairs)
    safety = score_pairs(gold.safety_pairs)
    sizes = [item.retrieval['retrieval_candidate_count'] for item in items]
    fallback_reasons = Counter(reason for item in items for reason in item.retrieval['retrieval_fallback_reasons'])
    token = token_metrics(items, retriever, catalog, pairs)
    total = token['estimated_total_input_tokens']
    token['estimated_token_reduction_percent'] = 100 * (baseline_total - total) / baseline_total if baseline_total else None
    stage1_by_key = token.pop('_stage1_by_key')

    pattern_stats = {}
    for pattern in sorted({p for _, _, p in gold.strict_pairs}):
        subset = [(r, n, p) for r, n, p in gold.strict_pairs if p == pattern]
        result = score_pairs(subset)
        pattern_stats[pattern] = {'n': len(subset), 'recall': result['recall'], 'misses': result['miss_count']}
    family_pairs = defaultdict(list)
    for pair in gold.strict_pairs:
        for family in family_labels(pair[2]):
            family_pairs[family].append(pair)
    family_stats = {}
    for family, subset in sorted(family_pairs.items()):
        result = score_pairs(subset)
        family_stats[family] = {'n': len(subset), 'recall': result['recall'], 'misses': result['miss_count']}

    historical_misses = []
    for repository, number, pattern in pairs:
        if pattern not in by_key[(repository, number)].retrieval['retrieval_candidates']:
            historical_misses.append({'repository': repository, 'issue_number': number, 'pattern': pattern})
    result = {
        'configuration': label, 'config': config.to_dict(),
        'human_candidate_recall': strict['recall'], 'human_candidate_misses': strict['miss_count'],
        'human_positive_pairs': strict['positive_pairs'],
        'issue_level_positive_coverage': strict['issue_level_positive_coverage'],
        'safety_recall_yes_plus_uncertain': safety['recall'],
        'safety_misses_yes_plus_uncertain': safety['miss_count'],
        'shortlist': distribution(sizes),
        'full_catalog_fallback_issues': sum(i.retrieval['retrieval_full_catalog_fallback'] for i in items),
        'full_catalog_fallback_rate': sum(i.retrieval['retrieval_full_catalog_fallback'] for i in items) / len(items),
        'fallback_reason_counts': dict(fallback_reasons),
        'lexical_match_rate': sum(bool(i.retrieval['lexical_match_count']) for i in items) / len(items),
        'semantic_only_recovery_rate': strict['semantic_only_recovery_rate'],
        'family_expansion_recovery_rate': strict['family_expansion_recovery_rate'],
        'average_patterns_added_by_family': statistics.mean(i.retrieval['family_patterns_added'] for i in items),
        **token,
        'recall_by_pattern': pattern_stats, 'recall_by_pattern_family': family_stats,
        'historical_candidate_retention': ((len(pairs) - len(historical_misses)) / len(pairs) if pairs else None),
        'historical_candidate_misses': len(historical_misses),
        'historical_candidates_are_gold': False,
        'local_timing': {'index_seconds': index_seconds, 'retrieval_and_context_seconds': retrieval_seconds,
                         'issues_per_second': len(items) / retrieval_seconds if retrieval_seconds else None},
        'semantic_index': {key: items[0].retrieval.get(key) for key in (
            'semantic_backend', 'semantic_index_fingerprint', 'semantic_index_cache_hit',
            'embedding_model', 'embedding_model_revision', 'embedding_model_artifact_sha256')},
        'misses': strict['misses'], 'safety_misses': safety['misses'],
        '_items': items, '_stage1_by_key': stage1_by_key,
    }
    return result


def fallback_report(result, gold, counterfactual=None):
    items = result['_items']
    counter_by_key = ({(item.repository, item.issue_number): item for item in counterfactual['_items']}
                      if counterfactual else {})
    gold_by_key = defaultdict(list)
    for repository, number, pattern in gold.strict_pairs:
        gold_by_key[(repository, number)].append(pattern)
    rules = {}
    for rule in sorted({reason for item in items for reason in item.retrieval['retrieval_fallback_reasons']}):
        affected = [item for item in items if rule in item.retrieval['retrieval_fallback_reasons']]
        protected = 0
        patterns_added = 0
        incremental_tokens = 0
        for item in affected:
            other = counter_by_key.get((item.repository, item.issue_number))
            if other is not None:
                patterns_added += len(set(item.retrieval['retrieval_candidates']) -
                                      set(other.retrieval['retrieval_candidates']))
                incremental_tokens += (result['_stage1_by_key'][(item.repository, item.issue_number)] -
                                       counterfactual['_stage1_by_key'][(item.repository, item.issue_number)])
            for pattern in gold_by_key[(item.repository, item.issue_number)]:
                if other is not None and pattern not in other.retrieval['retrieval_candidates']:
                    protected += 1
        rules[rule] = {
            'issues_affected': len(affected), 'human_positive_pairs_protected': protected,
            'patterns_added_vs_expanded_fallback_counterfactual': patterns_added if counterfactual else None,
            'estimated_incremental_stage1_tokens': incremental_tokens if counterfactual else None,
            'counterfactual_configuration': counterfactual['configuration'] if counterfactual else None,
            'note': 'Rules can overlap; counts and costs are not additive.',
        }
    return {'configuration': result['configuration'], 'rules': rules,
            'full_catalog_fallback_issues': result['full_catalog_fallback_issues']}


def public(result):
    return {key: value for key, value in result.items() if not key.startswith('_')}


def leave_one_positive_issue_out(results, strict_pairs):
    issue_keys = sorted({(r, n) for r, n, _ in strict_pairs})
    folds = []
    for held_out in issue_keys:
        training = [pair for pair in strict_pairs if pair[:2] != held_out]
        candidates = []
        for result in results:
            items = {(i.repository, i.issue_number): i for i in result['_items']}
            misses = sum(p not in items[(r, n)].retrieval['retrieval_candidates'] for r, n, p in training)
            candidates.append((misses, result['shortlist']['mean'], result['estimated_total_input_tokens'], result))
        chosen = min(candidates, key=lambda x: x[:3])[3]
        item = {(i.repository, i.issue_number): i for i in chosen['_items']}[held_out]
        held_pairs = [pair for pair in strict_pairs if pair[:2] == held_out]
        held_misses = [p for _, _, p in held_pairs if p not in item.retrieval['retrieval_candidates']]
        folds.append({'held_out_issue': '#'.join(held_out), 'selected_configuration': chosen['configuration'],
                      'positive_pairs': len(held_pairs), 'misses': held_misses})
    positives = sum(f['positive_pairs'] for f in folds)
    misses = sum(len(f['misses']) for f in folds)
    return {'method': 'deterministic leave-one-positive-issue-out internal validation',
            'reason': 'Only seven strict-positive issues; a 50-issue holdout would be statistically indefensible.',
            'positive_pairs': positives, 'misses': misses,
            'recall': (positives - misses) / positives if positives else None, 'folds': folds,
            'limitation': 'Internal pilot validation, not an unbiased population estimate.'}


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite benchmark artifacts: {output}')
    output.mkdir(parents=True, exist_ok=True)
    human_path = Path(args.human)
    patterns_path = Path(args.patterns)
    frame, _ = load_input(human_path)
    catalog = load_patterns(patterns_path)
    pair_path = Path(args.human_pairs) if args.human_pairs else None
    gold = load_human_gold(human_path, catalog, pair_path)
    write_dataframe_csv(output / 'human_gold_normalized.csv', gold.dataframe)
    write_json(output / 'dataset_summary.json', {
        **gold.summary,
        'human_directory_files': [str(path) for path in sorted(human_path.parent.glob('*')) if path.is_file()],
        'catalog': {'path': str(patterns_path), 'sha256': sha256_file(patterns_path), 'patterns': len(catalog.names)},
    })
    write_json(output / 'human_gold_summary.json', {
        'strict_definition': 'annotator relevance=yes plus annotator-confirmed patterns',
        'safety_definition': 'strict positives plus separately retained annotator uncertain patterns',
        'strict_pairs': [dict(repository=r, issue_number=n, pattern=p) for r, n, p in gold.strict_pairs],
        'uncertain_pairs': [dict(repository=r, issue_number=n, pattern=p) for r, n, p in gold.uncertain_pairs],
        'counts': {k: gold.summary[k] for k in ('strict_positive_issues','strict_positive_pairs','uncertain_issues','uncertain_pairs')},
    })

    baseline = [prepare_issue(row, 12000) for _, row in frame.iterrows()]
    baseline_by_key = {(item.repository, item.issue_number): item for item in baseline}
    issue_keys = set(baseline_by_key)
    pairs, historical = fixed_historical_pairs(Path(args.historical_stage1), issue_keys)
    baseline_token = token_metrics(baseline, None, catalog, pairs)
    baseline_token.pop('_stage1_by_key')
    baseline_total = baseline_token['estimated_total_input_tokens']
    full_control = {
        'configuration': 'legacy_full_catalog', 'human_candidate_recall': 1.0,
        'human_candidate_misses': 0, 'human_positive_pairs': len(gold.strict_pairs),
        'issue_level_positive_coverage': 1.0, 'safety_recall_yes_plus_uncertain': 1.0,
        'safety_misses_yes_plus_uncertain': 0,
        'shortlist': distribution([len(catalog.names)] * len(frame)),
        'full_catalog_fallback_issues': len(frame), 'full_catalog_fallback_rate': 1.0,
        **baseline_token, 'estimated_token_reduction_percent': 0.0,
        'note': '100% structural availability, not 100% LLM recall.',
    }

    configs = [
        ('conservative_current', OptimizationConfig(enabled=True)),
        ('conservative_expanded_fallback_counterfactual', OptimizationConfig(
            enabled=True, full_catalog_fallback=False)),
        ('lsa_anchor_off_high_confidence', OptimizationConfig(enabled=True, retrieval_require_lexical_anchor=False)),
        ('lsa_intermediate', OptimizationConfig(enabled=True, retrieval_confidence_threshold=.35,
                                                 retrieval_require_lexical_anchor=False)),
        ('aggressive_previous', OptimizationConfig(enabled=True, retrieval_confidence_threshold=.18,
                                                    retrieval_require_lexical_anchor=False)),
        ('lsa_expanded_fallback', OptimizationConfig(enabled=True, retrieval_confidence_threshold=.18,
                                                     retrieval_require_lexical_anchor=False,
                                                     full_catalog_fallback=False)),
    ]
    if not args.skip_neural:
        configs += [
            ('neural_a', OptimizationConfig(enabled=True, semantic_backend='neural',
                                            retrieval_similarity_threshold=.82,
                                            retrieval_confidence_threshold=.78,
                                            retrieval_require_lexical_anchor=False,
                                            full_catalog_fallback=False)),
            ('neural_b_lsa_union', OptimizationConfig(enabled=True, semantic_backend='lsa_neural',
                                                      retrieval_similarity_threshold=.18,
                                                      retrieval_confidence_threshold=.18,
                                                      retrieval_require_lexical_anchor=False,
                                                      retrieval_ambiguity_ratio=.99)),
        ]
    results, unavailable = [], []
    for label, config in configs:
        try:
            results.append(evaluate(label, config, frame, gold, catalog, pairs, baseline_total))
        except NeuralBackendUnavailable as exc:
            unavailable.append({'configuration': label, 'error': str(exc), 'simulated': False})

    by_label = {result['configuration']: result for result in results}
    write_json(output / 'conservative_current.json', public(by_label['conservative_current']))
    write_json(output / 'aggressive_previous.json', public(by_label['aggressive_previous']))
    write_json(output / 'neural_retrieval.json', {
        'results': [public(result) for result in results if result['config']['semantic_backend'] != 'lsa'],
        'unavailable': unavailable,
        'model_choice': {'name': 'intfloat/multilingual-e5-small',
                         'revision': '0e60b8d9d2166d80387f86e3b48ec9ced55f4d15',
                         'license': 'MIT', 'dimension': 384,
                         'rationale': 'Multilingual English/Portuguese encoder; CPU-sized; pinned local inference.'},
    })
    comparison = [full_control] + [public(result) for result in results]
    write_json(output / 'parameter_comparison.json', comparison)
    write_json(output / 'retrieval_misses.json', {
        result['configuration']: {'strict': result['misses'], 'safety': result['safety_misses']}
        for result in results})
    write_json(output / 'fallback_analysis.json', {
        'configurations': [fallback_report(
            result, gold,
            by_label.get('conservative_expanded_fallback_counterfactual')
            if result['configuration'] == 'conservative_current' else
            by_label.get('lsa_expanded_fallback')
            if result['configuration'] == 'aggressive_previous' else None
        ) for result in results],
        'conservative_explanation': ('The lexical-anchor guard fires on issues without exact canonical/alias/keyword '
                                     'phrases; it is independent of whether LSA has a useful top candidate.'),
    })
    write_json(output / 'token_comparison.json', {
        'estimator': 'uncalibrated ceil(Unicode characters/4), including schema; not official Gemini tokens',
        'fixed_stage2_workload': historical,
        'configurations': [{key: row[key] for key in (
            'configuration','estimated_stage1_input_tokens','estimated_total_input_tokens',
            'estimated_token_reduction_percent')} for row in comparison],
    })

    cv = leave_one_positive_issue_out(results, gold.strict_pairs)
    eligible = [result for result in results
                if result['human_candidate_misses'] == 0
                and result['safety_misses_yes_plus_uncertain'] == 0
                and result['full_catalog_fallback_rate'] <= .2
                and result['shortlist']['mean'] <= 20]
    # Retain a cheap safety fallback when its cost is small; efficiency breaks ties afterward.
    guarded = [result for result in eligible if result['config']['full_catalog_fallback']]
    pool = guarded or eligible
    recommended = min(pool, key=lambda r: (r['shortlist']['mean'], r['estimated_total_input_tokens'])) if pool else None
    acceptance = 'APROVAR' if recommended and recommended['estimated_token_reduction_percent'] >= 60 else 'NAO_APROVAR'
    recommendation = {
        'decision': acceptance,
        'recommended_configuration': recommended['configuration'] if recommended else None,
        'recommended_config': recommended['config'] if recommended else None,
        'reason': ('Zero known strict/safety misses, <=20 mean shortlist, <=20% full fallback and >=60% estimated '
                   'input reduction on this pilot.' if acceptance == 'APROVAR' else
                   'No measured configuration met all pilot recall and efficiency gates.'),
        'methodology': cv,
        'small_sample_warning': ('Only 8 strict-positive pairs across 7 issues and one uncertain pair. Approval is '
                                 'for an A/B candidate, not a population-level default guarantee.'),
        'human_reference_term': gold.summary['provenance']['term'],
        'independent_gold_standard': False,
        'neural_was_assumed_better': False,
        'ready_for_real_ab': bool(recommended),
        'ready_for_unattended_full_gpu_run': False,
    }
    write_json(output / 'final_recommendation.json', recommendation)
    print(json.dumps({'decision': acceptance, 'recommended': recommendation['recommended_configuration'],
                      'strict_pairs': len(gold.strict_pairs), 'uncertain_pairs': len(gold.uncertain_pairs),
                      'results': [{k: public(result)[k] for k in ('configuration','human_candidate_recall',
                          'human_candidate_misses','safety_recall_yes_plus_uncertain','shortlist',
                          'full_catalog_fallback_rate','estimated_total_input_tokens',
                          'estimated_token_reduction_percent')} for result in results]}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human', default='data/pilot/human/pilot_annotation_200_human.csv')
    parser.add_argument('--human-pairs')
    parser.add_argument('--patterns', default='blockchain_patterns_keywords_v3.csv')
    parser.add_argument('--historical-stage1',
                        default='outputs/runs/pilot_gemini_user/pilot_sample_run/stage1_results.csv')
    parser.add_argument('--output-dir', default='benchmarks/retrieval_human_200')
    parser.add_argument('--skip-neural', action='store_true')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
