"""Read-only adapter for human-reviewed pilot annotations.

Structural normalization happens in memory. No source annotation is rewritten,
and LLM-prefixed values are never treated as human decisions when an annotator
component is available.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd

from .normalization import CanonicalLookup, canonicalize_multivalue, clean_token, read_csv_robust
from .utils import sha256_file

HUMAN_SCHEMA_VERSION = 'human-pilot-adapter-v1'
_VERDICTS = {'yes', 'no', 'uncertain', 'insufficient_context'}


def human_component(value: str, annotator: str = 'A1') -> str:
    """Extract `A1: ...` from provenance-preserving composite cells."""
    text = '' if value is None else str(value).strip()
    match = re.search(rf'(?:^|;\s*){re.escape(annotator)}:\s*(.*?)(?=;\s*[^;]+:|$)',
                      text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


@dataclass(frozen=True)
class HumanGold:
    dataframe: pd.DataFrame
    strict_pairs: tuple[tuple[str, str, str], ...]
    uncertain_pairs: tuple[tuple[str, str, str], ...]
    summary: dict

    @property
    def safety_pairs(self):
        return tuple(dict.fromkeys((*self.strict_pairs, *self.uncertain_pairs)))


def _distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def load_human_gold(path: Path, catalog, pair_path: Path | None = None) -> HumanGold:
    raw_stat = path.stat()
    frame, normalization = read_csv_robust(path, source=str(path))
    repository_column = 'repository' if 'repository' in frame else 'repository_full_name'
    required = {repository_column, 'issue_number', 'relevant_to_pattern_study'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Human annotation file missing required columns: {sorted(missing)}')
    frame = frame.copy()
    frame['repository'] = frame[repository_column].map(clean_token)
    frame['issue_number'] = frame['issue_number'].map(clean_token).str.replace(r'\.0+$', '', regex=True)
    blank = (frame.repository == '') | (frame.issue_number == '')
    duplicates = frame.duplicated(['repository', 'issue_number'], keep=False)
    if blank.any() or duplicates.any():
        raise ValueError('Human issue keys must be nonblank and unique after structural normalization')
    lookup = CanonicalLookup(catalog.names, label='human gold patterns')
    confirmed_column = 'patterns_confirmed' if 'patterns_confirmed' in frame else 'patterns_present'
    uncertain_column = 'patterns_uncertain' if 'patterns_uncertain' in frame else ''
    normalized_rows = []
    strict, uncertain = [], []
    human_verdicts = []
    for index, row in frame.iterrows():
        verdict = human_component(row.get('relevant_to_pattern_study', ''))
        if verdict not in _VERDICTS:
            raise ValueError(f'Unknown human relevance at CSV row {index + 2}: {verdict!r}')
        confirmed_raw = human_component(row.get(confirmed_column, ''))
        uncertain_raw = human_component(row.get(uncertain_column, '')) if uncertain_column else ''
        confirmed = canonicalize_multivalue(confirmed_raw, lookup, row=index + 2,
                                             column=confirmed_column) if confirmed_raw else []
        uncertain_names = canonicalize_multivalue(uncertain_raw, lookup, row=index + 2,
                                                   column=uncertain_column) if uncertain_raw else []
        if verdict == 'yes' and not confirmed:
            raise ValueError(f'Human-positive issue without confirmed pattern at CSV row {index + 2}')
        key = (row['repository'], row['issue_number'])
        if verdict == 'yes':
            strict.extend((*key, name) for name in confirmed)
        if verdict == 'uncertain':
            uncertain.extend((*key, name) for name in uncertain_names)
        human_verdicts.append(verdict)
        normalized_rows.append({
            'repository': key[0], 'issue_number': key[1], 'human_relevance': verdict,
            'confirmed_patterns': '|'.join(confirmed),
            'uncertain_patterns': '|'.join(uncertain_names),
            'insufficient_context': human_component(row.get('insufficient_context', '')),
            'confidence': human_component(row.get('confidence', '')),
        })
    pair_summary = {'found': False, 'path': None, 'rows': 0, 'schema': []}
    if pair_path is not None and pair_path.exists():
        pairs, _ = read_csv_robust(pair_path, source=str(pair_path))
        needed = {'repository', 'issue_number', 'pattern', 'human_verdict'}
        if not needed <= set(pairs):
            raise ValueError(f'Human pair file missing columns: {sorted(needed - set(pairs))}')
        strict, uncertain = [], []
        seen = set()
        for index, row in pairs.iterrows():
            verdict = human_component(row['human_verdict'])
            name = lookup.canonicalize(row['pattern'], row=index + 2, column='pattern')
            key = (clean_token(row['repository']), clean_token(row['issue_number']), name)
            if key in seen:
                raise ValueError(f'Duplicate human pair at row {index + 2}: {key}')
            seen.add(key)
            if verdict == 'yes': strict.append(key)
            elif verdict == 'uncertain': uncertain.append(key)
            elif verdict not in {'no', 'insufficient_context'}:
                raise ValueError(f'Unknown pair verdict at row {index + 2}: {verdict!r}')
        pair_summary = {'found': True, 'path': str(pair_path), 'sha256': sha256_file(pair_path),
                        'rows': len(pairs), 'schema': list(pairs.columns)}
    normalized = pd.DataFrame(normalized_rows)
    missing_counts = {column: int((frame[column] == '').sum()) for column in frame.columns}
    pattern_counts = Counter(name for _, _, name in strict)
    uncertain_counts = Counter(name for _, _, name in uncertain)
    summary = {
        'adapter_version': HUMAN_SCHEMA_VERSION,
        'source': {'path': str(path), 'sha256': sha256_file(path), 'bytes': raw_stat.st_size,
                   'mtime_utc': datetime.fromtimestamp(raw_stat.st_mtime, timezone.utc).isoformat()},
        'rows': len(frame), 'unique_issue_keys': len(frame[['repository','issue_number']].drop_duplicates()),
        'duplicate_issue_rows': int(duplicates.sum()), 'schema': list(frame.columns),
        'missing_by_column': missing_counts,
        'raw_value_distributions': {column: _distribution(frame[column]) for column in (
            'relevant_to_pattern_study', 'patterns_present', 'patterns_confirmed',
            'patterns_uncertain', 'confidence', 'insufficient_context', 'annotator_id',
        ) if column in frame},
        'human_relevance_distribution': dict(Counter(human_verdicts)),
        'strict_positive_issues': len({(r, n) for r, n, _ in strict}),
        'strict_positive_pairs': len(strict), 'strict_patterns': dict(sorted(pattern_counts.items())),
        'uncertain_issues': len({(r, n) for r, n, _ in uncertain}),
        'uncertain_pairs': len(uncertain), 'uncertain_patterns': dict(sorted(uncertain_counts.items())),
        'pair_annotations': pair_summary,
        'normalization': normalization.summary(),
        'provenance': {
            'term': 'human-reviewed pilot annotations',
            'independent_gold_standard': False,
            'reason': ('Cells retain both llm and annotator-prefixed decisions; exposure to prior LLM '
                       'outputs is evident, while independent blinding is not documented.'),
        },
    }
    return HumanGold(normalized, tuple(strict), tuple(uncertain), summary)
