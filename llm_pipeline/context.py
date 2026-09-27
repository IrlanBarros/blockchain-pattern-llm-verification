"""Deterministic extractive context selection with auditable original offsets."""
from __future__ import annotations

from dataclasses import replace
import math
import re

from .models import PreparedIssue
from .normalization import clean_text
from .retrieval import HybridRetriever, terms


def chunks(text: str, max_chars: int):
    spans = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind('\n', start + max_chars // 2, end)
            if cut < 0:
                cut = text.rfind(' ', start + max_chars // 2, end)
            if cut >= 0:
                end = cut + 1
        spans.append((start, end))
        start = end
    return spans


def select_spans(text, budget, query, chunk_chars, *, priority_edges=False, required=()):
    if len(text) <= budget:
        return text, [(0, len(text))] if text else []
    if budget <= 0:
        return '', []
    spans = chunks(text, min(chunk_chars, max(80, budget // 3)))
    query_terms = set(terms(query))
    token_sets = [set(terms(text[a:b])) for a, b in spans]
    # IDF across passages downweights recurring boilerplate and log terms.
    idf = {t: math.log(1 + len(spans) / (1 + sum(t in ts for ts in token_sets))) for t in query_terms}
    protected = set()
    for evidence in required:
        offset = text.find(evidence) if evidence else -1
        if offset >= 0:
            protected.update(i for i, (a, b) in enumerate(spans) if a < offset + len(evidence) and b > offset)
    def score(i):
        ts = token_sets[i]
        return sum(idf[t] for t in query_terms & ts) / math.sqrt(max(1, len(ts)))
    order = sorted(range(len(spans)), key=lambda i: (i not in protected,
                    not (priority_edges and i in {0, len(spans)-1}), -score(i), i))
    selected, used = [], 0
    for i in order:
        a, b = spans[i]
        cost = b - a + (len('\n[... omitted ...]\n') if selected else 0)
        if used + cost <= budget:
            selected.append(i)
            used += cost
        elif i in protected:
            raise ValueError('Context budget would discard Stage 1 evidence; increase STAGE2_MAX_CONTEXT_TOKENS')
    chosen = [spans[i] for i in sorted(selected)]
    # Adjacent spans are joined without artificial separators so literal quotes survive.
    merged = []
    for a, b in chosen:
        if merged and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return '\n[... omitted ...]\n'.join(text[a:b] for a, b in merged), merged


def select_context(prepared: PreparedIssue, retriever: HybridRetriever, names, *, stage=1, required=()):
    cfg = retriever.config
    source = prepared.source_sections
    budget = (cfg.stage1_max_context_tokens if stage == 1 else cfg.stage2_max_context_tokens) * 4
    fixed = (f"<artifact_type>{source['type']}</artifact_type>\n<state>{source['state']}</state>\n"
             f"<labels>{source['labels']}</labels>\n<title>{source['title']}</title>\n")
    marker = '<context_omitted>Some source passages were omitted; do not infer absent evidence.</context_omitted>\n'
    available = budget - len(fixed) - len('<body></body>\n<comments></comments>') - len(marker)
    if available < 80:
        raise ValueError('Title/metadata exceed context budget; increase STAGE*_MAX_CONTEXT_TOKENS')
    body, comments = source['body'], source['comments']
    body_budget = min(len(body), available if not comments else int(available * cfg.body_budget_fraction))
    query = source['title'] + '\n' + '\n'.join(retriever.compact.records[n].short_definition for n in names)
    body_out, body_spans = select_spans(body, body_budget, query, cfg.context_chunk_chars,
                                      priority_edges=True, required=required)
    comment_budget = max(0, available - len(body_out))
    comment_query = query + '\n' + body_out if cfg.comment_retrieval_enabled else ''
    comment_out, comment_spans = select_spans(comments, comment_budget, comment_query,
                     cfg.context_chunk_chars, priority_edges=not cfg.comment_retrieval_enabled, required=required)
    omitted = sum(b-a for a,b in body_spans) < len(body) or sum(b-a for a,b in comment_spans) < len(comments)
    artifact = fixed + (marker if omitted else '') + f'<body>{body_out}</body>\n<comments>{comment_out}</comments>'
    for evidence in required:
        if evidence and (evidence in body or evidence in comments or evidence in source['title']) and evidence not in artifact:
            raise ValueError('Context selection lost Stage 1 evidence; increase context budget')
    return replace(prepared, artifact_text=artifact, included_char_count=len(artifact), input_truncated=omitted,
                   selected_spans={'body': body_spans, 'comments': comment_spans})


def prepare_optimized(row, max_chars, retriever):
    from .data import prepare_issue
    original = prepare_issue(row, max_chars)
    source = {key: clean_text(row.get(column, '')) for key, column in (
        ('title','issue_title'),('body','issue_body'),('comments','concatenated_comments'),
        ('type','type'),('state','state'),('labels','labels'))}
    # Retrieval sees all source text, including evidence beyond the LLM budget.
    query = '\n'.join(source[k] for k in ('title','body','comments'))
    result = retriever.retrieve(query)
    prepared = replace(original, source_sections=source, retrieval=result, optimization=retriever.config)
    return select_context(prepared, retriever, result['retrieval_candidates'])
