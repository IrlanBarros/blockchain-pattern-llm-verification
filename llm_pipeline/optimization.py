"""Explicit, versioned optimization configuration. Legacy remains the default."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os

from .utils import sha256_text

OPTIMIZATION_VERSION = 'retrieval-v2-human-neural'


@dataclass(frozen=True)
class OptimizationConfig:
    enabled: bool = False
    retrieval_enabled: bool = True
    retrieval_min_candidates: int = 5
    retrieval_max_candidates: int = 10  # soft cap; lexical/family unions may exceed it
    retrieval_similarity_threshold: float = 0.18
    retrieval_family_expansion: bool = True
    retrieval_confidence_threshold: float = 0.55
    retrieval_require_lexical_anchor: bool = True
    retrieval_ambiguity_ratio: float = 0.8
    retrieval_fallback_multiplier: int = 2
    full_catalog_fallback: bool = True
    force_full_catalog: bool = False
    semantic_dimensions: int = 32
    semantic_backend: str = 'lsa'
    embedding_model: str = 'intfloat/multilingual-e5-small'
    embedding_model_revision: str = '0e60b8d9d2166d80387f86e3b48ec9ced55f4d15'
    embedding_cache_dir: str = '.cache/retrieval_embeddings'
    embedding_local_files_only: bool = True
    embedding_batch_size: int = 16
    semantic_chunk_chars: int = 1200
    stage1_max_context_tokens: int = 1600
    stage2_max_context_tokens: int = 2400
    comment_retrieval_enabled: bool = True
    context_chunk_chars: int = 800
    body_budget_fraction: float = 0.75
    compact_output: bool = True
    token_telemetry_enabled: bool = True
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    def __post_init__(self):
        if not 1 <= self.retrieval_min_candidates <= self.retrieval_max_candidates:
            raise ValueError('Retrieval requires 1 <= min_candidates <= max_candidates')
        if not all(0 <= value <= 1 for value in (self.retrieval_similarity_threshold,
                self.retrieval_confidence_threshold, self.retrieval_ambiguity_ratio)):
            raise ValueError('Retrieval thresholds/ratios must be between 0 and 1')
        if self.retrieval_fallback_multiplier < 2:
            raise ValueError('Fallback multiplier must expand the shortlist')
        if not 0 <= self.retrieval_similarity_threshold <= 1:
            raise ValueError('Similarity threshold must be between 0 and 1')
        if min(self.stage1_max_context_tokens, self.stage2_max_context_tokens) < 128:
            raise ValueError('Context budgets must be at least 128 estimated tokens')
        if self.semantic_dimensions < 1 or self.context_chunk_chars < 80:
            raise ValueError('Invalid semantic dimensions or context chunk size')
        if self.semantic_backend not in {'lsa', 'neural', 'lsa_neural'}:
            raise ValueError("SEMANTIC_BACKEND must be 'lsa', 'neural', or 'lsa_neural'")
        if not self.embedding_model or not self.embedding_model_revision:
            raise ValueError('Neural embeddings require an exact model name and revision')
        if self.embedding_batch_size < 1 or self.semantic_chunk_chars < 80:
            raise ValueError('Invalid embedding batch size or semantic chunk size')
        if not 0 < self.body_budget_fraction <= 1:
            raise ValueError('body_budget_fraction must be in (0, 1]')
        for price in (self.input_price_per_million, self.output_price_per_million):
            if price is not None and price < 0:
                raise ValueError('Token prices cannot be negative')

    @classmethod
    def from_environment(cls, *, enabled=False):
        defaults = cls()
        values = {'enabled': enabled}
        for field in fields(cls):
            if field.name == 'enabled':
                continue
            raw = os.environ.get(field.name.upper())
            if raw is None:
                continue
            default = getattr(defaults, field.name)
            if isinstance(default, bool):
                if raw.lower() not in {'true', 'false', '1', '0'}:
                    raise ValueError(f'{field.name.upper()} must be true/false')
                values[field.name] = raw.lower() in {'true', '1'}
            elif isinstance(default, int):
                values[field.name] = int(raw)
            elif isinstance(default, str):
                values[field.name] = raw
            else:
                values[field.name] = float(raw)
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True))
