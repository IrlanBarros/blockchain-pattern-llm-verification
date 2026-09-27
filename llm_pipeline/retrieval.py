"""Permissive lexical + latent semantic retrieval. Never produces verdicts.

LSA is a local, deterministic alternative to neural embeddings using NumPy
(already required by pandas). It learns co-occurrence from catalog definitions,
not human labels. Its limited paraphrase coverage requires a conservative fallback.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

from .config import KNOWN_OVERLAP_GROUPS
from .models import PatternCatalog
from .optimization import OptimizationConfig
from .utils import sha256_text

FALSE_FRIENDS = {
    'Oracle': ['Oracle Database', 'Oracle Cloud', 'SyncOracle'],
    'Proxy contract': ['HTTP proxy', 'reverse proxy', 'nginx proxy'],
    'Snapshotting': ['test snapshot', 'UI snapshot', 'forge snapshot', 'SDK event checkpoint'],
    'Mutex': ['generic thread lock', 'OS/process/database mutex'],
    'Relay contract': ['Relay Chain'],
    'Blocklist': ['domain/IP blocklist'],
    'Router contract': ['network router'],
    'Vote': ['forum/GitHub votes'],
    'Off-chain Signatures': ['UI/type/function signature'],
}
ALIASES = {
    'Proxy contract': ['upgradeable proxy', 'upgradeability proxy'],
    'Role-based control': ['RBAC', 'role based access control'],
    'Emergency stop': ['circuit breaker'],
    'Rate limit': ['speed bump'],
    'Check-Effects-Interactions': ['checks effects interactions', 'checks-effects-interactions'],
    'Off-chain Signatures': ['offchain signatures'],
    'Zero-knowledge proof': ['zero knowledge proof', 'zk proof'],
    'Off-chain data storage': ['offchain storage'],
    'Commit and Reveal': ['commit reveal', 'commit-reveal'],
}
_STOP = set('a an the and or of to for in on by is as with from this that it be are de do da dos das e o os um uma para por em com que no na ao sua seu'.split())


def terms(text: str) -> list[str]:
    # Split camelCase identifiers but retain exact forms as additional terms.
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if not unicodedata.combining(c))
    return [word for word in re.findall(r'[a-z0-9]+', text) if len(word) > 1 and word not in _STOP]


@lru_cache(maxsize=4096)
def _phrase_regex(phrase):
    return re.compile(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', re.IGNORECASE)


def phrase_match(phrase: str, text: str) -> bool:
    phrase = phrase.strip()
    return bool(phrase and _phrase_regex(phrase).search(text))


@dataclass(frozen=True)
class CompactPattern:
    pattern_id: str
    canonical_name: str
    category: str
    short_definition: str
    distinctive_mechanisms: tuple[str, ...]
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]
    false_friends: tuple[str, ...]
    overlap_family: tuple[str, ...]

    def prompt_line(self):
        return f'{self.pattern_id} {self.canonical_name}: {self.short_definition}'


class CompactCatalog:
    def __init__(self, catalog: PatternCatalog):
        resource = json.loads(Path(__file__).with_name('compact_catalog_v1.json').read_text())
        self.version = resource['version']
        self.records = {}
        rows = {row['pattern']: row for row in catalog.dataframe.to_dict('records')}
        for name in catalog.names:
            source = catalog.by_name[name]
            entry = resource['patterns'].get(name)
            # Never use a curated description against a changed full definition.
            matched = entry and entry['source_description_sha256'] == sha256_text(source['description'])
            definition = entry['short_definition'] if matched else source['description']
            pattern_id = entry['pattern_id'] if entry else 'PX' + sha256_text(name)[:12]
            families = tuple(f'F{i:02}' for i, group in enumerate(KNOWN_OVERLAP_GROUPS) if name in group)
            self.records[name] = CompactPattern(
                pattern_id, name, source['category'], definition, (definition,),
                tuple(ALIASES.get(name, [])),
                tuple(x.strip() for x in str(rows[name].get('keywords', '')).split(';') if x.strip()),
                tuple(FALSE_FRIENDS.get(name, [])), families,
            )
        self.id_to_name = {r.pattern_id: r.canonical_name for r in self.records.values()}
        if len(self.id_to_name) != len(self.records):
            raise ValueError('Duplicate compact pattern IDs')
        self.fingerprint = sha256_text(json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False))

    def to_dict(self):
        return {'version': self.version, 'patterns': [asdict(r) for r in self.records.values()]}

    def canonicalize(self, value: str):
        if value in self.id_to_name:
            return self.id_to_name[value]
        if value in self.records:
            return value
        raise ValueError(f'Unknown pattern ID/name: {value!r}')

    def false_friends_context(self, names):
        return '\n'.join(f'{name}: ' + '; '.join(self.records[name].false_friends)
                         for name in names if self.records[name].false_friends)


class LatentSemanticIndex:
    """TF-IDF projected into a shared low-rank co-occurrence space (LSA)."""
    def __init__(self, documents: list[str], dimensions: int):
        counts = [Counter(terms(doc)) for doc in documents]
        self.vocabulary = {term: i for i, term in enumerate(sorted(set().union(*counts)))}
        n = len(documents)
        self.idf = np.array([math.log((1 + n) / (1 + sum(t in c for c in counts))) + 1
                             for t in self.vocabulary])
        matrix = np.zeros((n, len(self.vocabulary)))
        for i, count in enumerate(counts):
            for term, freq in count.items():
                matrix[i, self.vocabulary[term]] = 1 + math.log(freq)
        matrix *= self.idf
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norm, 1e-12)
        values, vectors = np.linalg.eigh(matrix @ matrix.T)
        indices = [i for i in np.argsort(values)[::-1][:dimensions] if values[i] > 1e-10]
        self.projection = matrix.T @ (vectors[:, indices] / np.sqrt(values[indices]))
        self.documents = matrix @ self.projection
        self.documents /= np.maximum(np.linalg.norm(self.documents, axis=1, keepdims=True), 1e-12)

    def scores(self, text: str) -> list[float]:
        query = np.zeros(len(self.vocabulary))
        for term, count in Counter(terms(text)).items():
            if term in self.vocabulary:
                i = self.vocabulary[term]
                query[i] = (1 + math.log(count)) * self.idf[i]
        latent = query @ self.projection
        latent /= max(float(np.linalg.norm(latent)), 1e-12)
        return [round(float(max(0, min(1, v))), 8) for v in self.documents @ latent]


class HybridRetriever:
    def __init__(self, catalog: PatternCatalog, config: OptimizationConfig):
        self.catalog, self.config = catalog, config
        self.compact = CompactCatalog(catalog)
        self.names = sorted(catalog.names, key=lambda n: self.compact.records[n].pattern_id)
        docs = []
        for name in self.names:
            r = self.compact.records[name]
            # False friends are intentionally excluded from the positive embedding.
            if config.semantic_backend in {'neural', 'lsa_neural'}:
                docs.append(' '.join([name, r.short_definition, *r.distinctive_mechanisms,
                                      *r.aliases, *r.keywords]))
            else:
                # Preserve retrieval-v1 LSA behavior for a valid before/after comparison.
                docs.append(' '.join([name, r.short_definition, catalog.by_name[name]['description'],
                                      *r.aliases, *r.keywords]))
        self.catalog_hash = sha256_text(json.dumps(
            {name: catalog.by_name[name] for name in sorted(catalog.names)},
            sort_keys=True, ensure_ascii=False))
        if config.semantic_backend in {'neural', 'lsa_neural'}:
            from .neural_embeddings import NeuralEmbeddingIndex
            self.semantic = NeuralEmbeddingIndex(
                docs, self.names,
                model_name=config.embedding_model,
                model_revision=config.embedding_model_revision,
                catalog_hash=self.catalog_hash,
                compact_catalog_hash=self.compact.fingerprint,
                cache_dir=config.embedding_cache_dir,
                local_files_only=config.embedding_local_files_only,
                batch_size=config.embedding_batch_size,
                chunk_chars=config.semantic_chunk_chars,
            )
            self.lsa = LatentSemanticIndex([
                ' '.join([name, self.compact.records[name].short_definition,
                          catalog.by_name[name]['description'],
                          *self.compact.records[name].aliases,
                          *self.compact.records[name].keywords])
                for name in self.names
            ], config.semantic_dimensions) if config.semantic_backend == 'lsa_neural' else None
        else:
            self.semantic = LatentSemanticIndex(docs, config.semantic_dimensions)
            self.lsa = None

    def _semantic_scores(self, text):
        if self.config.semantic_backend in {'neural', 'lsa_neural'}:
            values, chunks = self.semantic.scores(text)
            return values, chunks
        return self.semantic.scores(text), []

    def retrieve(self, text: str):
        cfg = self.config
        semantic_values, semantic_chunks = self._semantic_scores(text)
        semantic = dict(zip(self.names, semantic_values))
        lsa = dict(zip(self.names, self.lsa.scores(text))) if self.lsa is not None else {}
        lexical = {}
        direct = set()
        for name in self.names:
            record = self.compact.records[name]
            matches = [p for p in (name, *record.aliases, *record.keywords) if phrase_match(p, text)]
            if matches:
                lexical[name] = matches
            if any(phrase_match(p, text) for p in (name, *record.aliases)):
                direct.add(name)
        ranked = sorted(self.names, key=lambda n: (-semantic[n], self.compact.records[n].pattern_id))
        selected = set(lexical)  # All direct/keyword matches survive regardless of semantic ranking.
        selection_reasons = {name: ['lexical'] for name in lexical}
        for name in ranked[:min(cfg.retrieval_min_candidates, len(ranked))]:
            selected.add(name)
            selection_reasons.setdefault(name, []).append('semantic_minimum')
        for name in ranked[:cfg.retrieval_max_candidates]:
            if semantic[name] >= cfg.retrieval_similarity_threshold:
                selected.add(name)
                selection_reasons.setdefault(name, []).append('semantic_threshold')
        if lsa:
            ranked_lsa = sorted(self.names, key=lambda n: (-lsa[n], self.compact.records[n].pattern_id))
            for name in ranked_lsa[:min(cfg.retrieval_min_candidates, len(ranked_lsa))]:
                selected.add(name)
                selection_reasons.setdefault(name, []).append('lsa_minimum')
            for name in ranked_lsa[:cfg.retrieval_max_candidates]:
                if lsa[name] >= cfg.retrieval_similarity_threshold:
                    selected.add(name)
                    selection_reasons.setdefault(name, []).append('lsa_threshold')
        reasons = []
        top = semantic[ranked[0]] if ranked else 0
        weak = top < cfg.retrieval_confidence_threshold
        unanchored = cfg.retrieval_require_lexical_anchor and not lexical
        ambiguous = sum(v >= max(cfg.retrieval_similarity_threshold / 2, top * cfg.retrieval_ambiguity_ratio)
                        for v in semantic.values()) > cfg.retrieval_max_candidates
        if not cfg.retrieval_enabled or cfg.force_full_catalog:
            selected = set(self.names)
            reasons.append('disabled' if not cfg.retrieval_enabled else 'forced_full_catalog')
        elif weak or ambiguous or unanchored:
            reasons.extend(reason for flag, reason in ((weak, 'low_confidence'),
                (ambiguous, 'ambiguous_similarity'), (unanchored, 'no_lexical_anchor')) if flag)
            if cfg.full_catalog_fallback:
                selected = set(self.names)
            else:
                for name in ranked[:min(len(ranked), cfg.retrieval_fallback_multiplier * cfg.retrieval_max_candidates)]:
                    selected.add(name)
                    selection_reasons.setdefault(name, []).append('expanded_fallback')
        pre_family = set(selected)
        family_added = set()
        if cfg.retrieval_family_expansion:
            # Expand all retrieved families, even when members have weak scores.
            for group in KNOWN_OVERLAP_GROUPS:
                if selected.intersection(group):
                    additions = {n for n in group if n in self.compact.records} - selected
                    selected.update(additions)
                    family_added.update(additions)
                    for name in additions:
                        selection_reasons.setdefault(name, []).append('family_expansion')
        names = [n for n in self.names if n in selected]
        full_fallback = bool(reasons) and cfg.full_catalog_fallback and len(pre_family) == len(self.names)
        fallback_added = len(pre_family - set(lexical)) if full_fallback else 0
        backend = cfg.semantic_backend
        return {
            'retrieval_candidates': names,
            'retrieval_candidate_count': len(names),
            'retrieval_method': f'lexical+{backend}+family',
            'retrieval_scores': {n: {'semantic': semantic[n], 'lexical_matches': lexical.get(n, []),
                                     'direct_match': n in direct,
                                     'lsa': lsa.get(n),
                                     'semantic_rank': ranked.index(n) + 1,
                                     'selection_reasons': selection_reasons.get(n, [])}
                                 for n in self.names},
            'retrieval_fallback_used': bool(reasons),
            'retrieval_full_catalog_fallback': full_fallback,
            'retrieval_fallback_reasons': reasons,
            'fallback_patterns_added': fallback_added,
            'family_patterns_added': len(family_added),
            'family_added_patterns': sorted(family_added),
            'lexical_match_count': len(lexical),
            'semantic_query_chunks': semantic_chunks,
            'semantic_backend': backend,
            'semantic_index_fingerprint': getattr(self.semantic, 'fingerprint', None),
            'semantic_index_cache_hit': getattr(self.semantic, 'cache_hit', None),
            'embedding_model': getattr(self.semantic, 'model_name', None),
            'embedding_model_revision': getattr(self.semantic, 'model_revision', None),
            'embedding_model_artifact_sha256': getattr(self.semantic, 'model_artifact_sha256', None),
            'retrieval_config_sha256': cfg.fingerprint,
            'compact_catalog_sha256': self.compact.fingerprint,
            'source_text_sha256': sha256_text(text),
        }
