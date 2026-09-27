from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_pipeline.data import load_patterns
from llm_pipeline.human_annotations import human_component, load_human_gold
from llm_pipeline.neural_embeddings import NeuralBackendUnavailable, NeuralEmbeddingIndex, chunk_text
from llm_pipeline.optimization import OptimizationConfig
from llm_pipeline.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def catalog():
    return load_patterns(ROOT / 'blockchain_patterns_keywords_v3.csv')


def test_human_component_never_uses_llm_when_a1_is_present():
    assert human_component('llm: yes; A1: no') == 'no'
    assert human_component('llm: Factory contract; A1: ') == ''
    assert human_component('yes') == 'yes'


def test_human_gold_strict_and_uncertain_are_separate_and_source_is_unchanged(tmp_path, catalog):
    path = tmp_path / 'human.csv'
    frame = pd.DataFrame([
        {'repository':'r/a','issue_number':'1','relevant_to_pattern_study':'llm: no; A1: yes',
         'patterns_confirmed':'Oracle','patterns_uncertain':'','patterns_present':'Oracle',
         'insufficient_context':'no','confidence':'high','annotator_id':'A1'},
        {'repository':'r/a','issue_number':'2','relevant_to_pattern_study':'llm: yes; A1: uncertain',
         'patterns_confirmed':'llm: Check-Effects-Interactions; A1: ',
         'patterns_uncertain':'llm: ; A1: Check-Effects-Interactions',
         'patterns_present':'Check-Effects-Interactions','insufficient_context':'no',
         'confidence':'medium','annotator_id':'A1'},
        {'repository':'r/a','issue_number':'3','relevant_to_pattern_study':'llm: yes; A1: no',
         'patterns_confirmed':'llm: Tokenization; A1: ','patterns_uncertain':'',
         'patterns_present':'llm: Tokenization; A1: ','insufficient_context':'no',
         'confidence':'high','annotator_id':'A1'},
    ])
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    gold = load_human_gold(path, catalog)
    assert gold.strict_pairs == (('r/a','1','Oracle'),)
    assert gold.uncertain_pairs == (('r/a','2','Check-Effects-Interactions'),)
    assert path.read_bytes() == before
    assert not gold.summary['provenance']['independent_gold_standard']


def test_human_gold_rejects_unknown_pattern_without_silent_mapping(tmp_path, catalog):
    path = tmp_path / 'human.csv'
    pd.DataFrame([{'repository':'r','issue_number':'1','relevant_to_pattern_study':'yes',
                   'patterns_confirmed':'Made Up Pattern'}]).to_csv(path,index=False)
    with pytest.raises(ValueError, match='Made Up Pattern'):
        load_human_gold(path, catalog)


class FakeModel:
    def __init__(self):
        self.encoded = []

    def eval(self):
        return self

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        rows = []
        for text in texts:
            value = sum(map(ord, text))
            vector = np.array([value % 7 + 1, value % 11 + 1, value % 13 + 1], dtype=np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows)


def test_neural_index_cache_fingerprint_and_deterministic_ranking(tmp_path):
    models = []
    def loader(*args):
        model = FakeModel(); models.append(model); return model
    kwargs = dict(model_name='test/model', model_revision='abc', catalog_hash='catalog',
                  compact_catalog_hash='compact', cache_dir=tmp_path, local_files_only=True,
                  batch_size=2, chunk_chars=80, model_loader=loader)
    first = NeuralEmbeddingIndex(['oracle data','proxy upgrade'], ['Oracle','Proxy contract'], **kwargs)
    scores1, chunks1 = first.scores('oracle information from outside')
    second = NeuralEmbeddingIndex(['changed but cache must win','ignored'], ['Oracle','Proxy contract'], **kwargs)
    scores2, chunks2 = second.scores('oracle information from outside')
    assert not first.cache_hit and second.cache_hit
    assert first.fingerprint == second.fingerprint
    assert scores1 == scores2 and chunks1 == chunks2
    assert len(models[1].encoded) == 0


def test_neural_index_fingerprint_changes_with_catalog(tmp_path):
    loader = lambda *args: FakeModel()
    common = dict(model_name='m', model_revision='r', compact_catalog_hash='c', cache_dir=tmp_path,
                  local_files_only=True, batch_size=2, chunk_chars=80, model_loader=loader)
    one = NeuralEmbeddingIndex(['one'], ['A'], catalog_hash='one', **common)
    two = NeuralEmbeddingIndex(['one'], ['A'], catalog_hash='two', **common)
    assert one.fingerprint != two.fingerprint


def test_neural_backend_unavailable_is_explicit(tmp_path):
    def unavailable(*args):
        raise NeuralBackendUnavailable('not installed')
    with pytest.raises(NeuralBackendUnavailable, match='not installed'):
        NeuralEmbeddingIndex(['x'], ['X'], model_name='m', model_revision='r', catalog_hash='c',
                             compact_catalog_hash='d', cache_dir=tmp_path, local_files_only=True,
                             batch_size=1, chunk_chars=80, model_loader=unavailable)


def test_chunking_preserves_cross_language_text_and_offsets():
    text = 'English oracle request.\nDescrição portuguesa do oráculo. ' * 8
    chunks = chunk_text(text, 90)
    assert ''.join(value for _, _, value in chunks) == text
    assert all(text[start:end] == value for start, end, value in chunks)


def test_semantic_backend_config_and_environment(monkeypatch):
    monkeypatch.setenv('SEMANTIC_BACKEND','lsa_neural')
    monkeypatch.setenv('EMBEDDING_LOCAL_FILES_ONLY','false')
    config = OptimizationConfig.from_environment(enabled=True)
    assert config.semantic_backend == 'lsa_neural' and not config.embedding_local_files_only
    with pytest.raises(ValueError, match='SEMANTIC_BACKEND'):
        OptimizationConfig(semantic_backend='unknown')


def test_lexical_lsa_neural_union_and_tie_breaking(monkeypatch, tmp_path, catalog):
    class StubIndex:
        fingerprint = 'stub'
        cache_hit = True
        model_name = 'stub/model'
        model_revision = 'fixed'
        model_artifact_sha256 = 'hash'
        def __init__(self, documents, names, **kwargs):
            self.names = names
        def scores(self, text):
            return [0.0] * len(self.names), []
    monkeypatch.setattr('llm_pipeline.neural_embeddings.NeuralEmbeddingIndex', StubIndex)
    common = dict(enabled=True, retrieval_confidence_threshold=0,
                  retrieval_require_lexical_anchor=False, full_catalog_fallback=False,
                  retrieval_family_expansion=False, embedding_cache_dir=str(tmp_path))
    lsa = HybridRetriever(catalog, OptimizationConfig(**common)).retrieve(
        'Weather measurements submitted externally.')
    union_retriever = HybridRetriever(catalog, OptimizationConfig(**common, semantic_backend='lsa_neural'))
    first = union_retriever.retrieve('Weather measurements submitted externally.')
    second = union_retriever.retrieve('Weather measurements submitted externally.')
    assert set(lsa['retrieval_candidates']) <= set(first['retrieval_candidates'])
    assert first['retrieval_candidates'] == second['retrieval_candidates']
    assert first['semantic_index_fingerprint'] == 'stub'
