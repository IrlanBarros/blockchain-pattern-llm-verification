"""Optional, deterministic local neural embedding index.

The module deliberately imports sentence-transformers lazily. Environments that
only use the default LSA backend therefore do not need the optional dependency.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import random
from typing import Callable

import numpy as np

from .utils import sha256_file, sha256_text, write_json

NORMALIZATION_VERSION = 'neural-text-v1-e5-prefix-max-chunk'
_QUERY_CACHE: dict[tuple[str, str], tuple[list[float], list[dict]]] = {}


class NeuralBackendUnavailable(RuntimeError):
    """Raised when a neural backend was explicitly requested but cannot load."""


@lru_cache(maxsize=2)
def _default_model_loader(model_name: str, revision: str, local_files_only: bool):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise NeuralBackendUnavailable(
            'Neural retrieval requires the optional requirements-neural.txt dependencies'
        ) from exc
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    source = model_name
    if local_files_only:
        try:
            source = snapshot_download(model_name, revision=revision, local_files_only=True)
        except Exception as exc:
            raise NeuralBackendUnavailable(
                f'Pinned model is not fully cached: {model_name}@{revision}'
            ) from exc
    model = SentenceTransformer(
        source,
        revision=revision,
        device='cpu',
        local_files_only=local_files_only,
    )
    model.eval()
    return model


def _model_artifact_hash(model_name: str, revision: str) -> str:
    """Hash the pinned weights/config when Hugging Face exposes cached files."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return 'unavailable'
    paths = []
    for filename in ('model.safetensors', 'config.json', 'modules.json', 'tokenizer.json'):
        value = try_to_load_from_cache(model_name, filename, revision=revision)
        if isinstance(value, str) and Path(value).is_file():
            paths.append((filename, Path(value)))
    if not paths:
        return 'unavailable'
    digest = hashlib.sha256()
    for filename, path in paths:
        digest.update(filename.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def chunk_text(text: str, max_chars: int) -> list[tuple[int, int, str]]:
    """Split without summarization and preserve offsets for diagnostics."""
    if not text:
        return [(0, 0, '')]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = max(text.rfind('\n', start + max_chars // 2, end),
                      text.rfind(' ', start + max_chars // 2, end))
            if cut > start:
                end = cut + 1
        chunks.append((start, end, text[start:end]))
        start = end
    return chunks


class NeuralEmbeddingIndex:
    """Cached catalog embeddings plus max-over-chunks issue scoring."""

    def __init__(
        self,
        documents: list[str],
        names: list[str],
        *,
        model_name: str,
        model_revision: str,
        catalog_hash: str,
        compact_catalog_hash: str,
        cache_dir: str | Path,
        local_files_only: bool,
        batch_size: int,
        chunk_chars: int,
        model_loader: Callable = _default_model_loader,
    ):
        self.names = list(names)
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size = batch_size
        self.chunk_chars = chunk_chars
        try:
            self.model = model_loader(model_name, model_revision, local_files_only)
        except NeuralBackendUnavailable:
            raise
        except Exception as exc:
            raise NeuralBackendUnavailable(
                f'Cannot load pinned embedding model {model_name}@{model_revision}: {exc}'
            ) from exc
        self.model_artifact_sha256 = _model_artifact_hash(model_name, model_revision)
        fingerprint_payload = {
            'embedding_model': model_name,
            'model_revision': model_revision,
            'model_artifact_sha256': self.model_artifact_sha256,
            'catalog_hash': catalog_hash,
            'compact_catalog_hash': compact_catalog_hash,
            'normalization_version': NORMALIZATION_VERSION,
        }
        self.fingerprint = sha256_text(json.dumps(fingerprint_payload, sort_keys=True))
        self.metadata = {**fingerprint_payload, 'fingerprint': self.fingerprint,
                         'dimension': int(self.model.get_sentence_embedding_dimension())}
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.cache_path = root / f'{self.fingerprint}.npz'
        self.metadata_path = root / f'{self.fingerprint}.json'
        self.cache_hit = False
        if self.cache_path.exists() and self.metadata_path.exists():
            saved = json.loads(self.metadata_path.read_text(encoding='utf-8'))
            archive = np.load(self.cache_path, allow_pickle=False)
            saved_names = archive['names'].astype(str).tolist()
            if saved == self.metadata and saved_names == self.names:
                self.documents = archive['embeddings'].astype(np.float32)
                self.cache_hit = True
            else:
                self.documents = self._build(documents)
        else:
            self.documents = self._build(documents)

    def _encode(self, texts: list[str]) -> np.ndarray:
        values = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(values, dtype=np.float32)

    def _build(self, documents: list[str]) -> np.ndarray:
        embeddings = self._encode(['passage: ' + text for text in documents])
        np.savez_compressed(self.cache_path,
                            names=np.asarray(self.names, dtype=str), embeddings=embeddings)
        write_json(self.metadata_path, self.metadata)
        return embeddings

    def scores(self, text: str) -> tuple[list[float], list[dict]]:
        cache_key = (self.fingerprint, sha256_text(text))
        if cache_key in _QUERY_CACHE:
            scores, diagnostics = _QUERY_CACHE[cache_key]
            return list(scores), [dict(item) for item in diagnostics]
        chunks = chunk_text(text, self.chunk_chars)
        queries = self._encode(['query: ' + value for _, _, value in chunks])
        similarities = queries @ self.documents.T
        best_chunk = similarities.argmax(axis=0)
        maxima = similarities.max(axis=0)
        diagnostics = []
        for pattern_index, chunk_index in enumerate(best_chunk.tolist()):
            start, end, _ = chunks[chunk_index]
            diagnostics.append({'pattern': self.names[pattern_index], 'chunk_index': chunk_index,
                                'start': start, 'end': end})
        scores = [round(float(max(-1, min(1, value))), 8) for value in maxima]
        _QUERY_CACHE[cache_key] = (scores, diagnostics)
        return list(scores), [dict(item) for item in diagnostics]
