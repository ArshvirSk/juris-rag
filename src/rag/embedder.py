"""Embedding layer — sentence-transformers MiniLM.

Document and query embedding share one model instance. all-MiniLM-L6-v2 is
a symmetric bi-encoder, so no instruction prefix is used on queries.
Normalized embeddings → inner product = cosine similarity.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:
    """Lazy singleton so imports stay cheap for tests that don't embed."""

    _instance: "Embedder | None" = None

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL_NAME)

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = Embedder()
        return cls._instance

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
