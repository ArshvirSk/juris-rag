"""Two-stage retriever: bi-encoder recall → cross-encoder rerank.

Stage A (recall): MiniLM bi-encoder retrieves the top ``RECALL_K`` (20)
chunks from ChromaDB by cosine similarity — fast, approximate.
Stage B (rerank): the ms-marco MiniLM cross-encoder scores each
(query, chunk) pair jointly — slower, far more accurate — and the top
``FINAL_K`` (5) survive. Two-stage retrieval is the standard
recall-then-precision pattern: the bi-encoder narrows 100+ candidates to
20 cheaply, the cross-encoder spends its compute only where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunking import Chunk
from src.rag.embedder import Embedder

RECALL_K = 20
FINAL_K = 5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class Retrieved:
    chunk: Chunk
    bi_score: float        # cosine sim from stage A
    rerank_score: float    # cross-encoder logit from stage B


class CrossEncoderReranker:
    """Lazy singleton around the ms-marco cross-encoder."""

    _instance: "CrossEncoderReranker | None" = None

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(RERANKER_MODEL)

    @classmethod
    def get(cls) -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = CrossEncoderReranker()
        return cls._instance

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


def retrieve(
    query: str,
    final_k: int = FINAL_K,
    recall_k: int = RECALL_K,
    rerank: bool = True,
) -> list[Retrieved]:
    """Retrieve the top ``final_k`` chunks for ``query``.

    With ``rerank=False`` (or when fewer than ``recall_k`` chunks exist),
    stage B is skipped and bi-encoder order is returned — used for the
    ablation numbers in the README.
    """
    from src.rag.store import embedding_text, get_collection

    collection = get_collection()
    query_vector = Embedder.get().embed_query(query)

    n_total = collection.count()
    if n_total == 0:
        return []

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(recall_k, n_total),
        include=["metadatas", "documents", "distances"],
    )

    metadatas = result["metadatas"][0]
    documents = result["documents"][0]
    distances = result["distances"][0]

    candidates: list[Retrieved] = []
    for meta, doc, dist in zip(metadatas, documents, distances):
        chunk = Chunk(
            chunk_id="",  # not persisted in metadata; text hash suffices for eval
            source=meta["source"],
            section_title=meta["section_title"],
            heading_offset=meta.get("heading_offset", 0),
            char_start=meta.get("char_start", 0),
            char_end=meta.get("char_end", 0),
            text=doc,
        )
        candidates.append(Retrieved(chunk=chunk, bi_score=1.0 - float(dist), rerank_score=0.0))

    if rerank and len(candidates) > 1:
        # Cross-encoder sees the contextual header too — it disambiguates
        # short table-row chunks exactly as it helps the bi-encoder.
        scores = CrossEncoderReranker.get().score(
            query, [embedding_text(c.chunk) for c in candidates]
        )
        for c, s in zip(candidates, scores):
            c.rerank_score = s
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
    else:
        candidates.sort(key=lambda r: r.bi_score, reverse=True)

    return candidates[:final_k]
