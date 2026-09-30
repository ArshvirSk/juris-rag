"""Offline tests for juris-rag (no LLM calls)."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---- Chunker ----

class TestChunking:
    def test_chunks_respect_target_size(self):
        from src.rag.chunking import split_recursive

        text = "paragraph one.\n\n" * 400  # ~5200 chars of repetitive prose
        chunks = split_recursive(text, source="t", section_title="S")
        assert len(chunks) > 1
        assert all(len(c.text) <= 2200 for c in chunks)  # target + slack

    def test_provenance_fields_present(self):
        from src.rag.chunking import split_recursive

        chunks = split_recursive("hello world content", source="doc",
                                 section_title="Intro", heading_offset=100)
        c = chunks[0]
        assert c.source == "doc"
        assert c.section_title == "Intro"
        assert c.heading_offset == 100
        assert c.char_start >= 100
        assert c.chunk_id and len(c.chunk_id) == 16

    def test_chunk_ids_are_content_stable(self):
        from src.rag.chunking import split_recursive

        a = split_recursive("same text", source="d", section_title="S")
        b = split_recursive("same text", source="d", section_title="S")
        assert a[0].chunk_id == b[0].chunk_id


class TestLoaders:
    def test_corpus_loads(self):
        from src.rag.loaders import load_corpus_documents

        docs = load_corpus_documents()
        assert len(docs) >= 10
        assert all(name and text for name, text in docs)

    def test_heading_splits_cover_document(self):
        from src.rag.loaders import load_corpus_documents, heading_aware_splits

        source, text = load_corpus_documents()[0]
        spans = heading_aware_splits(text)
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)
        # Contiguity
        for (_, end_a, _), (start_b, _, _) in zip(spans, spans[1:]):
            assert start_b == end_a

    def test_build_document_chunks_returns_provenance(self):
        from src.rag.loaders import build_document_chunks

        chunks = build_document_chunks()
        assert len(chunks) > 50
        sources = {c.source for c in chunks}
        assert "Panch_TRD" in sources
        for c in chunks[:5]:
            assert c.section_title
            assert c.text.strip()


# ---- Store + retriever (requires the index; build it if absent) ----

@pytest.fixture(scope="module")
def indexed():
    """Ensure the collection exists (builds on first run)."""
    from src.rag.embedder import Embedder
    from src.rag.loaders import build_document_chunks
    from src.rag.store import get_collection, index_chunks

    collection = get_collection()
    if collection.count() == 0:
        index_chunks(build_document_chunks(), Embedder.get())
    return collection


class TestStore:
    def test_collection_nonempty(self, indexed):
        assert indexed.count() >= 90

    def test_metadata_has_provenance(self, indexed):
        sample = indexed.get(limit=2, include=["metadatas"])
        for meta in sample["metadatas"]:
            assert "source" in meta and "section_title" in meta


class TestRetriever:
    def test_known_fact_retrieval(self, indexed):
        """The SSM/model-IDs fact must surface in top-5."""
        from src.rag.retriever import retrieve

        hits = retrieve("Where are judge model IDs stored for runtime changes?",
                        final_k=5)
        assert hits, "no retrieval results"
        top_sources = {(h.chunk.source, h.chunk.section_title) for h in hits}
        assert ("PANCH_AGENTS", "Working rules") in top_sources

    def test_rerank_changes_ordering_or_not_but_returns_k(self, indexed):
        from src.rag.retriever import retrieve

        q = "What schema must judge outputs conform to?"
        with_rerank = retrieve(q, final_k=3, rerank=True)
        without = retrieve(q, final_k=3, rerank=False)
        assert len(with_rerank) == 3 and len(without) == 3

    def test_unanswerable_topic_returns_something_rational(self, indexed):
        """For out-of-corpus questions retrieval still returns chunks; the
        GENERATOR is responsible for refusing (tested live)."""
        from src.rag.retriever import retrieve

        hits = retrieve("What is PANCH's token economics for 2027?", final_k=3)
        assert len(hits) == 3
