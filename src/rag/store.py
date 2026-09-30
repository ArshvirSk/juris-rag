"""ChromaDB persistent vector store.

Collections are named per embedding model so switching models never mixes
vector spaces. Chunk metadata carries full provenance for citations.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from src.rag.chunking import Chunk

DB_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "panch_docs_minilm_l6_v2"


def embedding_text(chunk: Chunk) -> str:
    """Contextual chunk header: prefix source+section before embedding.

    Table rows and terse bullet fragments embed poorly without context;
    prepending the section header grounds them (cheap version of contextual
    retrieval). The stored document stays the raw chunk text.
    """
    return f"{chunk.source} — {chunk.section_title}\n{chunk.text}"


def get_collection(reset: bool = False):
    """Return (creating if needed) the persistent chunk collection."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass  # nothing to reset
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk], embedder, batch_size: int = 64, reset: bool = False) -> int:
    """Embed and upsert chunks into the store (idempotent via content-hash IDs).

    ``reset=True`` drops the collection first for a clean rebuild.
    Returns the total collection size after indexing.
    """
    collection = get_collection(reset=reset)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embedder.embed_documents([embedding_text(c) for c in batch])
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors.tolist(),
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "source": c.source,
                    "section_title": c.section_title,
                    "heading_offset": c.heading_offset,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                for c in batch
            ],
        )
    return collection.count()
