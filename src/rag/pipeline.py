"""juris-rag pipeline CLI.

    python -m src.rag.pipeline index          # chunk → embed → upsert
    python -m src.rag.pipeline ask "question" # retrieve → rerank → cite-answer
    python -m src.rag.pipeline stats
"""

from __future__ import annotations

import argparse
import sys
import time

# Windows consoles default to cp1252; corpus text contains emoji/unicode.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cmd_index(reset: bool = False) -> None:
    from src.rag.embedder import Embedder
    from src.rag.loaders import build_document_chunks
    from src.rag.store import index_chunks

    t0 = time.time()
    chunks = build_document_chunks()
    print(f"[index] {len(chunks)} chunks from corpus "
          f"({len(set(c.source for c in chunks))} documents)")

    n = index_chunks(chunks, Embedder.get(), reset=reset)
    print(f"[index] collection size: {n} | elapsed: {time.time() - t0:.1f}s")


def cmd_ask(question: str, final_k: int, no_rerank: bool, no_answer: bool) -> None:
    from src.rag.retriever import retrieve

    t0 = time.time()
    retrieved = retrieve(question, final_k=final_k, rerank=not no_rerank)
    elapsed = time.time() - t0

    if not retrieved:
        print("No results — run `python -m src.rag.pipeline index` first.")
        return

    print(f"[retrieve] top-{len(retrieved)} in {elapsed:.2f}s "
          f"(rerank={'off' if no_rerank else 'on'})")
    for i, r in enumerate(retrieved, start=1):
        score = r.rerank_score if r.rerank_score else r.bi_score
        print(f"  [{i}] {score:+.3f}  {r.chunk.citation}")

    if no_answer:
        return

    from src.rag.answer import generate_answer

    answer = generate_answer(question, retrieved)
    print(f"\n[answer | {answer.model}]")
    print(answer.text)
    print(f"\ncitations used: {answer.cited_numbers} of {answer.evidence_count} blocks")


def cmd_stats() -> None:
    from src.rag.store import get_collection

    collection = get_collection()
    n = collection.count()
    print(f"collection: {n} chunks")
    if n:
        sample = collection.get(limit=3, include=["metadatas"])
        for meta in sample["metadatas"]:
            print(f"  - {meta['source']} § {meta['section_title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="chunk, embed, and upsert the corpus")
    p_index.add_argument("--reset", action="store_true", help="drop and rebuild the collection")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--k", type=int, default=5, help="final evidence blocks")
    p_ask.add_argument("--no-rerank", action="store_true", help="skip cross-encoder (ablation)")
    p_ask.add_argument("--no-answer", action="store_true", help="retrieval only, no LLM call")

    sub.add_parser("stats", help="collection stats")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(reset=args.reset)
    elif args.command == "ask":
        cmd_ask(args.question, final_k=args.k, no_rerank=args.no_rerank, no_answer=args.no_answer)
    elif args.command == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()
