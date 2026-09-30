"""Document loaders for the juris-rag corpus.

Extracts (source_name, text) pairs from the markdown corpus in
``data/corpus``. Markdown-aware heading splitting gives the chunker
structural anchors so sections never straddle unrelated topics.
"""

from __future__ import annotations

from pathlib import Path

from src.rag.chunking import Chunk

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


def load_corpus_documents(corpus_dir: Path | None = None) -> list[tuple[str, str]]:
    """Load all corpus documents as (source_name, text) tuples.

    Sorted for deterministic ordering across runs (stable chunk IDs).
    """
    directory = Path(corpus_dir or CORPUS_DIR)
    docs: list[tuple[str, str]] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append((path.stem, text))
    if not docs:
        raise FileNotFoundError(
            f"No markdown documents found in {directory} — add the PANCH docs first"
        )
    return docs


def heading_aware_splits(text: str) -> list[tuple[int, int, str]]:
    """Split a markdown document at (at most) second-level headings.

    Returns (start_offset, end_offset, section_title) triples covering the
    whole document. Headings deeper than ## stay inside their parent section
    (the recursive splitter handles them by size). Falls back to a single
    whole-document span when no headings exist.
    """
    if not text.startswith("#"):
        text = "# Document\n" + text
        offset_shift = len("# Document\n")
    else:
        offset_shift = 0

    lines = text.splitlines(keepends=True)
    marks: list[tuple[int, str]] = []
    pos = 0
    for line in lines:
        if line.startswith("## ") or (not marks and line.startswith("# ")):
            marks.append((pos, line.lstrip("# ").strip()))
        pos += len(line)

    if not marks:
        return [(0, len(text), "Document")]

    spans: list[tuple[int, int, str]] = []
    for i, (start, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        spans.append((start, end, title))
    return [
        (max(0, start - offset_shift), max(0, end - offset_shift), title)
        for (start, end, title) in spans
    ]


def build_document_chunks(corpus_dir: Path | None = None) -> list[Chunk]:
    """Convenience: load docs and return their heading-level sections as
    pseudo-chunks (used by the chunker as top-level units)."""
    from src.rag.chunking import split_recursive

    chunks: list[Chunk] = []
    for source, text in load_corpus_documents(corpus_dir):
        for idx, (start, end, title) in enumerate(heading_aware_splits(text)):
            section_text = text[start:end]
            # Section longer than a handful of chunks gets split recursively
            chunks.extend(
                split_recursive(
                    section_text,
                    source=source,
                    section_title=title,
                    heading_offset=start,
                    global_base=idx,
                )
            )
    return chunks
