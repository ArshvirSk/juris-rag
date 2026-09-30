"""Recursive character chunking with section-aware metadata.

Strategy: recursive splitting at ~512 tokens with ~15% overlap. Splitting
priority: paragraph boundaries → line breaks → sentence ends → raw
characters. Every chunk carries provenance (source document, section title,
heading character offset) so citations can point back to a stable location,
and chunk IDs are content-hashed for idempotent re-indexing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# ~512 tokens ≈ 1900-2000 chars for prose; overlap ≈ 15%
TARGET_CHARS = 1900
OVERLAP_CHARS = 290
MIN_CHARS = 220  # fragments smaller than this get merged into neighbors


@dataclass
class Chunk:
    """One retrievable unit with full provenance."""

    chunk_id: str          # content hash — stable across re-indexing
    source: str            # e.g. "Panch_TRD"
    section_title: str     # markdown heading the chunk lives under
    heading_offset: int    # char offset of the section heading in the source
    char_start: int        # char offset of this chunk within the source
    char_end: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} § {self.section_title}"


_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_by_paragraphs(text: str) -> list[str]:
    parts = [p for p in _PARA_RE.split(text) if p and p.strip()]
    return parts or ([text] if text.strip() else [])


def _split_by_lines(text: str) -> list[str]:
    parts = [p for p in text.split("\n") if p.strip()]
    return parts or [text]


def _split_by_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_RE.split(text) if p.strip()]
    return parts or [text]


def _split_by_chars(text: str, target: int, overlap: int) -> list[str]:
    step = max(1, target - overlap)
    return [text[i:i + target] for i in range(0, len(text), step)]


def _recursive_split(text: str, target: int = TARGET_CHARS) -> list[str]:
    """Split ``text`` into pieces ≤ ~target chars, preferring semantic breaks."""
    if len(text) <= target:
        return [text]

    for splitter in (_split_by_paragraphs, _split_by_lines, _split_by_sentences):
        pieces = splitter(text)
        if len(pieces) > 1:
            break
    else:
        pieces = _split_by_chars(text, target, OVERLAP_CHARS)

    # Greedy pack pieces into windows ≤ target, with tail-overlap between
    # consecutive windows so answers spanning a boundary stay retrievable.
    windows: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) + 1 <= target or not current:
            current = (current + "\n" + piece).strip() if current else piece
        else:
            windows.append(current)
            tail = current[-OVERLAP_CHARS:] if OVERLAP_CHARS else ""
            current = (tail + "\n" + piece).strip()
    if current:
        windows.append(current)

    # Merge any tiny trailing fragment into its predecessor
    if len(windows) > 1 and len(windows[-1]) < MIN_CHARS:
        windows[-2] = windows[-2] + "\n" + windows[-1]
        windows.pop()
    return windows


def _chunk_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}::{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def split_recursive(
    section_text: str,
    source: str,
    section_title: str,
    heading_offset: int = 0,
    global_base: int = 0,
) -> list[Chunk]:
    """Split one heading-level section into overlapping chunks with provenance.

    ``global_base`` disambiguates sections that share a title within a doc.
    ``heading_offset`` is the section's char offset in the source document
    (passed through for citation pointers).
    """
    windows = _recursive_split(section_text)
    chunks: list[Chunk] = []
    consumed = 0
    for i, window in enumerate(windows):
        # Locate the window inside the section for a real char_start.
        found = section_text.find(window[:80], consumed)
        start = found if found != -1 else consumed
        consumed = start + len(window)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(f"{source}#{global_base}#{i}", window),
                source=source,
                section_title=section_title,
                heading_offset=heading_offset,
                char_start=heading_offset + start,
                char_end=heading_offset + start + len(window),
                text=window,
            )
        )
    return chunks
