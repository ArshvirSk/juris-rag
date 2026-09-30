"""Answer generation — Gemini with mandatory inline citations.

Contract with the model:
  - Answer ONLY from the numbered evidence blocks; every factual claim gets
    an inline [n] citation matching the evidence number.
  - If the evidence does not contain the answer, say so explicitly instead
    of guessing (refusal is a correct answer for unanswerable questions).
  - Output is plain text (not JSON) — prose with [n] markers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from src.rag.retriever import Retrieved

ANSWER_MODEL = "gemini-2.0-flash"

_SYSTEM_PROMPT = """You are a precise technical assistant answering questions about the \
PANCH project (an AWS-based AI dispute-resolution platform) using ONLY the numbered \
evidence blocks provided.

Rules:
1. Answer only from the evidence. Cite every factual claim with [n] where n is the \
evidence block number. Multiple claims from one block still get the [n] marker.
2. If the evidence does not contain the answer, reply exactly: \
"I don't know — the indexed documents don't cover this." and nothing else.
3. Do not merge speculation with citations. No citation = no claim.
4. Keep answers under 150 words unless the question needs a list."""


@dataclass
class Answer:
    text: str
    cited_numbers: list[int]
    evidence_count: int
    model: str


def build_evidence_block(retrieved: list[Retrieved]) -> str:
    """Render retrieved chunks as numbered evidence blocks."""
    blocks = []
    for i, r in enumerate(retrieved, start=1):
        blocks.append(f"[{i}] ({r.chunk.citation})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def _extract_citations(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", text)})


def generate_answer(question: str, retrieved: list[Retrieved]) -> Answer:
    """Generate a cited answer via Gemini for the given evidence."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set — needed for answer generation. "
            "Retrieval-only mode works without it."
        )

    client = genai.Client(api_key=api_key)
    evidence = build_evidence_block(retrieved)
    prompt = (
        f"Evidence blocks:\n\n{evidence}\n\nQuestion: {question}\n\n"
        "Answer with inline [n] citations, per the system rules."
    )

    response = client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.1,
        ),
    )

    text = response.text or ""
    return Answer(
        text=text.strip(),
        cited_numbers=_extract_citations(text),
        evidence_count=len(retrieved),
        model=ANSWER_MODEL,
    )
