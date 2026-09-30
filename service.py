"""juris-rag FastAPI service — the container entrypoint.

Endpoints:
    GET  /health          liveness (no model load)
    POST /ask             retrieve → rerank → cite-answer
    POST /ask/retrieval   retrieve → rerank only (no Gemini spend)

Model warm-up happens lazily on first inference; /health stays cheap for
ALB target checks.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="juris-rag", version="1.0.0")


class AskRequest(BaseModel):
    question: str
    k: int = 5


class RetrievedChunk(BaseModel):
    citation: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str | None
    citations: list[int] | None
    evidence: list[RetrievedChunk]
    latency_ms: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask/retrieval")
def ask_retrieval(req: AskRequest) -> dict:
    from src.rag.retriever import retrieve

    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    import time

    t0 = time.perf_counter()
    hits = retrieve(req.question, final_k=req.k, rerank=True)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "question": req.question,
        "evidence": [
            RetrievedChunk(
                citation=h.chunk.citation,
                score=round(h.rerank_score, 4),
                text=h.chunk.text[:600],
            ).model_dump()
            for h in hits
        ],
        "latency_ms": latency_ms,
    }


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    from src.rag.answer import generate_answer
    from src.rag.retriever import retrieve

    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    import time

    t0 = time.perf_counter()
    hits = retrieve(req.question, final_k=req.k, rerank=True)
    if not hits:
        raise HTTPException(status_code=503, detail="index empty — POST /index first")

    answer = generate_answer(req.question, hits)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AskResponse(
        question=req.question,
        answer=answer.text,
        citations=answer.cited_numbers,
        evidence=[
            RetrievedChunk(
                citation=h.chunk.citation,
                score=round(h.rerank_score, 4),
                text=h.chunk.text[:600],
            )
            for h in hits
        ],
        latency_ms=latency_ms,
    )
