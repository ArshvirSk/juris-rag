# juris-rag — end-to-end RAG with citations and a measured eval harness

A compact, fully-local RAG pipeline over the engineering docs of [PANCH](https://github.com/ArshvirSk/PANCH) (an AWS-based AI dispute-resolution platform): recursive chunking → MiniLM embeddings → ChromaDB → cross-encoder reranking → Gemini answers with **inline [n] citations**, evaluated on a 20-question golden set with hit-rate@5, a reranker ablation, and faithfulness checks.

**Python · sentence-transformers (bi-encoder + cross-encoder) · ChromaDB · Gemini 2.0 Flash · pytest**

Built to close a specific gap: every other project in my portfolio *uses* embeddings — this one measures the retrieval stack end-to-end.

## Pipeline

```text
data/corpus/*.md (12 PANCH docs)
   │  markdown heading-aware sectioning
   ▼
Recursive chunker          512-token target, ~15% overlap, provenance metadata
   │                       (source, section title, char offsets), content-hash IDs
   ▼
all-MiniLM-L6-v2           contextual chunk header ("source — section") prepended
   │                       at embedding time; L2-normalized → cosine = dot
   ▼
ChromaDB (persistent)      105 chunks, cosine space, idempotent upserts
   │
   ▼  stage A: bi-encoder recall top-20
   ▼  stage B: cross-encoder (ms-marco-MiniLM-L-6-v2) rerank → top-5
   ▼
Gemini 2.0 Flash           answers ONLY from numbered evidence blocks;
                           inline [n] citation per claim; refuses when
                           evidence doesn't cover the question
```

## Evaluation — real numbers

Golden set: 20 questions (18 answerable, 2 deliberately out-of-corpus), each with gold `(document, section)` pairs and answer keyphrases — [`tests/golden/qa_golden.json`](tests/golden/qa_golden.json). Run it yourself: `python -m eval.run_eval` (+ `--live` with a `GOOGLE_API_KEY`).

### Retrieval: hit-rate@5 (gold section in top-5)

| Configuration | hit-rate@5 |
|---|---|
| bi-encoder only | **70.0%** (14/20) |
| + cross-encoder rerank (top-20 → top-5) | **100.0%** (20/20) |

**The reranker is worth +30pp on this corpus.** Typical failure it fixes: the bi-encoder returns topically-adjacent sections (e.g. the PRD's solution overview instead of the tribunal workflow spec); the cross-encoder, scoring query+chunk jointly, pulls the precise section into top-5 (per-question detail: q02, q04, q06, q11, q15 all flip from miss to hit).

### Evidence faithfulness (no-LLM proxy — `eval/faithfulness_no_llm.py`)

**82.4%** of answer keyphrases are findable in top-5 chunks **attributed to gold sections** (i.e. the citation would point at evidence that actually contains the fact). The misses share one cause worth knowing: keyphrases sitting in table rows whose neighboring section (Architecture diagram vs AWS services table) owns the retrieval hit — a chunk-boundary artifact, and the first thing I'd fix with section-overlapping windows or late chunk merging.

### Generation (Gemini)

The generator is contract-bound: answer only from evidence, `[n]`-cite every claim, refuse otherwise (`"I don't know — the indexed documents don't cover this."`). The live runner (`--live`) verifies refusal behavior on the 2 unanswerable questions, keyphrase coverage in answers, and that every cited `[n]` is in range. The API key available at build time had been rotated, so live-generation numbers are pending a fresh key — the harness runs unchanged once one exists.

## What the eval caught (the point of the harness)

1. **Table rows without context embed poorly.** The SSM/model-registry fact lives in a TRD table; before contextual chunk headers it was unretrievable. Fix: prepend `source — section` at embedding time (for both bi- and cross-encoder).
2. **Reranker ablation quantified**, not vibes: 70% → 100%.
3. **Gold-label drift**: several "misses" were actually wrong gold labels (emoji-prefixed README headings), which the eval surfaced — label hygiene is part of the loop too.

## Layout

```text
src/rag/
  loaders.py     # markdown corpus → heading-level sections
  chunking.py    # recursive splitter, provenance, content-hash IDs
  embedder.py    # MiniLM bi-encoder singleton
  store.py       # ChromaDB wrapper, contextual header embedding
  retriever.py   # two-stage recall→rerank
  answer.py      # cited Gemini generation (refusal contract)
  pipeline.py    # CLI: index / ask / stats
eval/
  run_eval.py    # hit-rate@5 + ablation + live generation checks
  faithfulness_no_llm.py
tests/
  test_rag.py    # 11 offline tests (chunker, provenance, store, retrieval)
  golden/qa_golden.json
infra/           # AWS deployment artifacts (see DEPLOY.md)
```

## Run it

```bash
uv venv && uv pip install sentence-transformers chromadb google-genai pypdf python-docx fastapi uvicorn pytest
python -m src.rag.pipeline index --reset
python -m src.rag.pipeline ask "How does the three-judge tribunal work?"
python -m eval.run_eval              # retrieval eval (no API key needed)
python -m eval.run_eval --live       # + generation faithfulness (GOOGLE_API_KEY)
pytest tests/ -q                     # offline unit tests
```

Cloud deployment artifacts (container, IaC, load test, cost note): [`infra/`](infra/) and [`DEPLOY.md`](DEPLOY.md).

## Honest limitations

- 105-chunk corpus — hit-rate@5 at this scale is necessary, not sufficient; the harness (golden set format, ablation, attributed keyphrases) is designed to survive corpus growth.
- Citation [n] correctness is verified structurally (in-range, present) not semantically; a claim-level NLI check is the next iteration.
- Unanswerable-question refusal is enforced by prompt contract + verified by the live eval, not by a trained abstention head.
