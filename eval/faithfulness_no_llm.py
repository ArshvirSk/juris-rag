"""No-LLM citation-faithfulness eval.

When no GOOGLE_API_KEY is available, this measures the retrieval+evidence
layer the generator depends on: for each answerable question, the top-5
evidence blocks are treated as the extractive answer; the metric checks
gold keyphrase presence **with section attribution** (a keyphrase only
counts if it appears in a block whose section is in the question's gold
set — i.e., cited evidence is the *right* evidence, not noise).

For unanswerable questions, the metric is different: with NO Gemini, the
honest proxy is retrieval-score margin — if the best cross-encoder score
is far below the corpus's typical best-score for answerable questions,
the evidence layer is signaling "nothing relevant here".

Usage:
    python -m eval.faithfulness_no_llm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.run_eval import load_golden  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402


def _keyphrase_in_gold_evidence(question: dict, retrieved, k: int = 5) -> tuple[int, int, list[str]]:
    """Count keyphrases found in gold-section evidence among top-k."""
    gold_pairs = {tuple(g) for g in question["gold_sections"]}
    hits, total = 0, 0
    missing: list[str] = []
    for phrase in question["keyphrases"]:
        total += 1
        found = False
        for r in retrieved[:k]:
            if (r.chunk.source, r.chunk.section_title) in gold_pairs and phrase.lower() in r.chunk.text.lower():
                found = True
                break
        if found:
            hits += 1
        else:
            missing.append(phrase)
    return hits, total, missing


def main() -> int:
    questions = load_golden()
    answerable = [q for q in questions if not q.get("unanswerable")]
    unanswerable = [q for q in questions if q.get("unanswerable")]

    print("=" * 74)
    print("No-LLM citation-faithfulness proxy (extractive answering)")
    print("=" * 74)

    total_phrases = total_found = 0
    per_q: dict[str, str] = {}
    answerable_scores: list[float] = []

    for q in answerable:
        retrieved = retrieve(q["question"], final_k=5, rerank=True)
        answerable_scores.append(max(r.rerank_score for r in retrieved))
        hits, total, missing = _keyphrase_in_gold_evidence(q, retrieved)
        total_found += hits
        total_phrases += total
        per_q[q["id"]] = "PASS" if hits == total and total > 0 else f"partial ({hits}/{total})"
        print(f"  {q['id']}: keyphrases in gold evidence: {hits}/{total}"
              f"{'' if not missing else '  missing: ' + ', '.join(missing)}")

    print("-" * 74)
    print(f"Attributed keyphrase coverage: {total_found}/{total_phrases} "
          f"({total_found / total_phrases:.1%})")

    print("\nUnanswerable questions (score-margin signal, no generator):")
    for q in unanswerable:
        retrieved = retrieve(q["question"], final_k=5, rerank=True)
        best = max(r.rerank_score for r in retrieved)
        margin = best - (min(answerable_scores) if answerable_scores else 0.0)
        signal = "LOW — generator should refuse" if margin < 0 else "ambiguous — generator judgment"
        print(f"  {q['id']}: best rerank score {best:+.3f} vs worst answerable "
              f"{min(answerable_scores):+.3f} → margin {margin:+.3f} → {signal}")

    print("=" * 74)
    print("NOTE: full generation faithfulness (refusal wording, citation [n] usage)")
    print("runs with GOOGLE_API_KEY: python -m eval.run_eval --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
