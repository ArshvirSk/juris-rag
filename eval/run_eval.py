"""juris-rag evaluation runner.

Metrics (per tests/golden/qa_golden.json):
  - hit-rate@5: fraction of questions whose gold section appears in the
    retrieved top-5 (retrieval quality). Reported with AND without the
    cross-encoder reranker (ablation).
  - citation faithfulness (live, needs GOOGLE_API_KEY): every [n] cited in
    the answer must exist, and keyphrases for answerable questions should
    appear in the answer; unanswerable questions must be refused.

Usage:
    python -m eval.run_eval                 # retrieval metrics only (no API)
    python -m eval.run_eval --live          # + generation faithfulness
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252; corpus text contains emoji/unicode.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "tests" / "golden" / "qa_golden.json"


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]


def _hit(retrieved, gold_sections: list[list[str]]) -> bool:
    gold_pairs = {tuple(g) for g in gold_sections}
    return any((r.chunk.source, r.chunk.section_title) in gold_pairs for r in retrieved)


def eval_retrieval(questions: list[dict], k: int = 5) -> dict:
    """Run retrieval over the golden set with and without reranking."""
    from src.rag.retriever import retrieve

    results = {"rerank": {"hits": 0, "per_q": {}}, "no_rerank": {"hits": 0, "per_q": {}}}
    t0 = time.time()

    for q in questions:
        if q.get("unanswerable"):
            continue  # retrieval metrics apply to answerable questions only

        reranked = retrieve(q["question"], final_k=k, rerank=True)
        plain = retrieve(q["question"], final_k=k, rerank=False)

        hit_r = _hit(reranked, q["gold_sections"])
        hit_p = _hit(plain, q["gold_sections"])
        results["rerank"]["hits"] += hit_r
        results["no_rerank"]["hits"] += hit_p
        results["rerank"]["per_q"][q["id"]] = hit_r
        results["no_rerank"]["per_q"][q["id"]] = hit_p
        print(f"  {q['id']}: rerank={'HIT ' if hit_r else 'miss'} | "
              f"no-rerank={'HIT ' if hit_p else 'miss'} | top1={reranked[0].chunk.citation}")

    n_answerable = sum(1 for q in questions if not q.get("unanswerable"))
    results["n_answerable"] = n_answerable
    results["rerank"]["hit_rate"] = results["rerank"]["hits"] / n_answerable
    results["no_rerank"]["hit_rate"] = results["no_rerank"]["hits"] / n_answerable
    results["elapsed_s"] = round(time.time() - t0, 1)
    return results


def eval_generation(questions: list[dict], k: int = 5) -> dict:
    """Live generation eval: refusals for unanswerable, keyphrase coverage
    and citation sanity for answerable questions. Requires GOOGLE_API_KEY."""
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY required for --live")

    from src.rag.answer import generate_answer
    from src.rag.retriever import retrieve

    stats = {
        "answered": 0, "refused": 0, "refusals_correct": 0,
        "keyphrase_hits": 0, "keyphrase_total": 0,
        "citations_valid": 0, "answers_with_citations": 0, "n_answerable": 0,
        "per_q": {},
    }

    for q in questions:
        retrieved = retrieve(q["question"], final_k=k, rerank=True)
        answer = generate_answer(q["question"], retrieved)
        entry: dict = {"citations": answer.cited_numbers}

        lowered = answer.text.lower()
        is_refusal = "don't know" in lowered or "don't cover this" in lowered

        if q.get("unanswerable"):
            stats["refused"] += 1
            ok = is_refusal
            stats["refusals_correct"] += ok
            entry["result"] = "refused-correct" if ok else "HALLUCINATED"
        else:
            stats["n_answerable"] += 1
            if is_refusal:
                entry["result"] = "wrongly-refused"
            else:
                stats["answered"] += 1
                entry["result"] = "answered"
                stats["keyphrase_total"] += len(q["keyphrases"])
                for phrase in q["keyphrases"]:
                    stats["keyphrase_hits"] += phrase.lower() in lowered
                if answer.cited_numbers:
                    stats["answers_with_citations"] += 1
                    if all(1 <= n <= answer.evidence_count for n in answer.cited_numbers):
                        stats["citations_valid"] += 1

        stats["per_q"][q["id"]] = entry
        marker = entry["result"]
        print(f"  {q['id']}: {marker}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="also run generation faithfulness (needs GOOGLE_API_KEY)")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    questions = load_golden()
    print(f"juris-rag eval — {len(questions)} golden questions "
          f"({sum(1 for q in questions if q.get('unanswerable'))} unanswerable)\n")

    print("== Retrieval (hit-rate@5) ==")
    retrieval = eval_retrieval(questions, k=args.k)
    print(f"\n  with rerank   : {retrieval['rerank']['hits']}/{retrieval['n_answerable']} "
          f"= {retrieval['rerank']['hit_rate']:.2%}")
    print(f"  without rerank: {retrieval['no_rerank']['hits']}/{retrieval['n_answerable']} "
          f"= {retrieval['no_rerank']['hit_rate']:.2%}")
    print(f"  elapsed: {retrieval['elapsed_s']}s")

    if args.live:
        print("\n== Generation (live Gemini) ==")
        gen = eval_generation(questions, k=args.k)
        print(f"\n  answered: {gen['answered']} | correct refusals: "
              f"{gen['refusals_correct']}/{gen['refused']}")
        if gen["keyphrase_total"]:
            print(f"  keyphrase coverage: {gen['keyphrase_hits']}/{gen['keyphrase_total']}")
        print(f"  answers with citations: {gen['answers_with_citations']}/{gen['n_answerable']}"
              f" | citations in range: {gen['citations_valid']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
