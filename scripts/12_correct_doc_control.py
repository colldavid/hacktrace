"""Experiment 1: Correct-document control.

Mirror of Condition 1 in every respect — same TIER3 doc machinery, same venue
(venues[0]) and citation (citations[0]) as doc_a, same bare framing, same
thinking settings — except the document asserts the CORRECT answer.

Prediction from the conflict-detector account of deliberation: with no
conflict between parametric knowledge and document content, thinking traces
should be uniformly short (comparable to the no-document baseline), and
answers should be near-uniformly correct. If traces instead stay long for
the questions that deliberated under misinformation, deliberation is driven
by something other than conflict.

Per question: 1 doc-gen call (cached) + 1 extended-thinking call + 1
correctness-judge call.

Reads:
  data/all_questions_with_documents.json

Produces:
  data/correct_control_results.json   (one row per question)
  data/correct_control_progress.json  (crash recovery)

Args:
  --limit N    Only first N questions (test runs).
  --workers W  Concurrent workers. Default: 20.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tracing import init_tracing
from src.client import call_claude, MODEL
from src.judge import call_correctness_judge
from src.document_gen import (
    generate_academic_doc,
    select_citations_for_question,
    VENUE_POOLS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "correct_control_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "correct_control_progress.json")
SAVE_EVERY = 50

_lock = threading.Lock()


def save_progress(results, tested):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "tested": list(tested)}, f, ensure_ascii=False)


def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return [], set()
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        p = json.load(f)
    return p.get("results", []), set(p.get("tested", []))


def process_one(rec: dict) -> dict:
    question = rec["question"]
    correct = rec["answer"]

    venue = VENUE_POOLS[rec["domain"]][0]
    author, year = select_citations_for_question(question)[0]

    last_err = None
    for attempt in range(3):
        try:
            # Same doc slot as doc_a, but asserting the correct answer
            doc = generate_academic_doc(question, correct, venue, author, year)

            # Bare framing, identical structure to the bare C1 run
            prompt = f"[DOCUMENT 1]\n{doc}\n[/DOCUMENT 1]\n\n{question}"
            result = call_claude(
                messages=[{"role": "user", "content": prompt}],
                cache_key_parts=[question, "correct_control_c1", MODEL],
                max_tokens=16000,
                extended_thinking=True,
                budget_tokens=8000,
            )
            judge = call_correctness_judge(
                question=question,
                correct=correct,
                model_response=result["answer"],
            )
            return {
                "question": question,
                "correct_answer": correct,
                "aliases": rec.get("aliases", []),
                "wrong_answer": rec["wrong_answer"],
                "domain": rec["domain"],
                "doc": doc,
                "doc_contains_correct": correct.strip().lower() in doc.lower(),
                "answer": result["answer"],
                "thinking_trace": result["thinking"],
                "thinking_length": len(result["thinking"]),
                "judge_label": judge["label"],
                "judge_reason": judge["reason"],
                "cached": result["cached"],
            }
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    results, tested = load_progress()
    print(f"Correct-document control: {len(records)} questions | workers: {args.workers}")
    print(f"Resumed: {len(results)} already done")

    tasks = [r for r in records if r["question"] not in tested]
    print(f"Remaining: {len(tasks)}")
    print()

    errors, completed, last_save = 0, 0, len(results)
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, r): r for r in tasks}
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                row = fut.result()
                with _lock:
                    results.append(row)
                    tested.add(rec["question"])
                    completed += 1
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    label = row["judge_label"].upper()[:3]
                    line = (
                        f"  [{completed}/{len(tasks)}] {label:<3} | "
                        f"think={row['thinking_length']:>5}ch | "
                        f"{rec['domain']:<13} | "
                        f"Q: {rec['question'][:50]}"
                    )
                    print(line.encode("ascii", errors="replace").decode(), flush=True)
                    if len(results) - last_save >= SAVE_EVERY:
                        save_progress(results, tested)
                        last_save = len(results)
                        print(f"  --- saved ({len(results)} rows, {rate:.2f}/sec) ---", flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {rec['question'][:50]} -> {e}", flush=True)
                with _lock:
                    save_progress(results, tested)

    save_progress(results, tested)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print("CORRECT-DOCUMENT CONTROL COMPLETE")
    print(f"  Rows: {len(results)}  |  Errors: {errors}  |  {elapsed/60:.1f} min")

    import numpy as np
    if results:
        lengths = [r["thinking_length"] for r in results]
        n = len(results)
        for lbl in ("correct", "incorrect", "unclear"):
            k = sum(1 for r in results if r["judge_label"] == lbl)
            print(f"  {lbl:<10}: {k:4d} ({100*k/n:.1f}%)")
        print(f"  thinking: mean={np.mean(lengths):.0f}  median={np.median(lengths):.0f}  "
              f"p90={np.percentile(lengths, 90):.0f} chars")
        thin = sum(1 for L in lengths if L <= 200)
        print(f"  thin (<=200ch): {thin} ({100*thin/n:.1f}%)")
        bad_docs = sum(1 for r in results if not r["doc_contains_correct"])
        print(f"  docs missing correct answer: {bad_docs}")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
