"""Step 11: Baseline thinking-length run — no documents, raw question only.

For each question in data/all_questions_with_documents.json, makes N extended-
thinking calls with just the bare question as the prompt (no preamble, no
document context). Records thinking length and answer per call.

Purpose: establish each question's natural deliberation depth without document
pressure. Used to predict thinness in the main pipeline independently of
question surface features.

Reads:
  data/all_questions_with_documents.json

Produces:
  data/baseline_results.json    (flat list, one row per call)
  data/baseline_progress.json   (crash recovery)

Args:
  --calls N    Calls per question. Default: 3.
  --workers W  Concurrent workers. Default: 20.
  --limit N    Only first N questions (test runs).
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH  = os.path.join(DATA_DIR, "all_questions_with_documents.json")
OUTPUT_PATH  = os.path.join(DATA_DIR, "baseline_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "baseline_progress.json")
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
    return p.get("results", []), set(tuple(t) for t in p.get("tested", []))


def process_one(task):
    last_err = None
    for attempt in range(3):
        try:
            result = call_claude(
                messages=[{"role": "user", "content": task["question"]}],
                cache_key_parts=[task["question"], f"baseline_r{task['run']}", MODEL],
                max_tokens=16000,
                extended_thinking=True,
                budget_tokens=8000,
            )
            return {
                "question":       task["question"],
                "correct_answer": task["correct"],
                "domain":         task["domain"],
                "run":            task["run"],
                "thinking_length": len(result["thinking"]),
                "answer":         result["answer"],
                "cached":         result["cached"],
            }
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls",   type=int, default=3)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    total_calls = len(records) * args.calls
    print(f"Questions: {len(records)}  x  {args.calls} calls = {total_calls} total")
    print(f"Workers: {args.workers}")
    print()

    results, tested = load_progress()
    print(f"Resumed: {len(results)} calls already done")

    tasks = [
        {
            "question": rec["question"],
            "correct":  rec["answer"],
            "domain":   rec["domain"],
            "run":      run,
        }
        for rec in records
        for run in range(1, args.calls + 1)
        if (rec["question"], run) not in tested
    ]
    print(f"Remaining: {len(tasks)} calls")
    print()

    errors, completed, last_save = 0, 0, len(results)
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                row = fut.result()
                with _lock:
                    results.append(row)
                    tested.add((t["question"], t["run"]))
                    completed += 1
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(
                        f"  [{completed}/{len(tasks)}] "
                        f"think={row['thinking_length']:>5}ch | "
                        f"{t['domain']:<13} | "
                        f"Q: {t['question'][:60]}",
                        flush=True
                    )
                    if len(results) - last_save >= SAVE_EVERY:
                        save_progress(results, tested)
                        last_save = len(results)
                        print(f"  --- saved ({len(results)} rows, {rate:.2f}/sec) ---", flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {t['question'][:50]} run{t['run']} -> {e}", flush=True)
                with _lock:
                    save_progress(results, tested)

    save_progress(results, tested)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print(f"BASELINE RUN COMPLETE")
    print(f"  Total rows: {len(results)}")
    print(f"  Errors:     {errors}")
    print(f"  Elapsed:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if completed:
        print(f"  Throughput: {completed/elapsed:.2f} calls/sec")
    print(f"  Saved to:   {OUTPUT_PATH}")

    # Quick summary
    import numpy as np
    from collections import defaultdict
    by_q = defaultdict(list)
    for r in results:
        by_q[r["question"]].append(r["thinking_length"])
    means = [np.mean(v) for v in by_q.values()]
    print()
    print(f"  Thinking length (per-question mean):")
    print(f"    median: {np.median(means):.0f} chars")
    print(f"    mean:   {np.mean(means):.0f} chars")
    print(f"    p25:    {np.percentile(means, 25):.0f} chars")
    print(f"    p75:    {np.percentile(means, 75):.0f} chars")


if __name__ == "__main__":
    main()
