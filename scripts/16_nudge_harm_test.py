"""Experiment 4: Nudge harm/cost test on truthful documents.

The rescue experiment showed the deliberation nudge recovers ~40% of
thin-capitulation cases. But a deployable "always deliberate" policy runs the
nudge on truthful-doc queries too (the overwhelming majority in real
corpora). This measures what that costs:

  HARM — does induced deliberation on TRUE documents hurt? (overthinking,
         hedging, talked-out-of-it errors). Control arm = the no-nudge
         correct-document control (99.8% correct).
  COST — actual API token usage of nudged calls, vs the control arm's
         estimated usage (control was cached before usage tracking; its
         output tokens are estimated from stored text at ~3.8 chars/token,
         flagged as estimate in the summary).

Population: every question in data/correct_control_results.json (reuses the
exact same correct doc per question, so arms differ only in the nudge).

Reads:
  data/correct_control_results.json

Produces:
  data/nudge_harm_results.json
  data/nudge_harm_progress.json

Args:
  --limit N    Only first N questions.
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONTROL_PATH = os.path.join(DATA_DIR, "correct_control_results.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "nudge_harm_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "nudge_harm_progress.json")
SAVE_EVERY = 50

NUDGE_SUFFIX = "\n\nThink through this thoroughly in your reasoning before answering."
CHARS_PER_TOKEN = 3.8  # for estimating control-arm output tokens from stored text

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


def process_one(row: dict) -> dict:
    question = row["question"]
    prompt = f"[DOCUMENT 1]\n{row['doc']}\n[/DOCUMENT 1]\n\n{question}{NUDGE_SUFFIX}"

    last_err = None
    for attempt in range(3):
        try:
            result = call_claude(
                messages=[{"role": "user", "content": prompt}],
                cache_key_parts=[question, "nudge_harm_c1", MODEL],
                max_tokens=16000,
                extended_thinking=True,
                budget_tokens=8000,
            )
            judge = call_correctness_judge(
                question=question,
                correct=row["correct_answer"],
                model_response=result["answer"],
            )
            return {
                "question": question,
                "correct_answer": row["correct_answer"],
                "domain": row["domain"],
                "answer": result["answer"],
                "thinking_trace": result["thinking"],
                "thinking_length": len(result["thinking"]),
                "judge_label": judge["label"],
                "judge_reason": judge["reason"],
                "usage": result.get("usage"),
                "control_thinking_length": row["thinking_length"],
                "control_judge_label": row["judge_label"],
                "control_est_output_tokens": (len(row["thinking_trace"]) + len(row["answer"])) / CHARS_PER_TOKEN,
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

    with open(CONTROL_PATH, encoding="utf-8") as f:
        control = json.load(f)
    if args.limit:
        control = control[:args.limit]

    results, tested = load_progress()
    print(f"Nudge harm/cost test: {len(control)} truthful-doc questions | workers: {args.workers}")
    print(f"Resumed: {len(results)} already done")

    tasks = [r for r in control if r["question"] not in tested]
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
                    out_tok = row["usage"]["output_tokens"] if row["usage"] else -1
                    line = (
                        f"  [{completed}/{len(tasks)}] {label:<3} | "
                        f"think={row['thinking_length']:>5}ch (ctrl {row['control_thinking_length']:>4}) | "
                        f"out_tok={out_tok:>5} | "
                        f"Q: {rec['question'][:42]}"
                    )
                    print(line.encode("ascii", errors="replace").decode(), flush=True)
                    if len(results) - last_save >= SAVE_EVERY:
                        save_progress(results, tested)
                        last_save = len(results)
                        print(f"  --- saved ({len(results)} rows, {rate:.2f}/sec) ---", flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {rec['question'][:45]} -> {e}", flush=True)
                with _lock:
                    save_progress(results, tested)

    save_progress(results, tested)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    import numpy as np
    print()
    print("=" * 70)
    print("NUDGE HARM/COST TEST COMPLETE")
    print(f"  Rows: {len(results)}  |  Errors: {errors}  |  {elapsed/60:.1f} min")
    if results:
        n = len(results)
        print()
        print("  HARM (accuracy on truthful docs):")
        for lbl in ("correct", "incorrect", "unclear"):
            k = sum(1 for r in results if r["judge_label"] == lbl)
            kc = sum(1 for r in results if r["control_judge_label"] == lbl)
            print(f"    {lbl:<10}: nudge={k:4d} ({100*k/n:.1f}%)   control={kc:4d} ({100*kc/n:.1f}%)")
        print()
        print("  COST:")
        think_n = [r["thinking_length"] for r in results]
        think_c = [r["control_thinking_length"] for r in results]
        print(f"    thinking chars: nudge mean={np.mean(think_n):.0f}  control mean={np.mean(think_c):.0f}")
        with_usage = [r for r in results if r["usage"]]
        if with_usage:
            out_n = [r["usage"]["output_tokens"] for r in with_usage]
            out_c = [r["control_est_output_tokens"] for r in with_usage]
            in_n = [r["usage"]["input_tokens"] for r in with_usage]
            print(f"    output tokens/query: nudge={np.mean(out_n):.0f} (actual)  "
                  f"control~{np.mean(out_c):.0f} (est. from chars)")
            print(f"    input tokens/query (nudge arm): {np.mean(in_n):.0f}")
            print(f"    marginal output cost of nudge: ~{np.mean(out_n)-np.mean(out_c):+.0f} tokens/query")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
