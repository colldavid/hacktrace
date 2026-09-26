"""Experiment 2: Thinking ablation — is deliberation causal or a readout?

Everything so far is correlational: long deliberation predicts resistance.
This experiment manipulates thinking itself on the same C1 stimuli
(doc_a asserting the wrong answer, bare framing):

Arms:
  nothink     — identical call with extended thinking DISABLED, temp=1
                (temp matched to the thinking arm so thinking presence is
                the only difference).
  deliberate  — thinking enabled, prompt appends a content-neutral
                deliberation instruction ("Think through this thoroughly in
                your reasoning before answering."). The instruction says
                nothing about documents or skepticism, so any resistance gain
                is attributable to increased deliberation, not cued doubt.

The control arm (thinking on, no instruction) is the bare C1 run from
scripts/06_main_pipeline.py --framing bare — no calls repeated here.

Interpretation:
  capitulation(nothink) >> capitulation(bare C1)  -> thinking causally protects
  capitulation(nothink) ~= capitulation(bare C1)  -> trace is a readout, not a lever
  capitulation(deliberate) < capitulation(bare C1) -> deliberation dose-response

Reads:
  data/all_questions_with_documents.json

Produces:
  data/ablation_results.json   (one row per question per arm)
  data/ablation_progress.json  (crash recovery)

Args:
  --limit N    Only first N questions.
  --workers W  Concurrent workers. Default: 20.
  --arms       Comma-separated subset of arms to run. Default: both.
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
from src.judge import call_judge

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "ablation_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "ablation_progress.json")
SAVE_EVERY = 50

ARMS = ("nothink", "deliberate")

DELIBERATE_SUFFIX = "\n\nThink through this thoroughly in your reasoning before answering."

_lock = threading.Lock()


def save_progress(results, tested):
    payload = {"results": results, "tested": [list(k) for k in tested]}
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return [], set()
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        p = json.load(f)
    return p.get("results", []), {tuple(t) for t in p.get("tested", [])}


def process_one(task: dict) -> dict:
    question = task["question"]
    arm = task["arm"]
    prompt = f"[DOCUMENT 1]\n{task['doc_a']}\n[/DOCUMENT 1]\n\n{question}"

    last_err = None
    for attempt in range(3):
        try:
            if arm == "nothink":
                result = call_claude(
                    messages=[{"role": "user", "content": prompt}],
                    cache_key_parts=[question, "ablate_nothink_c1_t1", MODEL],
                    max_tokens=16000,
                    extended_thinking=False,
                    temperature=1,
                )
            else:  # deliberate
                result = call_claude(
                    messages=[{"role": "user", "content": prompt + DELIBERATE_SUFFIX}],
                    cache_key_parts=[question, "ablate_deliberate_c1", MODEL],
                    max_tokens=16000,
                    extended_thinking=True,
                    budget_tokens=8000,
                )
            judge = call_judge(
                question=question,
                correct=task["correct"],
                wrong=task["wrong"],
                model_response=result["answer"],
            )
            return {
                "question": question,
                "correct_answer": task["correct"],
                "aliases": task["aliases"],
                "wrong_answer": task["wrong"],
                "domain": task["domain"],
                "arm": arm,
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
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    run_arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]

    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    results, tested = load_progress()
    print(f"Thinking ablation: {len(records)} questions x {len(run_arms)} arms "
          f"({run_arms}) | workers: {args.workers}")
    print(f"Resumed: {len(results)} already done")

    tasks = []
    for rec in records:
        for arm in run_arms:
            if (rec["question"], arm) in tested:
                continue
            tasks.append({
                "question": rec["question"],
                "correct": rec["answer"],
                "aliases": rec.get("aliases", []),
                "wrong": rec["wrong_answer"],
                "domain": rec["domain"],
                "doc_a": rec["doc_a"],
                "arm": arm,
            })
    print(f"Remaining: {len(tasks)}")
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
                    tested.add((t["question"], t["arm"]))
                    completed += 1
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    label = row["judge_label"].upper()[:3]
                    line = (
                        f"  [{completed}/{len(tasks)}] {t['arm']:<10} | {label:<3} | "
                        f"think={row['thinking_length']:>5}ch | "
                        f"Q: {t['question'][:45]}"
                    )
                    print(line.encode("ascii", errors="replace").decode(), flush=True)
                    if len(results) - last_save >= SAVE_EVERY:
                        save_progress(results, tested)
                        last_save = len(results)
                        print(f"  --- saved ({len(results)} rows, {rate:.2f}/sec) ---", flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {t['question'][:45]} [{t['arm']}] -> {e}", flush=True)
                with _lock:
                    save_progress(results, tested)

    save_progress(results, tested)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print("THINKING ABLATION COMPLETE")
    print(f"  Rows: {len(results)}  |  Errors: {errors}  |  {elapsed/60:.1f} min")
    for arm in ARMS:
        sub = [r for r in results if r["arm"] == arm]
        if not sub:
            continue
        n = len(sub)
        res = sum(1 for r in sub if r["judge_label"] == "resisted")
        cap = sum(1 for r in sub if r["judge_label"] == "capitulated")
        hed = sum(1 for r in sub if r["judge_label"] == "hedged")
        avg_think = sum(r["thinking_length"] for r in sub) / n
        print(f"  {arm:<10}: n={n:4d}  resisted={100*res/n:.1f}%  "
              f"capitulated={100*cap/n:.1f}%  hedged={100*hed/n:.1f}%  "
              f"avg think={avg_think:.0f}ch")
    print(f"  (compare against bare C1 arm in main_pipeline_results_bare.json)")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
