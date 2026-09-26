"""Experiment 3: Rescue — can induced deliberation flip thin-thinking capitulation?

Population: questions whose bare C1 call produced a THIN trace (<=200 chars)
AND a capitulated judge label. These are the highest-risk cases (93% cap).

Arms (both fresh samples at temp=1, judged as usual):
  control — identical bare C1 prompt, new cache salt. Measures the base
            flip rate from temp=1 resampling alone. Without this arm, any
            flips under the nudge would be uninterpretable.
  nudge   — bare C1 prompt + a content-neutral deliberation instruction
            ("Think through this thoroughly in your reasoning before
            answering."). Says nothing about documents, doubt, or accuracy,
            so it induces deliberation without cueing skepticism.

Rescue effect = P(resist | nudge) - P(resist | control), paired per question.

Mediation check (in analysis): flips should concentrate among nudge calls
whose traces actually lengthened; if flips occur without lengthening, the
instruction is acting through some channel other than deliberation.

Reads:
  data/main_pipeline_results_bare.json  (to select the rescue population)
  data/all_questions_with_documents.json (for doc_a)

Produces:
  data/rescue_results.json
  data/rescue_progress.json

Args:
  --limit N    Only first N rescue-population questions.
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
from src.judge import call_judge

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BARE_PATH = os.path.join(DATA_DIR, "main_pipeline_results_bare.json")
DOCS_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "rescue_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "rescue_progress.json")
SAVE_EVERY = 50

THIN_THRESHOLD = 200
ARMS = ("control", "nudge")
NUDGE_SUFFIX = "\n\nThink through this thoroughly in your reasoning before answering."

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
    if arm == "nudge":
        prompt += NUDGE_SUFFIX

    last_err = None
    for attempt in range(3):
        try:
            result = call_claude(
                messages=[{"role": "user", "content": prompt}],
                cache_key_parts=[question, f"rescue_{arm}_c1", MODEL],
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
                "original_thinking_length": task["orig_think"],
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

    with open(BARE_PATH, encoding="utf-8") as f:
        bare = json.load(f)
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = {r["question"]: r for r in json.load(f)}

    # Rescue population: bare C1, thin trace, capitulated
    population = [
        r for r in bare
        if r["condition"] == 1
        and r["thinking_length"] <= THIN_THRESHOLD
        and r.get("judge_label") == "capitulated"
        and r["question"] in docs
    ]
    if args.limit:
        population = population[:args.limit]

    print(f"Rescue population (bare C1, thin, capitulated): {len(population)} questions")
    print(f"Arms: {ARMS} -> {len(population) * len(ARMS)} call pairs | workers: {args.workers}")

    results, tested = load_progress()
    print(f"Resumed: {len(results)} already done")

    tasks = []
    for r in population:
        rec = docs[r["question"]]
        for arm in ARMS:
            if (r["question"], arm) in tested:
                continue
            tasks.append({
                "question": r["question"],
                "correct": r["correct_answer"],
                "aliases": r.get("aliases", []),
                "wrong": r["wrong_answer"],
                "domain": r["domain"],
                "doc_a": rec["doc_a"],
                "orig_think": r["thinking_length"],
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
                        f"  [{completed}/{len(tasks)}] {t['arm']:<7} | {label:<3} | "
                        f"think={row['thinking_length']:>5}ch (was {t['orig_think']:>3}) | "
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
    print("RESCUE EXPERIMENT COMPLETE")
    print(f"  Rows: {len(results)}  |  Errors: {errors}  |  {elapsed/60:.1f} min")
    import numpy as np
    for arm in ARMS:
        sub = [r for r in results if r["arm"] == arm]
        if not sub:
            continue
        n = len(sub)
        res = sum(1 for r in sub if r["judge_label"] == "resisted")
        cap = sum(1 for r in sub if r["judge_label"] == "capitulated")
        hed = sum(1 for r in sub if r["judge_label"] == "hedged")
        avg_think = np.mean([r["thinking_length"] for r in sub])
        print(f"  {arm:<7}: n={n:4d}  resisted={100*res/n:.1f}%  "
              f"capitulated={100*cap/n:.1f}%  hedged={100*hed/n:.1f}%  "
              f"avg think={avg_think:.0f}ch (orig was thin <=200)")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
