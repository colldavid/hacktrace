"""Experiment 5: Always-nudge on false docs + truncated detection on nudged traces.

Runs the deliberation nudge on the FULL false-doc population (thin and thick
alike — not just the thin-capitulation rescue subset), on the same first-200
questions as the baseline run. Two questions:

A. What is the overall capitulation rate under an "always deliberate" policy
   on false docs? (The rescue only measured the thin-cap subset.)
   Compared against the same questions' bare C1 results, paired.

B. Does truncated-trace detection survive the nudge? Nudged traces are longer,
   so at small prefixes the "trace already ended" (thinness) signal vanishes —
   content features must carry detection. Deployment-realistic transfer test:
   detector trained on ALL bare (unnudged) C1..C3 rows, evaluated on the 200
   nudged traces, at each prefix length.

Reads:
  data/all_questions_with_documents.json
  data/main_pipeline_results_bare.json

Produces:
  data/nudge_full_results.json
  data/nudge_full_progress.json

Args:
  --limit N    Questions to run (default 200, matching the baseline subset).
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
INPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
BARE_PATH = os.path.join(DATA_DIR, "main_pipeline_results_bare.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "nudge_full_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "nudge_full_progress.json")
SAVE_EVERY = 50

NUDGE_SUFFIX = "\n\nThink through this thoroughly in your reasoning before answering."
PREFIX_LENGTHS = (50, 100, 150, 200, 300, 500, None)

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


def process_one(task: dict) -> dict:
    question = task["question"]
    prompt = f"[DOCUMENT 1]\n{task['doc_a']}\n[/DOCUMENT 1]\n\n{question}{NUDGE_SUFFIX}"
    last_err = None
    for attempt in range(3):
        try:
            result = call_claude(
                messages=[{"role": "user", "content": prompt}],
                cache_key_parts=[question, "nudge_full_c1", MODEL],
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
                "condition": 1,
                "n_documents": 1,
                "answer": result["answer"],
                "thinking_trace": result["thinking"],
                "thinking_length": len(result["thinking"]),
                "judge_label": judge["label"],
                "judge_reason": judge["reason"],
                "bare_judge_label": task["bare_label"],
                "bare_thinking_length": task["bare_think"],
                "usage": result.get("usage"),
                "cached": result["cached"],
            }
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def run_calls(args):
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)[: args.limit]
    with open(BARE_PATH, encoding="utf-8") as f:
        bare_c1 = {r["question"]: r for r in json.load(f)
                   if r["condition"] == 1 and r.get("judge_label")}

    results, tested = load_progress()
    print(f"Always-nudge on false docs: {len(records)} questions | workers: {args.workers}")
    print(f"Resumed: {len(results)} already done")

    tasks = []
    for rec in records:
        q = rec["question"]
        if q in tested or q not in bare_c1:
            continue
        tasks.append({
            "question": q,
            "correct": rec["answer"],
            "aliases": rec.get("aliases", []),
            "wrong": rec["wrong_answer"],
            "domain": rec["domain"],
            "doc_a": rec["doc_a"],
            "bare_label": bare_c1[q]["judge_label"],
            "bare_think": bare_c1[q]["thinking_length"],
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
                    tested.add(t["question"])
                    completed += 1
                    label = row["judge_label"].upper()[:3]
                    bare_label = row["bare_judge_label"].upper()[:3]
                    print(
                        f"  [{completed}/{len(tasks)}] {bare_label}->{label} | "
                        f"think={row['thinking_length']:>5}ch (bare {row['bare_thinking_length']:>4}) | "
                        f"Q: {t['question'][:45]}".encode("ascii", errors="replace").decode(),
                        flush=True,
                    )
                    if len(results) - last_save >= SAVE_EVERY:
                        save_progress(results, tested)
                        last_save = len(results)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {t['question'][:45]} -> {e}", flush=True)
                with _lock:
                    save_progress(results, tested)

    save_progress(results, tested)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nCalls done: {len(results)} rows, {errors} errors, {(time.time()-start)/60:.1f} min")
    return results


def analyze(results):
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score
    from src.features import extract_with_meta, FEATURE_NAMES

    DEPLOY = FEATURE_NAMES + ["condition", "n_documents"]

    n = len(results)
    print()
    print("=" * 70)
    print("A. ALWAYS-NUDGE ON FALSE DOCS (paired vs bare C1, same questions)")
    for src, key in (("bare C1", "bare_judge_label"), ("nudged", "judge_label")):
        res = sum(1 for r in results if r[key] == "resisted")
        cap = sum(1 for r in results if r[key] == "capitulated")
        hed = sum(1 for r in results if r[key] == "hedged")
        print(f"  {src:<8}: resisted={100*res/n:.1f}%  capitulated={100*cap/n:.1f}%  hedged={100*hed/n:.1f}%")
    think_nudge = np.mean([r["thinking_length"] for r in results])
    think_bare = np.mean([r["bare_thinking_length"] for r in results])
    print(f"  thinking: nudged mean={think_nudge:.0f}ch  bare mean={think_bare:.0f}ch")

    print()
    print("B. TRUNCATED DETECTION ON NUDGED TRACES")
    print("   (detector trained on ALL bare rows, deployable features, tested on nudged)")
    with open(BARE_PATH, encoding="utf-8") as f:
        bare_all = [r for r in json.load(f) if r.get("judge_label")]

    def feats(rows, prefix):
        out = []
        for r in rows:
            rr = dict(r)
            if prefix is not None:
                rr["thinking_trace"] = r["thinking_trace"][:prefix]
            d = extract_with_meta(rr)
            out.append([d[k] for k in DEPLOY])
        return np.array(out)

    y_train = np.array([1 if r["judge_label"] == "capitulated" else 0 for r in bare_all])
    y_test = np.array([1 if r["judge_label"] == "capitulated" else 0 for r in results])
    print(f"   train n={len(bare_all)} (bare)  test n={len(results)} (nudged, cap rate {100*y_test.mean():.1f}%)")
    print(f"   {'prefix':>7} | {'acc':>6} {'AUC':>6}")
    for prefix in PREFIX_LENGTHS:
        Xtr = feats(bare_all, prefix)
        Xte = feats(results, prefix)
        m = Pipeline([("c", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42))])
        m.fit(Xtr, y_train)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        label = "full" if prefix is None else str(prefix)
        print(f"   {label:>7} | {accuracy_score(y_test, pred):>6.3f} {roc_auc_score(y_test, proba):>6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            results = json.load(f)
    else:
        init_tracing()
        results = run_calls(args)
    analyze(results)


if __name__ == "__main__":
    main()
