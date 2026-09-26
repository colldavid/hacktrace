"""Step 20: GPT capitulation pipeline — cross-model replication of the bare run.

Mirrors scripts/06_main_pipeline.py --framing bare, but with GPT as the
subject model. Restricted to questions GPT knows closed-book (from
scripts/19_gpt_gate.py) so capitulation can't be confused with ignorance —
the same role Claude's 3/3 correctness gate plays for the main dataset.

Design parity with the Claude bare run:
  - Same documents (doc_a / doc_b / doc_c from the shared corpus)
  - Same 3 conditions (1 / 2 / 3 documents)
  - Same bare framing: documents then the raw question, no instruction
  - Same Claude judge (label consistency with 15k+ existing labels)

Difference: OpenAI reasoning models expose reasoning *summaries*, not raw
traces. We record both the summary text and the reasoning-token count; the
token count is the GPT analog of Claude's thinking_length for the
thin/thick analysis.

Reads:
  data/all_questions_with_documents.json
  data/gpt_gate_results.json

Produces:
  data/gpt_pipeline_results.json
  data/gpt_pipeline_progress.json

Args:
  --limit N    Only first N gate-passing questions.
  --workers W  Concurrent workers. Default: 10.
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
from src.openai_client import call_openai, DEFAULT_REASONING_MODEL, DEFAULT_REASONING_EFFORT
from src.judge import call_judge

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
GATE_PATH = os.path.join(DATA_DIR, "gpt_gate_results.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "gpt_pipeline_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "gpt_pipeline_progress.json")
SAVE_EVERY = 50
CONDITIONS = (1, 2, 3)

_lock = threading.Lock()


def docs_for_condition(rec: dict, condition: int) -> list[str]:
    return [rec["doc_a"], rec["doc_b"], rec["doc_c"]][:condition]


def build_prompt(question: str, documents: list[str]) -> str:
    """Bare framing — identical structure to the Claude bare run."""
    blocks = ""
    for i, doc in enumerate(documents, 1):
        blocks += f"[DOCUMENT {i}]\n{doc}\n[/DOCUMENT {i}]\n\n"
    return blocks + question


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
    prompt = build_prompt(task["question"], task["documents"])
    last_err = None
    for attempt in range(3):
        try:
            result = call_openai(
                prompt,
                cache_key_parts=[task["question"], f"gpt_bare_c{task['condition']}",
                                 DEFAULT_REASONING_MODEL, DEFAULT_REASONING_EFFORT],
                reasoning=True,
                max_output_tokens=16000,
            )
            judge = call_judge(
                question=task["question"],
                correct=task["correct"],
                wrong=task["wrong"],
                model_response=result["answer"],
            )
            usage = result.get("usage") or {}
            return {
                "question": task["question"],
                "correct_answer": task["correct"],
                "aliases": task["aliases"],
                "wrong_answer": task["wrong"],
                "domain": task["domain"],
                "model": DEFAULT_REASONING_MODEL,
                "condition": task["condition"],
                "n_documents": len(task["documents"]),
                "answer": result["answer"],
                "reasoning_summary": result["reasoning_summary"],
                "reasoning_summary_length": len(result["reasoning_summary"]),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
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
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    init_tracing()

    with open(GATE_PATH, encoding="utf-8") as f:
        knows = {r["question"] for r in json.load(f) if r["knows"]}
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["question"] in knows]
    if args.limit:
        records = records[:args.limit]

    results, tested = load_progress()
    print(f"GPT pipeline: {len(records)} gate-passing questions x {len(CONDITIONS)} conditions "
          f"= {len(records)*len(CONDITIONS)} calls | model {DEFAULT_REASONING_MODEL}")
    print(f"Resumed: {len(results)} already done")

    tasks = []
    for rec in records:
        for cond in CONDITIONS:
            if (rec["question"], cond) in tested:
                continue
            tasks.append({
                "question": rec["question"],
                "correct": rec["answer"],
                "aliases": rec.get("aliases", []),
                "wrong": rec["wrong_answer"],
                "domain": rec["domain"],
                "condition": cond,
                "documents": docs_for_condition(rec, cond),
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
                    tested.add((t["question"], t["condition"]))
                    completed += 1
                    rate = completed / max(time.time() - start, 1e-9)
                    print(
                        f"  [{completed}/{len(tasks)}] C{t['condition']} | "
                        f"{row['judge_label'].upper()[:3]} | rt={row['reasoning_tokens']:>5} | "
                        f"correct={t['correct'][:16]:<16} wrong={t['wrong'][:16]:<16} | "
                        f"{t['question'][:34]}".encode("ascii", errors="replace").decode(),
                        flush=True,
                    )
                    if len(results) - last_save >= SAVE_EVERY:
                        save_progress(results, tested)
                        last_save = len(results)
                        print(f"  --- saved ({len(results)} rows, {rate:.2f}/sec) ---", flush=True)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {t['question'][:45]} [C{t['condition']}] -> {e}", flush=True)
                with _lock:
                    save_progress(results, tested)

    save_progress(results, tested)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    import numpy as np
    elapsed = time.time() - start
    print()
    print("=" * 70)
    print("GPT PIPELINE COMPLETE")
    print(f"  Rows: {len(results)}  |  Errors: {errors}  |  {elapsed/60:.1f} min")
    for cond in CONDITIONS:
        sub = [r for r in results if r["condition"] == cond]
        if not sub:
            continue
        n = len(sub)
        res = sum(1 for r in sub if r["judge_label"] == "resisted")
        cap = sum(1 for r in sub if r["judge_label"] == "capitulated")
        hed = sum(1 for r in sub if r["judge_label"] == "hedged")
        rt = np.mean([r["reasoning_tokens"] for r in sub])
        print(f"    C{cond}: n={n:4d}  resisted={100*res/n:.1f}%  capitulated={100*cap/n:.1f}%  "
              f"hedged={100*hed/n:.1f}%  avg reasoning tokens={rt:.0f}")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
