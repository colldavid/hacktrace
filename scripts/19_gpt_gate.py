"""Step 19: GPT closed-book knowledge gate on the anchor question subset.

The Claude experiments used questions gated on CLAUDE's knowledge (3/3 at
temp=0). Before any cross-model capitulation comparison, GPT gets the same
treatment: closed-book probe per question; comparisons later restrict to the
intersection both models know.

Wrongness check mirrors the harvester: string/alias match first; on miss, the
Claude semantic-equivalence judge adjudicates (stage names, abbreviations,
formatting). Judge calls are tiny and tolerate running alongside other work.

Reads:
  data/all_questions_with_documents.json

Produces:
  data/gpt_gate_results.json   (question, gpt_answer, knows: bool)
  data/gpt_gate_progress.json

Args:
  --limit N    Questions to gate. Default: 200 (the anchor subset).
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

from src.openai_client import call_openai, DEFAULT_REASONING_MODEL, DEFAULT_REASONING_EFFORT

# Reuse the harvester's matching + equivalence logic
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
_harvest = import_module("18_harvest_wrong_priors")
answer_matches = _harvest.answer_matches
semantically_equivalent = _harvest.semantically_equivalent

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "gpt_gate_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "gpt_gate_progress.json")

GATE_PROMPT = "{question}\n\nProvide your answer in a single short phrase or name. Do not explain."

_lock = threading.Lock()


def process_one(rec: dict) -> dict:
    q = rec["question"]
    last_err = None
    for attempt in range(3):
        try:
            r = call_openai(
                GATE_PROMPT.format(question=q),
                cache_key_parts=[q, "gpt_gate_v1", DEFAULT_REASONING_MODEL, DEFAULT_REASONING_EFFORT],
                reasoning=True,
                max_output_tokens=2000,
            )
            ans = r["answer"].strip()
            knows = answer_matches(ans, rec["answer"], rec.get("aliases", []))
            if not knows:
                knows = semantically_equivalent(q, rec["answer"], rec.get("aliases", []), ans)
            return {
                "question": q,
                "gold": rec["answer"],
                "gpt_answer": ans,
                "knows": knows,
                "reasoning_tokens": (r["usage"] or {}).get("reasoning_tokens", 0),
            }
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)[: args.limit]

    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            p = json.load(f)
        results, done = p["results"], set(p["done"])
    else:
        results, done = [], set()

    tasks = [r for r in records if r["question"] not in done]
    print(f"GPT gate: {len(records)} questions | remaining {len(tasks)} | model {DEFAULT_REASONING_MODEL}")

    errors = 0
    start = time.time()

    def save():
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": results, "done": list(done)}, f, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, r): r for r in tasks}
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                row = fut.result()
                with _lock:
                    results.append(row)
                    done.add(rec["question"])
                    flag = "KNOWS" if row["knows"] else "MISS "
                    print(f"  [{len(results)}/{len(records)}] {flag} | rt={row['reasoning_tokens']:>4} | "
                          f"gold={row['gold'][:22]:<22} gpt={row['gpt_answer'][:28]:<28} | "
                          f"{row['question'][:38]}".encode("ascii", errors="replace").decode(),
                          flush=True)
                    if len(results) % 25 == 0:
                        save()
            except Exception as e:
                errors += 1
                print(f"  ERROR: {rec['question'][:45]} -> {e}", flush=True)

    save()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_knows = sum(1 for r in results if r["knows"])
    print()
    print(f"GPT GATE COMPLETE — {len(results)} gated, {errors} errors, {(time.time()-start)/60:.1f} min")
    print(f"  knows: {n_knows} ({100*n_knows/len(results):.1f}%)")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
