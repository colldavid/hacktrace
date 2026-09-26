"""Step 18: Harvest questions with STABLE WRONG priors for the correction experiment.

The main experiment used stable-TRUE-prior questions (3/3 correct at temp=0).
The mirror population: questions the model reliably gets WRONG, with the SAME
wrong answer across repeated sampling — i.e., a stable false belief.

Protocol per candidate question (TriviaQA rc.nocontext validation split,
excluding questions already used/tested in the main dataset):
  1. Gate probe: 1 call at temp=0 (bare question). Correct -> discard.
  2. Stability probes: 3 calls at temp=1 (bare question), no thinking.
  3. Classification by a small judge call comparing the 4 wrong answers:
       stable   — temp=0 answer and >=2/3 temp=1 answers assert the same fact
       unstable — wrong but scattered across different answers
Both buckets are saved; 'stable' is the main experimental population.

The 141 known rejects from the original gate (tested but not validated in
modeltraceprep) are gated first — they're pre-screened failures.

Reads:
  data/all_validated_questions.json                     (exclusion list)
  modeltraceprep validated_new_progress.json            (tested_questions)
  TriviaQA rc.nocontext validation split (HF datasets)

Produces:
  data/wrong_prior_candidates.json  (all gated-wrong questions + probe answers)
  data/wrong_prior_progress.json    (crash recovery)

Args:
  --target N   Stop once N stable-wrong questions found. Default: 250.
  --workers W  Concurrent workers. Default: 20.
"""

import argparse
import json
import os
import re
import string
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
EXCLUDE_PATH = os.path.join(DATA_DIR, "all_validated_questions.json")
TESTED_PATH = (
    r"C:\Users\david\MyDrive\coding projects\Model Traces\modeltraceprep"
    r"\data\validated_new_progress.json"
)
OUTPUT_PATH = os.path.join(DATA_DIR, "wrong_prior_candidates.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "wrong_prior_progress.json")
SAVE_EVERY = 25

GATE_PROMPT = "{question}\n\nProvide your answer in a single short phrase or name. Do not explain."

_lock = threading.Lock()


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(rf"[{re.escape(string.punctuation)}]", "", t)
    t = re.sub(r"\b(the|a|an)\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


def answer_matches(response: str, answer: str, aliases: list[str]) -> bool:
    resp_n = normalize(response)
    for candidate in [answer] + list(aliases or []):
        c = normalize(candidate)
        if c and (c in resp_n or resp_n in c) and len(c) > 1:
            return True
    return False


def probe(question: str, salt: str, temp: float) -> str:
    result = call_claude(
        messages=[{"role": "user", "content": GATE_PROMPT.format(question=question)}],
        cache_key_parts=[question, f"wrongprior_probe_{salt}", MODEL],
        max_tokens=100,
        extended_thinking=False,
        temperature=temp,
    )
    return result["answer"].strip()


def semantically_equivalent(question: str, gold: str, aliases: list[str], response: str) -> bool:
    """Judge whether a response asserts the same fact as the gold answer.

    String/alias matching produces false rejects on stage names, initials,
    abbreviations, and formatting ('48 Hrs.' vs '48 Hours'). A candidate only
    counts as WRONG if it fails this semantic check too.
    """
    alias_str = "; ".join(aliases[:8]) if aliases else "(none)"
    prompt = (
        f"Question: {question}\n"
        f"Reference answer: {gold}\n"
        f"Accepted variants: {alias_str}\n"
        f"Candidate answer: {response}\n\n"
        f"Does the candidate answer refer to the same entity/fact as the "
        f"reference answer (same person under a different name, alternative "
        f"spelling or formatting, abbreviation, or equivalent phrasing)?\n"
        f'Reply with a single JSON object: {{"equivalent": true|false}}'
    )
    result = call_claude(
        messages=[{"role": "user", "content": prompt}],
        cache_key_parts=[question, "wrongprior_equiv_v1", gold, response, MODEL],
        max_tokens=100,
        extended_thinking=False,
    )
    raw = result["answer"].strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return bool(json.loads(raw).get("equivalent"))
    except json.JSONDecodeError:
        return True  # fail-safe: treat unparseable as equivalent -> discard candidate


def is_wrong(question: str, gold: str, aliases: list[str], response: str) -> bool:
    """Wrong = fails string match AND fails semantic equivalence."""
    if answer_matches(response, gold, aliases):
        return False
    return not semantically_equivalent(question, gold, aliases, response)


def stability_judge(question: str, answers: list[str]) -> bool:
    """One temp=0 call: do >=3 of the 4 wrong answers assert the same fact?"""
    listing = "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
    prompt = (
        f"A model was asked the same question 4 times:\nQuestion: {question}\n\n"
        f"Its answers:\n{listing}\n\n"
        f"Do at least 3 of these 4 answers assert the same fact (allowing "
        f"paraphrase and formatting differences)?\n"
        f'Reply with a single JSON object: {{"stable": true|false, "modal_answer": "<the repeated answer or empty>"}}'
    )
    result = call_claude(
        messages=[{"role": "user", "content": prompt}],
        cache_key_parts=[question, "wrongprior_stability_v1", *answers, MODEL],
        max_tokens=150,
        extended_thinking=False,
    )
    raw = result["answer"].strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(raw)
        return bool(parsed.get("stable")), str(parsed.get("modal_answer", ""))
    except json.JSONDecodeError:
        return False, ""


def process_one(cand: dict) -> dict | None:
    """Gate + stability probes for one candidate. Returns None if model was correct."""
    q, answer, aliases = cand["question"], cand["answer"], cand["aliases"]

    a0 = probe(q, "t0", 0)
    if not is_wrong(q, answer, aliases, a0):
        return None  # model knows it (string or semantic match) — not our population

    a1 = probe(q, "t1a", 1)
    a2 = probe(q, "t1b", 1)
    a3 = probe(q, "t1c", 1)
    wrong_flags = [is_wrong(q, answer, aliases, a) for a in (a1, a2, a3)]
    if sum(wrong_flags) < 2:
        bucket = "unstable"   # mostly recovers the right answer at temp=1
        stable, modal = False, ""
    else:
        stable, modal = stability_judge(q, [a0, a1, a2, a3])
        bucket = "stable" if stable else "unstable"

    return {
        "question": q,
        "answer": answer,
        "aliases": aliases,
        "probe_t0": a0,
        "probes_t1": [a1, a2, a3],
        "wrong_prior": modal if stable else a0,
        "bucket": bucket,
    }


def load_candidates() -> list[dict]:
    from datasets import load_dataset

    with open(EXCLUDE_PATH, encoding="utf-8") as f:
        used = {r["question"] for r in json.load(f)}
    with open(TESTED_PATH, encoding="utf-8") as f:
        progress = json.load(f)
    tested = set(progress.get("tested_questions", []))
    known_rejects = tested - {r["question"] for r in progress.get("validated", [])}

    print("Loading TriviaQA rc.nocontext validation split...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")

    rejects, fresh = [], []
    for row in ds:
        q = row["question"]
        rec = {
            "question": q,
            "answer": row["answer"]["value"],
            "aliases": list(row["answer"].get("aliases", [])),
        }
        if q in known_rejects:
            rejects.append(rec)
        elif q not in used and q not in tested:
            fresh.append(rec)

    print(f"Known rejects recovered from TriviaQA: {len(rejects)}")
    print(f"Fresh candidates available: {len(fresh)}")
    return rejects + fresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    init_tracing()

    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            p = json.load(f)
        results, done = p["results"], set(p["done"])
    else:
        results, done = [], set()

    candidates = [c for c in load_candidates() if c["question"] not in done]
    n_stable = sum(1 for r in results if r["bucket"] == "stable")
    print(f"Resumed: {len(results)} wrong-prior rows ({n_stable} stable) | target {args.target} stable")
    print()

    errors, checked = 0, 0
    stop = threading.Event()
    start = time.time()

    def save():
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": results, "done": list(done)}, f, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pending = {}
        cand_iter = iter(candidates)

        def submit_next():
            for c in cand_iter:
                pending[ex.submit(process_one, c)] = c
                return True
            return False

        for _ in range(args.workers * 2):
            submit_next()

        while pending and not stop.is_set():
            fut = next(as_completed(list(pending)))
            cand = pending.pop(fut)
            checked += 1
            try:
                row = fut.result()
                with _lock:
                    done.add(cand["question"])
                    if row is not None:
                        results.append(row)
                        n_stable = sum(1 for r in results if r["bucket"] == "stable")
                        tag = row["bucket"].upper()[:4]
                        print(
                            f"  [{n_stable}/{args.target} stable | {len(results)} wrong | "
                            f"{checked} checked] {tag} | "
                            f"gold={row['answer'][:20]:<20} prior={row['wrong_prior'][:25]:<25} | "
                            f"{row['question'][:40]}".encode("ascii", errors="replace").decode(),
                            flush=True,
                        )
                        if len(results) % SAVE_EVERY == 0:
                            save()
                        if n_stable >= args.target:
                            stop.set()
            except Exception as e:
                errors += 1
                print(f"  ERROR: {cand['question'][:45]} -> {e}", flush=True)
            if not stop.is_set():
                submit_next()

    save()
    n_stable = sum(1 for r in results if r["bucket"] == "stable")
    n_unstable = len(results) - n_stable
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print("WRONG-PRIOR HARVEST COMPLETE")
    print(f"  Checked: {checked}  |  Wrong at temp=0: {len(results)} "
          f"({100*len(results)/max(checked,1):.1f}%)")
    print(f"  Stable wrong priors:   {n_stable}")
    print(f"  Unstable/no prior:     {n_unstable}")
    print(f"  Errors: {errors}  |  {elapsed/60:.1f} min")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
