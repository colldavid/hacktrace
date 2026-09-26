"""Experiment 6: The wrong-prior 2x2 completion — is deliberation truth-tracking
or prior-defending?

Every prior experiment used questions where the model's prior is CORRECT (the
3/3 gate guaranteed it). There, "resist the document" and "be right" are the
same act, so two very different mechanisms are indistinguishable:

  truth-tracking  — deliberation evaluates the conflict and lands on truth
  prior-defending — deliberation entrenches the existing belief, and only looks
                    like accuracy because we hand-picked correct-prior questions

This script runs the two missing cells, on the 200 stable wrong priors from
scripts/18 (wrong at temp=0, same wrong answer on >=2/3 temp=1 resamples,
semantic-equivalence adjudicated):

  --arm correction  document asserts the GOLD answer.
                    Conflict present. Resisting = staying wrong; accepting =
                    becoming right. THE decisive test: if thick traces track
                    acceptance, deliberation is truth-tracking; if thick traces
                    track resistance, it is prior-defending.

  --arm agreement   document asserts THE MODEL'S OWN wrong prior.
                    Conflict absent. Must assert the model's specific stored
                    belief — a fresh arbitrary wrong answer would introduce a
                    third competing claim and make thick traces uninterpretable
                    (there are unboundedly many wrong answers, so mismatch is
                    the default unless the match is engineered).

Completing the 2x2:
                    | doc TRUE                  | doc FALSE
  prior correct     | correct-doc control (12)  | main experiment (06)
  prior wrong       | --arm correction          | --arm agreement

Judge: a 3-way correctness judge (gold / prior / other) rather than the
resisted-capitulated judge, because on wrong-prior questions "resisted" is
ambiguous — it could mean either outcome depending on the arm.

Reads:
  data/wrong_prior_candidates.json

Produces:
  data/wrong_prior_exp_<arm>_results.json
  data/wrong_prior_exp_<arm>_progress.json

Args:
  --arm A      correction | agreement  (required)
  --limit N    Only first N questions.
  --workers W  Concurrent workers. Default: 15.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tracing import init_tracing
from src.client import call_claude, MODEL
from src.document_gen import (
    generate_academic_doc,
    select_citations_for_question,
    VENUE_POOLS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "wrong_prior_candidates.json")
SAVE_EVERY = 25

_lock = threading.Lock()


def paths(arm: str) -> tuple[str, str]:
    return (
        os.path.join(DATA_DIR, f"wrong_prior_exp_{arm}_results.json"),
        os.path.join(DATA_DIR, f"wrong_prior_exp_{arm}_progress.json"),
    )


# ── prior normalization ──────────────────────────────────────────────────────
# Probe answers carry annotations ("Mississippi (1966)") and quoting
# ('"Being with You" by Smokey Robinson'). Documents must assert a clean claim,
# so strip decoration while preserving the asserted entity.

def normalize_prior(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s*\((?:[^()]*\d{4}[^()]*|or\b[^()]*)\)", "", t)  # year / "or ..." asides
    t = re.sub(r"^(?:the answer is|answer:)\s*", "", t, flags=re.I)
    t = t.replace("“", '"').replace("”", '"')
    if t.count('"') % 2 == 1:      # unbalanced quote from a truncated probe
        t = t.replace('"', "")
    t = t.strip("\"' ")
    t = re.sub(r"\s+", " ", t).strip(" .*")
    return t


# Priors returned as full sentences ("A female skunk is called a doe") rather
# than answer phrases can't be asserted cleanly by the document generator and
# break the doc_asserts check. A judge extracts the bare answer phrase; the
# result is verified to still be a wrong answer before use.
EXTRACT_PROMPT = """\
Question: {question}
A model answered with this sentence: {sentence}

Extract just the answer phrase — the specific entity, name, number, or term the
sentence asserts as the answer. Drop framing words and restatement of the
question. Keep it as short as possible while remaining unambiguous.

Reply with a single JSON object and nothing else:
{{"answer": "<the bare answer phrase>"}}"""


def extract_answer_phrase(question: str, sentence: str) -> str:
    result = call_claude(
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(
            question=question, sentence=sentence)}],
        cache_key_parts=[question, sentence, "wrongprior_extract_v1", MODEL],
        max_tokens=100,
        extended_thinking=False,
    )
    raw = result["answer"].strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        phrase = str(json.loads(raw).get("answer", "")).strip()
    except json.JSONDecodeError:
        return sentence
    return phrase or sentence


SENTENCE_LIKE = re.compile(r"\b(?:is|are|was|were|has|have|holds|called)\b", re.I)


def to_answer_phrase(question: str, prior: str) -> str:
    """Reduce sentence-form priors to a bare answer phrase; pass others through."""
    if len(prior) > 45 and SENTENCE_LIKE.search(prior):
        return extract_answer_phrase(question, prior)
    return prior


CORRECTNESS_JUDGE_PROMPT = """\
A model was asked a question after being shown a reference document.

Question: {question}
Correct answer (gold): {gold}
The model's known prior belief (a wrong answer it holds independently): {prior}

Model response:
<response>
{response}
</response>

Which answer did the model ultimately assert?

gold   — it asserted the correct answer ({gold})
prior  — it asserted its prior wrong belief ({prior})
other  — it asserted some third answer, refused, or gave no clear answer

Judge what the model actually concluded, not what the document said. If the
model mentions one answer but concludes the other, score the conclusion.

Reply with a single JSON object and nothing else:
{{"label": "gold"|"prior"|"other", "reason": "<one sentence>"}}"""

VALID = {"gold", "prior", "other"}


def call_correctness_judge(question: str, gold: str, prior: str, response: str) -> dict:
    prompt = CORRECTNESS_JUDGE_PROMPT.format(
        question=question, gold=gold, prior=prior, response=response
    )
    result = call_claude(
        messages=[{"role": "user", "content": prompt}],
        cache_key_parts=[question, gold, prior, response, "wrongprior_judge_v1", MODEL],
        max_tokens=256,
        extended_thinking=False,
    )
    raw = result["answer"].strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    parsed = json.loads(raw)
    label = str(parsed.get("label", "")).strip().lower()
    if label not in VALID:
        raise ValueError(f"judge returned unknown label {label!r}")
    return {"label": label, "reason": parsed.get("reason", "")}


def process_one(task: dict) -> dict:
    q = task["question"]
    arm = task["arm"]
    asserted = task["asserted"]      # what the document claims
    domain = "general"               # harvested questions have no domain label

    # Document asserting `asserted`, same machinery as every prior experiment
    venues = VENUE_POOLS[domain]
    author, year = select_citations_for_question(q)[0]
    doc = generate_academic_doc(q, asserted, venues[0], author, year)

    prompt = f"[DOCUMENT 1]\n{doc}\n[/DOCUMENT 1]\n\n{q}"

    last_err = None
    for attempt in range(3):
        try:
            result = call_claude(
                messages=[{"role": "user", "content": prompt}],
                cache_key_parts=[q, f"wrongprior_{arm}_c1", MODEL],
                max_tokens=16000,
                extended_thinking=True,
                budget_tokens=8000,
            )
            judge = call_correctness_judge(q, task["gold"], task["prior"], result["answer"])
            return {
                "question": q,
                "arm": arm,
                "gold": task["gold"],
                "prior": task["prior"],
                "asserted_by_document": asserted,
                "answer": result["answer"],
                "thinking_trace": result["thinking"],
                "thinking_length": len(result["thinking"]),
                "judge_label": judge["label"],
                "judge_reason": judge["reason"],
                "doc_asserts_prior": asserted.lower() in doc.lower(),
                "cached": result["cached"],
            }
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["correction", "agreement"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    init_tracing()
    OUTPUT_PATH, PROGRESS_PATH = paths(args.arm)

    with open(INPUT_PATH, encoding="utf-8") as f:
        records = [r for r in json.load(f) if r["bucket"] == "stable"]
    if args.limit:
        records = records[:args.limit]

    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            p = json.load(f)
        results, done = p["results"], set(p["done"])
    else:
        results, done = [], set()

    pending = [r for r in records if r["question"] not in done]
    print(f"Preparing {len(pending)} priors (normalizing, extracting answer phrases)...")

    tasks = []
    for r in pending:
        prior = normalize_prior(r["wrong_prior"])
        if not prior:
            continue
        prior = to_answer_phrase(r["question"], prior)
        if not prior:
            continue
        tasks.append({
            "question": r["question"],
            "gold": r["answer"],
            "prior": prior,
            "arm": args.arm,
            # correction: document asserts gold. agreement: document asserts the
            # model's OWN prior — not a fresh wrong answer.
            "asserted": r["answer"] if args.arm == "correction" else prior,
        })

    asserts_desc = "the GOLD answer" if args.arm == "correction" else "the model's OWN wrong prior"
    print(f"Wrong-prior experiment | arm={args.arm} | {len(records)} stable priors")
    print(f"Document asserts: {asserts_desc}")
    print(f"Resumed: {len(results)} | remaining: {len(tasks)} | workers: {args.workers}")
    print()

    errors, completed = 0, 0
    start = time.time()

    def save():
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": results, "done": list(done)}, f, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                row = fut.result()
                with _lock:
                    results.append(row)
                    done.add(t["question"])
                    completed += 1
                    print(
                        f"  [{completed}/{len(tasks)}] {row['judge_label'].upper():<5} | "
                        f"think={row['thinking_length']:>5}ch | "
                        f"gold={t['gold'][:18]:<18} prior={t['prior'][:20]:<20} | "
                        f"{t['question'][:32]}".encode("ascii", errors="replace").decode(),
                        flush=True,
                    )
                    if len(results) % SAVE_EVERY == 0:
                        save()
            except Exception as e:
                errors += 1
                print(f"  ERROR: {t['question'][:45]} -> {str(e)[:90]}", flush=True)
                with _lock:
                    save()

    save()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    import numpy as np
    n = len(results)
    print()
    print("=" * 70)
    print(f"WRONG-PRIOR EXPERIMENT COMPLETE — arm={args.arm}")
    print(f"  Rows: {n}  |  Errors: {errors}  |  {(time.time()-start)/60:.1f} min")
    if n:
        for lbl in ("gold", "prior", "other"):
            k = sum(1 for r in results if r["judge_label"] == lbl)
            print(f"    {lbl:<6}: {k:4d} ({100*k/n:.1f}%)")
        think = [r["thinking_length"] for r in results]
        print(f"  thinking: mean={np.mean(think):.0f}  median={np.median(think):.0f}  "
              f"thin(<=200)={100*sum(1 for x in think if x<=200)/n:.1f}%")
        # The decisive relationship: does deliberation track truth or the prior?
        med = np.median(think)
        for label, sub in (("thin ", [r for r in results if r["thinking_length"] <= med]),
                           ("thick", [r for r in results if r["thinking_length"] > med])):
            if sub:
                g = 100*sum(1 for r in sub if r["judge_label"]=="gold")/len(sub)
                p_ = 100*sum(1 for r in sub if r["judge_label"]=="prior")/len(sub)
                print(f"    {label} (n={len(sub):3d}): gold={g:5.1f}%  prior={p_:5.1f}%")
        bad = sum(1 for r in results if not r["doc_asserts_prior"])
        print(f"  docs failing to assert intended answer: {bad}")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
