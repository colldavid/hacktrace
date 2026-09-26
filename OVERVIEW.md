# HackTrace — Project Overview

**Question:** When a language model is shown documents that contradict what it
knows, does it abandon its own answer — and can you tell from its reasoning
trace *before* it answers?

Started at the UC Berkeley AI Hackathon (June 2026), continued as an
independent research project. ~25,000 API calls across 12 experiments on
Claude Sonnet 4.6, with a cross-model replication on GPT-5.4.

---

## The setup

1,756 trivia questions from TriviaQA, each filtered so the model **provably
knows the answer** — it had to answer correctly 3/3 times at temperature 0.
This gate is the most important design decision in the project: without it,
"the model gave the wrong answer" is indistinguishable from "the model never
knew," and the entire result would be about knowledge gaps rather than belief
override.

For each question, a fabricated academic paper asserting a *wrong* answer —
with invented authors, years, and venues, generated deterministically from a
hash of the question so the corpus is reproducible. Documents were made
maximally authoritative so capitulation couldn't be dismissed as the model
correctly discounting a weak source.

The model then sees the document(s) plus the raw question, with extended
thinking enabled. Its reasoning trace and final answer are recorded. A separate
judge model labels each answer **resisted / capitulated / hedged** — critically,
the judge sees only the answer, never the trace, so the classifier's target
isn't contaminated by the signal it's meant to learn.

Independent variable: number of corroborating documents (1, 2, or 3).

---

## The core finding

**Whether the model deliberates at all is the dominant predictor of
resistance** — far more than anything about the documents.

| Reasoning trace | Share of calls | Capitulation rate |
|---|---|---|
| Thin (≤200 chars) | 46% | **91%** |
| Thick (>200 chars) | 54% | **16%** |

A gradient-boosting classifier on trace features predicts capitulation at **84%
accuracy** — and reaches **94% from the first ~40 reasoning tokens**, before a
single answer token is generated.

Two manipulations that *should* have mattered didn't. Source authority (tested
in a predecessor project) showed no effect. Document count produced only a weak
effect — and one that **reverses direction between model families** (see
cross-model, below). The pressure isn't coming from the documents; it's about
whether the model engages.

---

## Establishing that deliberation *causes* resistance

The correlation above is easy to explain away: maybe confident knowledge
produces both long traces and resistance, and the trace is just a symptom. Three
experiments closed that off.

**Conflict control.** Same document machinery, same fake citations — but the
document asserts the *correct* answer. Result: 99.8% correct, 95% thin traces.
Deliberation is *conflict-triggered*, not difficulty-triggered, and the odd
document format isn't what causes it. A bonus finding: an agreeing document
suppresses thinking *below* the no-document baseline — corroboration actively
short-circuits deliberation.

**Ablation.** Same questions, same documents, thinking disabled. Capitulation +
hedging rose from 58.0% to **71.5%** (+13.5pp, p≈10⁻³⁹). Removing deliberation
causally increases capitulation.

**Rescue with mediation.** Take every thin-trace capitulation and re-run it two
ways: a control (identical prompt, fresh sample) and a nudge ("Think through
this thoroughly before answering" — deliberately content-neutral, saying nothing
about documents or accuracy). The control confirms thin capitulation is stable
at 1.1%. The nudge lifts resistance to 26.9%.

The mediation split is what makes this causal at the per-question level:

| Nudge result | Resisted |
|---|---|
| Trace stayed thin (46% of cases) | 0.7% — *identical to control* |
| Moderate lengthening | 5.3% |
| Long traces (>600 chars) | **60.6%** |

Where the instruction failed to lengthen the trace, it did nothing at all. The
effect is *fully mediated* by realized deliberation — the words carry no
independent power.

---

## Is it truth-tracking, or just stubbornness?

The sharpest objection to everything above: every question was one where the
model's prior was *correct*, so "resisting the document" and "being right" are
the same act. Deliberation might simply entrench whatever the model already
believed, looking like accuracy only because of how questions were selected.

To test this, I harvested 200 questions where the model holds a **stable false
belief** — wrong at temp 0, and the *same* wrong answer across repeated
temp-1 resamples, with a semantic-equivalence judge so paraphrases count as the
same belief. Screened from 6,899 candidates (15.5% of TriviaQA questions are
answered wrong; ~62% of those are stable).

Then two arms: a document asserting the **correct** answer (a correction), and
a document asserting **the model's own wrong answer** (agreement). The second
required asserting the model's *specific* stored belief — an arbitrary third
falsehood would have created a three-way conflict and made the traces
uninterpretable.

Combined with the earlier work, this completes a 2×2:

| | Document says TRUTH | Document says FALSEHOOD |
|---|---|---|
| **Prior correct** | 99.8% right · 95% thin | 46% capitulate · thick when resisting |
| **Prior wrong** | **91.5% corrected** · 82% thin | **95.5% stays wrong** · 90% thin |

**The answer is truth-tracking.** The model defends a *true* belief against a
false document ~54% of the time, but defends a *false* belief against a true
correction only **1.5%** of the time. False priors are held weakly; true priors
strongly. Deliberation isn't stubbornness.

The diagonal is the mechanism: deliberation fires only on disagreement, and in
both disagreement cells it pushes toward truth. Both agreement cells produce
thin traces regardless of whether the shared belief is true or false.

---

## Does it generalize?

Replicated on **GPT-5.4** (cost-matched to Sonnet 4.6, so any difference can't
be attributed to capability tier), n=5,171, same documents, same conditions,
same judge for label consistency. Because reasoning-token counts run on a
different scale than Claude's trace lengths, thin/thick was derived from each
model's own distribution.

**Replicates:** thin-trace capitulation is **82.7% in both models**. Gaps of
72.6pp and 55.1pp, both p < 10⁻³⁰⁰. The core finding is not a Claude artifact.

**Reverses:** the document-count effect runs *opposite* directions — Claude 43.3%
→ 47.6% across 1→3 documents, GPT 60.7% → 49.7%. Both monotonic, opposite signs.
Corroboration is not a universal pressure, and no general "more sources = more
capitulation" claim survives.

---

## What this is good for (and what it isn't)

**The deployable result is a prompt, not a filter.** Appending one deliberation
sentence to RAG prompts cuts capitulation ~40% on the full false-document
population (36.5% → 21.5%). Cost: **+67 output tokens per query** (~$0.001).
Harm on truthful documents: accuracy 99.5% vs 99.8% — and manual inspection
showed 5 of the 6 "errors" were **stale ground-truth labels the deliberating
model correctly overrode** (e.g. largest African country → Algeria post-2011).
Genuine harm ≈ 0–1 in 1,722. Essentially free, essentially harmless.

**The detector does not work as an answer firewall**, and establishing that was
one of the more valuable results. Truthful documents also produce thin traces
(94.9%), so at a realistically truthful corpus the thin-trace flag fires on ~90%
of queries at 4.2% precision. Within thin traces, ground-truth-free features are
near chance.

The reason is structural: the trace detects **absence of deliberation**, not
**document falsity**. A model that never engages surfaces no knowledge to
compare against, so there's nothing to adjudicate. Distinguishing model-vs-
document requires external ground truth by definition — no trace signal can
supply it.

What survives as applied value: **prompt-level hardening** (the nudge),
**corpus auditing** (thick traces flag documents that conflict with model
knowledge at ~55% precision — a batch tool for finding wrong, stale, or
adversarial documents, including ones matching your own outdated labels), and
**adversarial settings** where documents are untrusted by default and the base
rates work.

---

## Infrastructure

Concurrent multi-threaded pipeline (10–30 workers) with per-call Redis caching,
incremental progress files, and crash recovery — which mattered: several runs
survived process kills and an unplanned laptop shutdown with zero lost work,
resuming from cache rather than re-paying. OpenTelemetry tracing through Phoenix
for observability. Deterministic document generation for reproducibility.
Roughly $80 of API spend total.

---

## Known limitations

- **Judge is unvalidated** — Claude judging Claude, no human agreement numbers.
  The most attackable methodological point.
- **Single sample per cell at temp 1** — 19.4% of questions flip thin/thick from
  sampling noise alone, so per-question claims are noisy (aggregate claims are
  fine at n=1,756).
- **Stale gold labels** in TriviaQA, surfaced repeatedly by the model itself.
- **~3% document generation failure** rate (documents that fail to assert the
  intended answer).
- **Trivia domain only** — untested on medical, legal, or financial content.
- **Correlational on faithfulness** — the mediation results show reasoning is
  causally load-bearing, but whether traces *accurately describe* their own
  reasoning was never tested directly.
