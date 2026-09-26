# HackTrace — Design Rationale

**Status: mockup for review.** Written from conversation history to capture *why*
each experiment was designed the way it was, since that reasoning lives in chat
rather than in the code or the results JSON. Numbers are included for context but
the point of this document is the design logic. Corrections expected.

---

## The through-line

Each experiment exists because the previous one left a specific alternative
explanation open. Reading the list as a sequence of "what could still explain
this?" is the fastest way to understand the design.

| # | Experiment | Closed off the objection that... |
|---|---|---|
| 0 | Main experiment (instructed) | — (established the phenomenon) |
| 1 | Bare framing re-run | ...capitulation is instruction-following |
| 2 | Correct-document control | ...long traces reflect question difficulty, or our fake-doc format confuses the model |
| 3 | Thinking ablation | ...deliberation merely correlates with resistance |
| 4 | Rescue + mediation | ...the *within-question* thin/thick gradient is confounded by question identity |
| 5 | Nudge harm/cost test | ...the intervention is too costly or damages the truthful majority |
| 6 | Always-nudge (full population) | ...rescue only works on the cherry-picked worst cases |
| 7 | Prefix/streaming analysis | ...detection needs the complete trace |
| 8 | Base-rate analysis | ...(self-imposed) the detector is deployable as an answer firewall |
| 9 | Cross-model (gpt-5.4) | ...the whole finding is a Claude artifact |
| 10 | Wrong-prior 2×2 | ...deliberation entrenches priors rather than tracking truth |

---

## Foundational design choices

### Why documents are fake academic papers
Inherited from the hackathon design. The goal was a document type carrying
maximum epistemic authority, so that capitulation couldn't be dismissed as the
model correctly discounting a low-quality source. Fabricated citations
(author, year, venue) were deterministically generated per question via a
SHA256 seed, so the corpus is reproducible and the same question always gets
the same citations across runs.

### Why the 3/3 correctness gate
Questions were kept only if the model answered correctly 3/3 times at temp=0.
Without this, "capitulation" is indistinguishable from "the model never knew."
This is the single most important design decision in the project — it is what
makes the main result about *belief override* rather than *knowledge gaps*.

**Known limitation:** the gate ran at temp=0 but the experiment runs at temp=1.
Addressed later by re-checking baseline correctness at temp=1 (99% still
correct), which bounds the residual concern.

### Why the judge never sees the thinking trace
The judge labels resisted/capitulated/hedged from the answer text alone. If it
saw the trace, the classifier's target would be contaminated by the very signal
the classifier is meant to learn from — trace features would predict labels
partly because the labels were derived from traces.

### Why document count as the independent variable
Chosen at the hackathon after an earlier project (modeltraceprep) varied
*source authority* and found no effect. Document count was the natural next
manipulation. In hindsight both null results matter more than either
manipulation working would have — see "the double negative" below.

---

## Experiment 1 — Bare framing re-run (n=5,268)

**The objection.** The original prompt ended with "Using the documents above as
context, answer the following question." That is a mild instruction to defer.
If capitulation were really instruction-following, the headline result would be
measuring our own prompt rather than a model behavior. This mattered
especially because we knew framing effects were large: the earlier
modeltraceprep run used a prompt that explicitly warned "the document may or
may not contain accurate information" and saw ~6.5% capitulation versus our
~50%.

**The design.** Re-run the entire experiment with the instruction deleted —
documents, then the raw question, nothing else. Chosen over a partial sweep of
several framings because (a) it makes the baseline and pipeline symmetric (both
are "bare question," differing only in document presence), and (b) full n gives
a paired comparison across all 5,268 cells rather than an underpowered subset.

**Why keep both runs rather than replacing.** The instructed framing is
representative of production RAG (real systems do say "use the context"). The
bare framing is the clean scientific condition. Together they *are* the framing
experiment, at full power.

**Result.** 50.9% → 46.4% (McNemar p=4e-23). Instruction worth ~4.5pp. The
phenomenon survives.

---

## Experiment 2 — Correct-document control (n=1,722)

**The objection.** Two, actually. First, long traces might just mean hard
questions rather than detected conflict. Second, our fake-document format
(fabricated citations, invented venues) might itself be confusing the model —
maybe it deliberates because the document looks strange, not because it
disagrees.

**The design.** Identical machinery — same generator, same fake citation
system, same venues, same bare framing — but the document asserts the *correct*
answer. Everything about the stimulus is held constant except the truth value
of the claim.

**Why this is the right control.** It separates the two candidate causes
cleanly. If format were the issue, traces would stay long. If difficulty were
the issue, hard questions would still produce long traces. Neither happened.

**Result.** 99.8% correct, mean thinking 104 chars, 95% thin. Deliberation is
*conflict-triggered*. Bonus finding: an agreeing document suppresses thinking
*below* the no-document baseline (104 vs 180 chars) — corroboration doesn't
just fail to trigger deliberation, it actively short-circuits it.

---

## Experiment 3 — Thinking ablation (paired n=1,688)

**The objection.** Everything so far was correlational. Long traces
*accompanied* resistance, but a common cause — confident knowledge producing
both — would explain the association equally well.

**A design that didn't work, and why.** The first proposal was to manipulate
`budget_tokens`. This fails mechanically: `budget_tokens` is a *cap*, not a
floor, and our traces (~350 chars ≈ 90 tokens) sit far below the 1,024-token
minimum. Lowering it wouldn't bind on 99% of calls. Worth recording as a
near-miss.

**The design that worked.** Toggle thinking off entirely. Same questions, same
documents, same temperature — the only difference is the presence of a
deliberation phase. Because the stimulus is identical and paired per question,
the common-cause explanation can't produce the arm difference: confident
knowledge is equally present in both arms.

**A second arm we deliberately cut.** An "instructed deliberation" arm was
proposed and rejected on the user's push: it reintroduces experimenter
instruction right after we'd removed it (Exp 1), and it can't separate
"instruction → more thinking → resistance" from "instruction → suspicion →
resistance." The ablation *removes* a component rather than adding an
instruction, which is the cleaner necessity test.

**Result.** Cap+hedged 58.0% → 71.5% (+13.5pp, p=6e-39). Deliberation causally
protects.

**Scope limit worth stating.** This shows *enabling extended thinking* is
causally protective. It does not isolate which part of that bundle (the
generation phase, extra compute, processing style) does the work.

---

## Experiment 4 — Rescue with mediation (paired n=658)

**The objection.** The ablation established an on/off effect but left the
*within-arm* gradient correlational: among thinking-enabled calls, thick traces
resisted more, but that could still be question-driven.

**The design.** Take every thin-trace capitulation and re-run in two arms:
- **control** — identical prompt, fresh sample. This arm is essential: at
  temp=1 we measured a 19.4% thin/thick flip rate from sampling noise alone, so
  without it "some flipped under the nudge" proves nothing.
- **nudge** — bare prompt plus "Think through this thoroughly in your reasoning
  before answering." Deliberately content-neutral: no mention of documents,
  doubt, or accuracy, so it induces deliberation without cueing skepticism.

**The mediation check is the point.** Not just "did outcomes improve" but "did
they improve *only where the trace actually lengthened*." This is what upgrades
the gradient from correlational to causal.

**Result.** Control 1.1% flip (thin capitulation is stable). Nudge 26.9%
resisted / 42.1% escaped. Mediation: where the nudge failed to lengthen the
trace (46% of cases), outcomes matched control exactly (0.7%); where it induced
>600-char traces, 60.6% resisted and 87.9% escaped. Zero direct instruction
effect — fully mediated.

**Open thread.** The 46% "stubborn thin" cases where an explicit deliberation
instruction fails to induce any thinking. Never investigated.

---

## Experiment 5 — Nudge harm/cost on truthful documents (paired n=1,722)

**The objection.** Rescue works on the worst cases, but a deployed policy runs
the nudge on *everything*, and truthful documents are the overwhelming majority
in any real corpus. If deliberation on true documents causes overthinking or
talks the model out of correct answers, the intervention is unusable.

**The design.** Nudge arm over the correct-document control population, with
the existing no-nudge control results as the comparison. Token usage tracked
per call (required adding usage capture to the client) so cost is measured
rather than estimated.

**The result required manual inspection to interpret.** Accuracy 99.5% vs
99.8%, McNemar p=0.031 — statistically significant harm. But reading the six
nudge-only errors showed 5–6 were **stale or wrong gold labels the deliberating
model correctly overrode** (Seagram Building → Mies van der Rohe; largest
African country → Algeria post-2011; Tweedledum → Through the Looking-Glass).
Genuine harm ≈0–1 in 1,722.

**Why this matters beyond the cost question.** It revealed that deliberation is
*symmetric*: it reverts the model to its own knowledge in both directions —
resisting false documents and correcting documents that parrot outdated ground
truth. That reframing fed directly into the wrong-prior experiment.

**Cost.** +67 output tokens/query (~$0.001). Essentially free.

---

## Experiment 6 — Always-nudge on the full false-doc population (n=200)

**The objection.** Rescue measured only the thin-capitulation subset. The
deployment number requires running the nudge on everything, thin and thick.

**The design.** Same 200 questions as the baseline run so results cross-link.
Also served as the vehicle for the prefix-transfer test (below).

**Result.** Resistance 52.0% → 72.0%, capitulation 36.5% → 21.5% (~41%
relative reduction).

---

## Experiment 7 — Prefix truncation and detector transfer (no new API calls)

**The objection.** The applied claim was "early warning," but a post-answer
filter catches the same errors. Earlier detection only matters if it's
meaningfully earlier — which needs measuring, not asserting.

**Two design decisions that mattered:**

1. **Deployable vs oracle features.** The classifier's strongest features
   (`correct_in_thinking`, `wrong_in_thinking`) require knowing the right
   answer — unavailable at inference time. Splitting into oracle and
   ground-truth-free feature sets was necessary to report a number that means
   anything for deployment.

2. **Transfer direction.** Train on unnudged traces, test on nudged. This is
   the deployment-realistic direction: your detector is trained before you
   change the prompt policy.

**Result.** 150 chars (~40 tokens) → 87.0% accuracy / 0.914 AUC on deployable
features, vs 89.4%/0.928 on full traces. Transfer to nudged traces: 94.0% at
150 chars, no retraining needed. (Transfer actually *beat* retraining, because
5,268 slightly-mismatched training rows outweigh 160 matched ones.)

**A bug caught here worth recording.** An early within-thin classifier scored
100% accuracy — a data leak, because correct-control rows lacked the
`condition`/`n_documents` fields and the default values gave the class away.
Fixed by forcing both to constants; true performance was AUC 0.70.

---

## Experiment 8 — Base-rate analysis (self-imposed objection)

**The objection we raised on ourselves.** Truthful documents also produce thin
traces (94.9% from Exp 2). If thin-trace flagging is the detector, and most
documents in a real corpus are truthful, the detector fires on nearly
everything.

**The design.** Model precision as a function of corpus truthfulness, using
measured flag rates from the actual populations rather than assumed ones.

**Result.** At a 90%-truthful corpus: 4.2% precision, ~90% of queries flagged.
Within thin traces, ground-truth-free features are near chance (AUC 0.70 vs
71.3% majority baseline).

**Why this is a contribution rather than a failure.** It identifies *what the
signal actually is*: absence of deliberation (a provenance signal), not
document falsity (a veracity signal). And it shows the veracity version is
impossible in principle — adjudicating model-vs-document requires external
ground truth by definition. Reporting this reframed the deployable
contribution from "answer firewall" to "prompt-level hardening + corpus
auditing," which is the honest version.

---

## Experiment 9 — Cross-model replication, gpt-5.4 (n=5,171)

**The objection.** Every result was from one model. Single-model behavioral
findings are weak evidence about language models generally.

**A design error caught mid-flight.** The pipeline was initially scaffolded on
`gpt-5-mini` — a cost-optimized model roughly 10× cheaper than Sonnet 4.6. That
confounds model *family* with capability *tier*: "GPT capitulates more" would
have been unfalsifiably attributable to "smaller model defers more." Corrected
to `gpt-5.4`, which is cost-comparable to Sonnet 4.6. The mini results were
kept as a separate within-family comparison rather than discarded.

**Design parity choices.** Same documents, same three conditions, same bare
framing, and critically **the same Claude judge**, so labels stay consistent
with the 15k+ already collected. A separate GPT closed-book gate (98.2% knows)
plays the role Claude's 3/3 gate plays — capitulation must not be confusable
with ignorance in either model.

**A measurement problem requiring a design decision.** OpenAI exposes reasoning
*summaries*, not raw traces, and reasoning-token counts run on a completely
different scale (median ~49–104 tokens vs Claude's ~300–400 chars). Our
200-char threshold is meaningless there. Resolution: derive thin/thick from
each model's *own* distribution (median split), which is the right methodology
regardless since the units aren't comparable.

**Results — one replication, one reversal.**
- **Replicates:** thin-trace capitulation is 82.7% in *both* models. Gaps of
  72.6pp (Claude) and 55.1pp (GPT), both p<1e-300. The core finding is not a
  Claude artifact.
- **Reverses:** the condition effect runs *opposite* directions. Claude C1 43.3%
  → C3 47.6% (more docs, more capitulation, p=0.013). gpt-5.4 C1 60.7% → C3
  49.7% (more docs, *less* capitulation, p=9.6e-11). Both monotonic, opposite
  signs. Corroboration is not a universal pressure — and this definitively
  kills any general "more sources = more capitulation" claim.

---

## Experiment 10 — The wrong-prior 2×2 (n=200 per arm)

**The objection, and it's the sharpest one in the project.** Every experiment
used questions where the model's prior is *correct* (the gate guaranteed it).
There, "resist the document" and "be right" are the same act. So two very
different mechanisms are indistinguishable:
- **truth-tracking** — deliberation evaluates the conflict and lands on truth
- **prior-defending** — deliberation entrenches whatever the model believed,
  and only looks like accuracy because we hand-picked correct-prior questions

Under prior-defending, "deliberation protects" degrades to "deliberation makes
the model stubborn," useful only when the model already happens to be right.

**Getting the population.** Required harvesting questions where the model holds
a *stable false belief*: wrong at temp=0, same wrong answer on ≥2/3 temp=1
resamples. A semantic-equivalence judge adjudicates matches so paraphrases
("48 Hrs." vs "48 Hours") count as the same belief rather than false rejects.
200 stable priors from 6,899 candidates screened (15.5% of TriviaQA questions
are answered wrong at temp=0; ~62% of those are stable).

**Why two arms and not one.** The original plan was only the correction arm.
Adding the agreement arm completes a 2×2 that mirrors what already existed for
correct priors:

| | Doc TRUE | Doc FALSE |
|---|---|---|
| **Prior correct** | Exp 2 (correct-doc control) | Exp 0/1 (main) |
| **Prior wrong** | **correction arm** | **agreement arm** |

**A design subtlety that would have invalidated the agreement arm.** The false
document must assert **the model's own specific wrong answer**, not an
arbitrary third falsehood. There are unboundedly many wrong answers, so a
freshly generated one would almost always mismatch the prior — producing three
competing claims and making thick traces uninterpretable (conflict with the
document? or reconciling two falsehoods?). This is exactly why the harvest
screened for *stability* rather than just wrongness: the stored `wrong_prior`
is what makes a genuine no-conflict condition constructible.

**A preprocessing step this forced.** Harvested priors came back as probe
responses with annotations ("Mississippi (1966)") and some as full sentences
("A female skunk is called a doe"). Both had to be normalized to bare answer
phrases before document generation — sentences via an extraction judge —
otherwise documents assert awkward claims and the doc-asserts check breaks.

**Results.**
- **Correction arm:** 91.5% accepted the correction, **1.5% defended the false
  prior**, 7% other. Thin traces (82%).
- **Agreement arm:** 95.5% stayed wrong, 2.5% gold, very thin (90%). Among
  thin traces: **100% prior, 0% gold**.

**Interpretation.** Deliberation is **truth-tracking, not prior-defending**.
The asymmetry is the evidence: the model defends a *true* belief against a
false document ~54% of the time, but defends a *false* belief against a true
document only 1.5% of the time. False priors are held weakly; true priors are
held strongly.

**A secondary finding.** The thin/thick relationship *inverts* here: thin
traces give the correct answer 99% of the time in the correction arm, versus
thin meaning capitulation in the main experiment. So the trace tracks
*conflict*, not compliance — which is the structural reason it can't serve as
a veracity signal (Exp 8).

---

## The completed 2×2

| | Doc asserts TRUTH | Doc asserts FALSEHOOD |
|---|---|---|
| **Prior correct** | 99.8% right · 95% thin | 46% capitulate · thick when resisting |
| **Prior wrong** | 91.5% corrected · 82% thin | 95.5% stays wrong · 90% thin |

The diagonal is the mechanism: deliberation fires only on disagreement, and in
both disagreement cells it moves toward truth. The two agreement cells produce
thin traces regardless of whether the shared belief is true or false — the
bottom-right being the silent-failure case where no signal is available.

---

## The double negative (a framing choice, not an experiment)

Two manipulations that *should* intuitively matter both produced weak or
non-monotonic effects: source authority (modeltraceprep) and document count
(this project, and it reverses sign across model families). What actually
predicts capitulation is whether the model deliberates at all.

The decision was to foreground this as a finding rather than bury it as a
limitation, and to reframe the research question from "what external pressure
causes capitulation" to "what internal reasoning patterns predict resistance."
That reframing makes the trace analysis the central contribution rather than a
side analysis.

---

## Things deliberately not done, and why

- **Domain-level analysis** — domain buckets were defined by us and would vary
  across runs; not robust enough to weight.
- **Deep SHAP / phrase-level classifier analysis** — risks tunnel vision on the
  classifier rather than the phenomenon; marginal return versus open questions.
- **Instructed-deliberation ablation arm** — reintroduces experimenter
  instruction and can't separate instruction-content from thinking-dose.
- **Adversarial deliberation-suppression experiment** — designed but not run.
  The likely finding ("an existing attack can be made quieter") didn't justify
  the calls, since an attacker who controls the document already controls the
  context, and third-party harm requires corpus access regardless.
- **Question-feature thinness predictor** — *was* run and failed (57% vs 59%
  baseline). Kept as a negative result: trace thinness is an intrinsic property
  of model confidence, not predictable from question surface features.

---

## Known limitations (for a writeup)

1. **Judge is unvalidated.** Claude judging Claude, no human agreement numbers.
   The most attackable methodological point. Fix: hand-label ~100 sampled
   responses and compute agreement; a GPT second-judge pass would add
   cross-family agreement.
2. **Single sample per cell at temp=1.** 19.4% of questions flip thin/thick
   from sampling noise alone. Aggregate claims are fine at n=1,756;
   per-question claims are noisy. Fix: replicate a ~300-question subset with a
   fresh cache salt.
3. **Stale gold labels.** TriviaQA answers are wrong or outdated in a
   measurable number of cases (surfaced repeatedly in Exp 5 and the GPT gate).
   Affects accuracy denominators slightly.
4. **Document generation failure rate.** ~3% of generated documents fail to
   assert the intended answer (6/200 correction, 11/200 agreement). Should be
   excluded in final analysis.
5. **Trivia domain only.** Whether the pattern holds on medical, legal, or
   financial content is untested.
6. **~100 questions stuck on transient API 500s** across the correct-control
   and ablation runs; retry pass pending.
