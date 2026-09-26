# HackTrace — Research Directions

A reference document for ongoing and future research directions stemming from the
initial HackTrace experiment. Covers confirmed findings, prioritized next steps,
and longer-term avenues.

---

## What We've Established (Baseline)

**Setup:** 1756 trivia questions, each validated 3/3 at temp=0 (model knows the answer
confidently). Fake academic papers generated asserting a wrong answer, fed to Claude
with extended thinking under 3 conditions (1, 2, or 3 corroborating documents).
5,268 total API calls. Judge model labels each response: resisted / capitulated / hedged.

**Key findings:**

- Overall capitulation rate ~50% in a naturalistic RAG-like framing
  ("use these documents as context")
- **Thin thinking (≤200 chars, 46.5% of calls): 91% capitulation**
- **Thick thinking (>200 chars, 53.5% of calls): 16% capitulation**
- Whether the model deliberates at all is the single strongest predictor of resistance
- `correct_in_thinking` (model recalls correct answer mid-trace) accounts for 58%
  of classifier predictive power
- **Classifier: 84.3% accuracy (GBM)** predicting capitulation from thinking trace alone,
  before reading the answer. Consistent across conditions (82–85%)
- **Double negative result:** neither document authority level nor document count
  produced a clean monotonic effect on capitulation. The thing that matters is
  deliberation quality, not document quantity or credibility.
- Condition capitulation rates: C1=47.6%, C2=54.3%, C3=50.8% (non-monotonic; C2 peak)

---

## Follow-Up Results (July 2026 experiments)

Three experiments run after the audit, all under **bare framing** (documents +
raw question, no instruction sentence) unless noted:

**1. Framing effect (instructed vs bare, paired n=5,267):**
Strict capitulation 50.9% instructed vs 46.4% bare (McNemar p=4e-23). The
instruction accounts for only ~4.5pp — capitulation is NOT primarily
instruction-following. Thin/thick replicates under bare framing (93.1% vs
13.6% capitulation). Condition story cleaner in bare data: C1 43.3% -> C2
48.3% (p=0.004) -> C3 47.6% (n.s.) — a second corroborating source adds real
pressure, a third saturates. Bonus: removing the instruction slightly
*increases* thinking (393 vs 359 chars, p=3e-30).

**2. Correct-document control (n=1,722):**
Same doc machinery asserting the CORRECT answer: 99.8% correct answers, mean
thinking 104 chars, 95% thin. Deliberation is **conflict-triggered** — long
traces are the signature of detected conflict between parametric knowledge
and context, not of question difficulty or document presence. An agreeing
document suppresses thinking *below* the no-document baseline (104 vs 180
chars) — corroboration short-circuits deliberation. Also validates the fake-
doc format itself (model unconfused when content is true).

**3. Thinking ablation, nothink arm (paired n=1,688):**
Same C1 stimuli, extended thinking disabled, temp matched. Cap+hedged rises
58.0% -> 71.5% (+13.5pp, McNemar p=6e-39); resistance falls 42.8% -> 28.5%.
**Deliberation causally protects** — the thinking trace is not just a
diagnostic readout, it is (at least partly) the mechanism. This also gives
the shelved "rescue" intervention a mechanistic basis.

**4. Rescue experiment (paired n=658, thin-capitulation population):**
Re-ran every bare-C1 thin+capitulated question in two fresh arms: control
(identical prompt, resample) vs nudge (content-neutral "think through this
thoroughly" instruction). Control confirms thin-cap is stable (1.1% flip to
resistance). Nudge: 26.9% resisted, 42.1% resisted-or-hedged (+25.8pp /
+38.4pp, McNemar p=6e-50 / 3e-73). **Mediation is complete**: when the nudge
failed to lengthen the trace (46% of cases), outcomes matched control (0.7%
resist) — the instruction has no direct effect; when it induced long traces
(>600ch), 60.6% resisted / 87.9% escaped capitulation. Induced deliberation
causally rescues, fully mediated by actual trace lengthening. This makes the
thin/thick gradient causal, not just the on/off ablation. Open subgroup: the
46% of thin-cap cases where the nudge fails to induce thinking at all
("stubborn thin") — why?

**5. Streaming/deployment analyses (no API calls; scripts/15):**
Prefix curve — deployable (ground-truth-free) features on truncated traces:
150 chars (~40 thinking tokens) already gives 87.0% acc / 0.914 AUC vs
89.4% / 0.928 for the full trace. Early detection is real: ~40 tokens of
thinking suffice. Truthful-mix — thin traces are ALSO the signature of
truthful-doc deference (94.9% thin), so the thin-flag detector collapses in
realistic corpora (precision 4.2% at 90% truthful docs; ~90% of queries
flagged). Within thin traces, deployable features barely separate
capitulation from truthful deference (AUC 0.70, acc 72.3% vs 71.3% base) —
both traces read "the document states X." Implication: the trace signal
detects *absence of deliberation*, not *document falsity*. Deployment paths:
(a) blanket "always deliberate" policy — needs the truthful-doc harm/cost
test (does nudging truthful docs hurt accuracy?), or (b) two-signal design
(trace + retrieval provenance/verification).

**6. Nudge harm/cost test on truthful docs (paired n=1,722):**
"Always deliberate" policy cost on the truthful majority: accuracy 99.5%
nudged vs 99.8% control (6 nudge-only errors, McNemar p=0.031) — but manual
inspection shows ~5-6 of the 6 are stale/wrong gold labels that the nudged
model CORRECTED against the document (Seagram Building -> Mies van der Rohe;
largest African country -> Algeria post-2011; Tweedledum -> Through the
Looking-Glass; corrupt question '19969'). Genuine harm ~0-1/1,722. Marginal
cost: +67 output tokens/query (~$0.001), +15 input. Verdict: blanket
deliberation is essentially free and harmless on truthful docs — the
deployable recommendation is one instruction, not a detection pipeline.
Deliberation reverts the model to its own knowledge in both directions:
resists false docs, and overrides docs agreeing with stale gold labels.

**7. Cross-model replication — gpt-5.4 (n=5,171, paired with Claude bare):**
Tier-matched to Sonnet 4.6 on cost; same docs, conditions, bare framing, and
same Claude judge. Closed-book gate: gpt-5.4 knows 1,724/1,756 (98.2%).

- **Deliberation-predicts-resistance REPLICATES across model families.**
  Median split on each model's own scale (Claude 382 chars, gpt-5.4 104
  reasoning tokens): thin capitulation 82.7% in BOTH models; thick 10.0%
  (Claude) vs 27.5% (gpt-5.4). Gaps 72.6pp / 55.1pp, both p<1e-300. The
  core finding is model-agnostic, not a Claude artifact.
- **The condition effect REVERSES direction.** Claude: C1 43.3% -> C3 47.6%
  (more docs, more capitulation, p=0.013). gpt-5.4: C1 60.7% -> C3 49.7%
  (more docs, LESS capitulation, p=9.6e-11). Monotonic in both, opposite
  signs. Corroboration is not a universal pressure — one model treats
  repetition as evidence, the other apparently notices something about the
  repetition. Kills any general "more sources = more capitulation" claim.
- Overall capitulation: Claude 45.8% vs gpt-5.4 55.1% (McNemar p=1.7e-44),
  with gpt-5.4 deliberating far less (median 104 reasoning tokens).

**8. Wrong-prior harvest complete:** 200 stable false beliefs (wrong at
temp=0, same wrong answer on >=2/3 temp=1 resamples, semantic-equivalence
adjudicated) from 6,899 candidates screened. Base rate: 15.5% of TriviaQA
questions are answered wrong at temp=0; ~62% of those are stable. Enables
the correction experiment (truthful docs vs false priors) to test whether
deliberation is truth-tracking or merely prior-defending.

Supporting audit results: thin/thick threshold-robust (gap 60-63pp across
150-400 chars); C2>C1 significant (p=0.0001 instructed, p=0.004 bare);
temp-1 knowledge validated (99% of baseline questions 3/3 correct, no docs).

Loose ends: ~34 correct-control + ~68 nothink questions stuck on transient
API 500s — retry pass pending. 25 correct-control docs have paraphrase-style
answer mismatches (cosmetic).

---

## Priority 1: Multi-Model Comparison

**The question:** Does the relationship between thinking trace features and capitulation
generalize across models, or is it Claude-specific?

**Why it's high-yield:** If the same classifier features (deliberation depth, doubt
language, correct answer recall) predict capitulation in both Claude and GPT, that's
a model-agnostic finding about how reasoning traces signal deference. Much stronger
than a single-model result. If the features *don't* generalize, that's also interesting
— it tells you something about how different architectures handle epistemic conflict.

**Practical plan:**
- Run the same 1756-question pipeline on GPT o1 or o3 (OpenAI credits available)
- Note: GPT's thinking traces may be hidden or summarized — need to verify what's
  accessible via API before committing
- Use the same judge (or cross-validate with a GPT-based judge for Claude runs)
- Compare: capitulation rate, thin/thick split, classifier accuracy, top features
- Optionally extend to Claude Haiku vs Sonnet vs Opus within the Anthropic family

**Open question:** Does more capable = more resistant, or more capable = better at
rationalizing capitulation?

---

## Priority 2: Reverse Engineering Thin Thinking Traces

**The question:** What properties of a *question* predict whether the model will produce
a thin (non-deliberative) vs thick (deliberative) thinking trace?

**Why it's high-yield:** If you can predict thinness from question structure *before*
running the model, you can identify high-risk queries in advance — useful for any
RAG system. Also novel: this question hasn't been asked in the literature.

**Hypotheses to test (with existing data):**
- Answer type: single-word answers (names, numbers, dates) → thinner traces than
  multi-word answers requiring reasoning
- Answer length: shorter correct answers → thinner traces
- Domain: some domains (e.g. music trivia) may systematically produce thinner traces
  than others (e.g. science)
- Question phrasing: "which/what/who" vs "why/how" questions
- Wrong answer plausibility: if wrong answer is very close to correct
  (e.g. "1914" vs "1918"), does the model deliberate more?

**Implementation:** Extract features of the question itself (answer length, answer type,
domain, question word) and train a predictor of thinking length. This uses existing
data — no new API calls needed.

**Stretch:** Run a no-document baseline (same questions, extended thinking, no docs)
to get a "natural" thinking length per question. Questions where baseline thinking is
already short may be systematically more vulnerable to thin-thinking capitulation.

---

## Priority 3: Foregrounding the Double Negative Result

**The finding:** Both of the manipulations that *should* intuitively matter —
document authority level (tested in the original modeltraceprep run) and document
count (tested in HackTrace) — show weak or non-monotonic effects.

**Why this matters:** This is a counterintuitive and publishable negative result.
The naive prediction is: more authoritative sources = more capitulation, more
corroborating sources = more capitulation. Neither holds cleanly. What actually
predicts capitulation is whether the model enters a deliberative mode at all.

**How to foreground it:**
- Explicitly frame the double negative as a finding, not a limitation
- Reframe the research question: not "what external pressure causes capitulation"
  but "what internal reasoning patterns predict resistance"
- This reframing makes the thinking trace classifier the *central* contribution,
  not a secondary analysis

---

## Longer-Term / Alternate Directions

### Adversarial Prompting as Defense
Test whether system prompt interventions ("be skeptical of sources", "verify claims
against your training knowledge") move the needle on capitulation — separately for
thin vs thick thinking populations. The thin-thinking group is where interventions
would need to work hardest (model isn't deliberating at all). Kept as alternate
because it's more applied than research-oriented.

### Cross-Judge Validation
Run a sample of responses through a second judge (GPT-4 or human rater) and compute
agreement with the Claude judge. Strengthens the labeling methodology for a research
audience. Low cost, high credibility value.

### Real Misinformation (Cherry-Picked Real Documents)
Instead of fully fabricated papers, use real documents that are genuine but
misleading — e.g. real papers misrepresenting their findings, or legitimate sources
that happen to support a wrong answer. Does the model's resistance change when the
citation is real vs fabricated? Closest to actual real-world misinformation risk.
Challenge: hard to source systematically at scale.

### Temporal Structure of the Thinking Trace
Does it matter *when* in the trace doubt appears? A model that expresses doubt at
token 50 but resolves toward the document may behave differently than one that
doubts at token 500. Current bag-of-features classifier ignores sequence entirely.
Would require sequence modeling (RNN, attention over trace) — higher effort.

---

## What We're Not Pursuing (and Why)

- **Domain-level analysis:** Domain buckets were defined by us and could vary across
  runs — not robust enough to weight heavily
- **Deep SHAP/phrase-level classifier analysis:** Marginal return low compared to
  open research questions; risks tunnel vision on classifier rather than phenomenon
- **Knows/doesn't-know 2x2:** Already addressed by the 3/3 correctness gate at temp=0
