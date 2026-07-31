# Measured: the pipeline on real open-weight models

**Date:** 2026-07-30 · **Config:** `mode: local` — `bge-small-en-v1.5` embeddings,
`bge-reranker-base`, `nli-deberta-v3-base`, generation + critic on
`llama3.2:3b` via Ollama · **Set:** 16-question stratified subset (10 easy /
2 multi-hop / 2 unanswerable / 2 adversarial) of the then-62-question gold set
(since grown to 109 — numbers in this file predate that change unless stated) · **Pipeline:** as shipped
(hybrid + rerank + self-correction + CRAG) · **Wall clock:** 31 min.

Reproduce: `ARAG_MODE=local ARAG_EMBEDDINGS__PROVIDER=sentence_transformers
ARAG_LLM__PROVIDER=ollama ARAG_LLM__OLLAMA_MODEL=llama3.2:3b
ARAG_EVAL__SUBSET=16 make eval`

## The numbers

| Metric | mock (62 Q) | **local, llama3.2:3b (16 Q)** | |
|---|---|---|---|
| hallucination_rate ↓ | 0.000 | **0.313** | ✗ does not reproduce |
| faithfulness ↑ | 1.000 | 0.886 | ✗ |
| answer_correctness (token-F1) ↑ | 0.323 | 0.201 | ✗ (see caveat) |
| over_abstention_rate ↓ | 0.104 | **0.333** | ✗ |
| correct_abstention_rate ↑ | 1.000 | **1.000** | ✓ holds |
| adversarial_robustness ↑ | 1.000 | **1.000** | ✓ holds |
| recall@1 ↑ | 0.833 | **0.958** | ✓ better |
| MRR ↑ | 0.913 | **1.000** | ✓ better |
| latency p50 | 1.4 ms | 73,195 ms | ~50,000× |

## What this means, stated plainly

**The headline "hallucination 30.6% → 0.0%" is a `mock`-mode result and does not
reproduce on a 3B model.** `RESULTS.md` labels its mode, but the README used to
claim the "direction and mechanism hold in every mode." Half of that is now
measured false and has been corrected.

**What does hold:** the CRAG answerability gate declined 2/2 unanswerable
questions and 2/2 adversarial traps — the abstention and injection-robustness
mechanisms are model-agnostic, as claimed. And real embeddings genuinely improve
retrieval: recall@1 0.833 → 0.958, MRR 0.913 → 1.000, which is the first direct
evidence for the "neural embeddings will move the retrieval rows" claim.

## Diagnosis: the critic is the bottleneck, not the generator

The 31% is **judge false-positives, not fabrication.** All five flagged answers
were checked by hand against the corpus and the gold answers — every one is
substantively correct:

| Question | Model answer | Corpus | Verdict |
|---|---|---|---|
| Status code for an unconvertible `int` path param | "422 Unprocessable Entity" | `01_routing.md:14` | correct |
| URL for a named route | "`app.url_for()`, pass the route name" | `01_routing.md:27` | correct |
| Multiple values for one query key | "annotate as a list, `tags: list`" | `02_query_params.md:22` (`list[str]`) | correct, dropped `[str]` |
| Receive a JSON body | "argument typed as a `Schema` subclass" | `03_request_body.md` | correct |
| Missing required field | "422, body lists each invalid field" | `03_request_body.md:25` | correct |

What actually happens: a real model *elaborates* ("…because validation runs
before the handler"). Those added clauses aren't verbatim in the retrieved
context, so a 3B judge marks ≥1 claim unsupported — and one unsupported claim is
all it takes for the record to count as a hallucination.

The same weakness produces the 33% over-abstention from the other direction:
three of four over-abstentions ran to `max_iterations` and then declined — the
loop could not satisfy its own critic about answers that were correct.

The first reading of this was "generation is fine and judging is not." **That was
tested and refuted the same day — see the correction section below.** A 7B judge
leaves the per-answer flag rate unchanged; the cause is the generator's
elaboration meeting a strict claim-level metric.

`answer_correctness` 0.201 is partly an artifact: it is token-F1 against terse
gold answers ("A 422 response."), and real models write paragraphs. It penalizes
verbosity, not wrongness.

## Caveats

- **n=16.** The 95% bootstrap CI on a 0.313 rate at n=16 spans roughly
  ±0.23 — treat it as "clearly nonzero," not as a point estimate.
- **3B is the floor**, deliberately: the cheapest model anyone would run. A 7–8B
  instruct model, or a stronger judge, is the obvious next data point.
- **Thresholds were tuned on mock.** `agent.support_threshold`,
  `nli_entail_threshold` and the CRAG thresholds were fitted against mock
  behavior; they were never calibrated for this judge.

## Correction (2026-07-30, same day): the judge was not the bottleneck

The diagnosis above predicted that a stronger judge would fix both error rates.
**It was tested and it did not.** Two corrections were run on the same 16-question
subset, one variable at a time:

| Config | hallu ↓ | overAbst ↓ | faith ↑ | ansF1 ↑ | abstOK ↑ | advRobust ↑ | p50 latency |
|---|---|---|---|---|---|---|---|
| 3B judge (baseline) | 0.313 | 0.333 | 0.886 | 0.201 | 1.000 | 1.000 | 73 s |
| NLI-only critic | 0.438 | **0.083** | 0.845 | **0.323** | 1.000 | 1.000 | **28 s** |
| **7B judge** (qwen2.5:7b) + NLI | 0.438 | 0.167 | 0.857 | 0.289 | 1.000 | 1.000 | 208 s |

The headline rates moved in *opposite* directions, which is the tell. Normalizing
by the answers actually produced — hallucination is only counted on questions the
system answers — dissolves the difference entirely:

| Config | answered | abstained | flagged | **flag rate per answer** |
|---|---|---|---|---|
| 3B judge | 8/16 | 8 | 5 | **0.625** |
| NLI-only | 11/16 | 5 | 7 | **0.636** |
| 7B judge | 10/16 | 6 | 7 | **0.700** |

**A produced answer gets flagged ~2/3 of the time regardless of judge strength.**
Every difference in the headline `hallucination_rate` was a coverage artifact: a
more permissive gate answers more questions, exposing more answers to the metric.
Judge capability changed nothing.

So the bottleneck is **the generator's output style and the strictness of a
claim-level metric — not the critic.** A 3B model answers correctly and then adds
an explanatory clause that isn't in the retrieved context; one such clause flags
the whole record. Upgrading the grader cannot fix that, and this experiment shows
it doesn't.

Two secondary findings:

- **The 7B judge is strictly dominated.** Same flagged rate, worse
  over-abstention than NLI-only (0.167 vs 0.083), worse answer correctness
  (0.289 vs 0.323), and **7× the latency** (208 s vs 28 s p50; 2.2 h for a
  16-question run). There is no argument for it in this configuration.
- **NLI-only is the best local config**: lowest over-abstention, highest answer
  correctness, 2.6× faster than the baseline, and it keeps abstention (1.000) and
  adversarial robustness (1.000) intact. `agent.critic: nli` for local mode.

## Second correction: the prompt fix failed too, and the flag rate is the metric

The step-1 recommendation above was "constrain generation, not grading" — tighten
the answer prompt against ungrounded elaboration. That was implemented, measured,
and **reverted**. All runs below: 3B generator, NLI-only critic, 16-question
subset.

| Run | prompt | claim split | flag rate ↓ | faith ↑ | ansF1 ↑ | overAbst ↓ |
|---|---|---|---|---|---|---|
| A | original | LLM-extracted | 0.636 | 0.845 | 0.323 | 0.083 |
| **C** | original | **deterministic** | **0.545** | 0.771 | 0.323 | 0.083 |
| B | tightened | deterministic | 0.727 | 0.552 | 0.291 | 0.083 |

The first attempt changed *both* the prompt and the metric's claim splitting in
one run — a self-inflicted confound, and the reason run C exists. With that
control:

- **C vs A** (metric only, prompt held): flag rate **improves** 0.636 → 0.545.
  Deterministic splitting is both more comparable *and* slightly kinder.
- **B vs C** (prompt only, metric held): flag rate **degrades** 0.545 → 0.727,
  faithfulness 0.771 → 0.552, correctness 0.323 → 0.291.

**The tightened prompt is solely responsible for the degradation**, and the
mechanism is visible in the outputs: some answers dropped their `[id]` citations
entirely, and one cited `03_request_body` for a question about path-parameter
declaration (`01_routing`). Piling five more prohibitions on a 3B model consumed
its instruction-following budget and crowded out the *original* requirements.
Over-constraining a small model degrades it rather than disciplining it. The
prompt is reverted, with that finding recorded in `prompts.py` so nobody
re-attempts it.

### The flag rate looks like a property of the metric

Across five configurations — 3B judge, 7B judge, NLI-only, two prompts, two claim
splits — the per-answer flag rate stayed in **0.545–0.727**. Judge strength
didn't move it; prompt strictness moved it the wrong way. A claim-level metric
that flags an entire record for one unsupported unit, scoring "correct and
grounded plus one aside" identically to a fabrication, appears to sit around
0.5–0.7 for any small model.

**This is now the open question, and it is a definition decision rather than a
bug:** should a correct, grounded answer containing one unsupported aside count as
a hallucination? If yes, these local numbers are simply what a 3B model scores,
and the lever is a bigger model. If no, the metric needs a severity notion — for
instance only flagging claims that *contradict* the context rather than merely
going beyond it, or weighting by claim count instead of any-claim-fails. That
changes the definition of the project's headline metric after it has been
published, so it is left as an explicit decision, not made silently.

## Third correction: severity metric + NLI premise granularity

The open definition question — *should a correct, grounded answer with one
unsupported aside count as a hallucination?* — was answered **no**. Implemented,
plus a real bug found on the way.

`eval.hallucination.mode: severity` (new default) flags an answer when it
**contradicts** the sources or its **core is ungrounded**
(`faithfulness < min_support_fraction`), rather than whenever any single claim
isn't entailed. `mode: strict` keeps the old rule, and **both are computed every
run** (`hallucination_rate` and `hallucination_rate_strict`) so a definition
change can't quietly improve a headline number.

This requires separating *contradicted* from *merely-not-stated*, which the NLI
model reports directly and the critic was discarding. Real deberta separates them
cleanly — against "The default color is blue.":

| Claim | entail | contra | verdict |
|---|---|---|---|
| "…is red." | 0.000 | 1.000 | fabrication |
| "…is blue." | 0.998 | 0.000 | supported |
| "Colors are configurable per widget." | 0.002 | 0.001 | aside — excused |

### The bug: NLI premises were the wrong size

The first severity run barely moved (flag rate 0.545 → 0.455) and still flagged
answers verified correct by hand, with contradiction scores of 0.499 and 0.881.
Cause: the critic passed **whole retrieved chunks** as the NLI premise. Cross-
encoders are trained on sentence-pair-scale input; a multi-paragraph chunk is out
of distribution and the model drifts toward "contradiction":

| Premise for a claim quoted near-verbatim from the source | entail | contra |
|---|---|---|
| whole 1200-char chunk | 0.138 | **0.499** |
| the paragraph containing it | **0.993** | 0.001 |

Premises are now sentences and adjacent sentence pairs (`agent.nli_premise`,
`chunk` restores the old behavior). A first attempt at splitting on blank lines
was **inert** — chunk text keeps line wrapping but loses blank lines, so it
produced one unit per chunk. Verified by checking the unit count, not by assuming.

### Result

| | strict rule | **severity + premise fix** |
|---|---|---|
| hallucination_rate | 0.375 | **0.188** |
| per-answer flag rate | 0.545 | **0.273** |
| contradicted_claim_rate | 0.151 (mostly false) | **0.021** |
| over-abstention / abstention / robustness | 0.083 / 1.000 / 1.000 | 0.083 / 1.000 / 1.000 |

Three records that were flagged are now correctly clean; the false contradictions
are gone (0.151 → 0.021). **Mock numbers did not move at all** (0.306 → 0.000
across the ablation, adversarial robustness still 1.000) — checked specifically,
because a metric change that improves the headline benchmark would be suspect.
Mock fabrications are *mostly* ungrounded, so they fail the core-support branch
exactly as they failed the strict rule.

### What still limits it: entailment false-negatives

The three remaining flags are **not** contradiction false-positives any more —
they're entailment false-negatives. deberta rejects a claim that narrows a
premise's condition:

| Premise: "If the incoming body is missing a required field **or a field has the wrong type**, Breeze returns a 422…" | entail |
|---|---|
| "Breeze returns a 422 … lists each invalid field and the reason." | **0.993** |
| "**If a JSON body is missing a required field,** Breeze returns a 422 …" | **0.000** |

That is a model-capability limit, not a text-splitting problem — no amount of
premise surgery fixes it. It also exposes a tension in the comparability fix:
deterministic sentence splitting made the metric model-independent but produces
**compound, conditional** claims, which are exactly what this NLI model handles
worst. LLM claim extraction produced more atomic claims but wasn't reproducible.
Resolving that properly needs either a stronger NLI model or deterministic
*clause-level* decomposition.

## Fourth correction: clause-level claim decomposition

The entailment false-negatives above were compound conditional claims. Fixed by
splitting claims at clause level (`eval.claim_decomposition: clause`, default) —
a leading framing conditional is stripped so the main clause carries the
assertion, and trailing `so`/`because` clauses become their own claims. All
deterministic, no LLM, so the metric stays reproducible.

The probe that motivated it, against a premise that supports the statement:

| Hypothesis | entail |
|---|---|
| "If a JSON body is missing a required field, Breeze returns a 422…" | 0.000 |
| main clause alone: "Breeze returns a 422…" | **0.992** |

The condition is deliberately **not** emitted as its own claim: the source states
a broader condition ("…or a field has the wrong type"), so the model's narrowed
version scores 0.000 alone and would reintroduce the false negative from the
other side. Cost of that choice: a fabricated *condition* now goes unchecked.

### Result — measured in the same run, so no cross-run noise

| | strict rule | **severity + premise + clause fixes** |
|---|---|---|
| flagged records | 6/10 answers (**0.600**) | 1/10 answers (**0.100**) |
| hallucination_rate | 0.375 | **0.063** |
| faithfulness | — | 0.818 (from 0.771) |
| contradicted_claim_rate | — | **0.000** |
| correct abstention / adversarial robustness | 1.000 / 1.000 | 1.000 / 1.000 |

Five records that the strict rule flagged are now correctly clean, each
hand-verified against the corpus.

### The last flagged record is still a false positive

`"How do you declare a path parameter in Breeze?"` scores `faithfulness 0.0`, and
its answer is **verbatim correct** against `01_routing.md:9-10`. Two failure modes
NLI can't handle, both now visible:

1. **Code snippets** — ``For example: `@app.get("/items/{item_id}")`.`` is not a
   natural-language proposition; an NLI model has nothing to reason over.
2. **Anaphora broken by isolation** — "This value is passed to the handler as a
   function argument with the same name" is verbatim from the source, but once
   split out as a standalone claim, "This value" has no referent.

Fixing (1) means skipping code-only claims; (2) needs coreference resolution or
carrying the preceding sentence as context. Neither is attempted here.

### Caveat: local runs are not reproducible run-to-run

Unlike `mock`, the local pipeline varies slightly between identical runs — this
run answered 10/16 where the previous answered 11/16, moving over-abstention
0.083 → 0.167. Ollama at `temperature: 0` is not bit-deterministic, and at n=16
one record is ±6 pp. **Compare configurations within a single run** (as the table
above does, strict vs severity on identical answers) rather than across runs, and
treat cross-run deltas smaller than ~2 records as noise.

## A parser bug that broke citations on every real-model run

Spot-checking the 7B generator's flagged answers found three in a row that were
**correct** — near-verbatim restatements of the corpus, one scored as an outright
contradiction. The raw text explained why:

```
'The string values `1`, `true`, `on`, and `yes` are parsed as True for a
 boolean query parameter, according to [02_query_params::2].'
```

The citation marker is still *in* the answer. Chunk ids contain `::`, and the
parser matched `\[([A-Za-z0-9_\-]+)\]` — which cannot match a colon. So on every
real-model run:

- **no citation was ever extracted** — 0/31 answered records for the 3B run and
  0/29 for the 7B run had a single parsed citation, making `citation_precision`
  meaningless (it scored above zero only because abstentions count as 1.0);
- **the markers stayed in the answer text**, so `[02_query_params::2]` became part
  of a claim handed to the NLI model, which is not a proposition and cannot be
  entailed.

`mock` mode never showed this because `MockLLM` constructs citations directly
instead of parsing them — the bug lived entirely in the real-model path, which
had never been exercised until this week.

The fix anchors matching on the ids actually supplied as context, rather than
guessing a character class. A permissive "anything in brackets" pattern is *not*
the fix: this corpus contains `list[str]`, and eating that would corrupt correct
answers. A stray leading `c` is tolerated because models imitate the prompt's
`[c3]` example and emit `[c01_routing::2]`.

### Measured impact (3B generator, same 40-question subset, parser the only change)

| | pre-fix | **post-fix** |
|---|---|---|
| answers carrying a parsed citation | **0/31** | **13/31** |
| citation_precision | 0.062 | **0.438** |
| faithfulness ↑ | 0.685 | **0.708** |
| hallucination_strict ↓ | 0.525 | **0.475** |
| hallucination (severity) ↓ | 0.250 | 0.250 |
| over-abstention / correct abstention | 0.063 / 0.750 | 0.063 / 0.750 |

The headline effect is on `citation_precision`, which went from measuring nothing
to measuring something. Faithfulness and the strict rate improve modestly — the
markers were polluting claims, but they were not the main driver of the residual
flags. The severity rate is unchanged, which is the honest result: **this fixed a
broken metric, not the model's grounding.**

A second finding falls out of it: **18 of 31 answers still carry no citation at
all**, despite the prompt requiring one per sentence. Citation-format
instruction-following is weak at 3B, and that is now visible rather than hidden
behind a parser that discarded every citation anyway. Worth re-measuring on the
7B model, which is likelier to follow the format.

## Judge calibration against human labels — the measurement that settles it

`make calibrate` had only ever graded the mock judge. Run against the real
judges on the 16-example human-labelled set:

| Judge | accuracy | Cohen's κ | precision | recall | errors |
|---|---|---|---|---|---|
| **NLI (`nli-deberta-v3-base`)** | **0.938** | **0.875** | 1.000 | 0.875 | 1 false negative |
| LLM `llama3.2:3b` | 0.750 | 0.500 | 1.000 | **0.500** | 4 false negatives |
| LLM `qwen2.5:7b` | 0.938 | 0.875 | 1.000 | 0.875 | 1 false negative |

**This explains two days of downstream symptoms directly.** The 3B judge's errors
are all *false negatives* — it rejects claims that humans mark supported ("the
default color of a widget is blue", "validation happens before the handler
runs"). A judge with recall 0.5 rejects half the supported claims, so the
self-correction loop cannot satisfy itself and abstains on answers that were
correct. That is exactly the over-abstention and inflated flag rate chased
through the sections above, now measured at the source instead of inferred from
downstream rates.

It also **corrects an earlier conclusion here**. The 7B-judge experiment reported
"judge capability changed nothing," but that was measured under the strict metric
with whole-chunk premises — a metric broken badly enough to mask the difference.
Judge capability differs enormously (κ 0.50 vs 0.875); the metric was hiding it.

The practical recommendation is unchanged and now has a direct reason:
**`agent.critic: nli`**. The NLI model matches the 7B judge exactly (κ 0.875)
while being roughly 7× cheaper, so the LLM judge earns nothing on this pipeline.

Note the mock judge fails in the *opposite* direction — 4 false positives, κ 0.50
— which is why mock over-reports support and real 3B under-reports it.

## 3B vs 7B generator

The last untested pipeline variable — every earlier experiment swapped the
*judge*. 40-question stratified subset of the expanded gold set, `critic: nli`:

| | llama3.2:3b | qwen2.5:7b |
|---|---|---|
| correct_abstention ↑ | 0.750 | **1.000** |
| answer_correctness ↑ | 0.299 | **0.366** |
| hallucination ↓ | **0.250** | 0.375 |
| faithfulness ↑ | **0.685** | 0.533 |
| per-answer flag rate ↓ | **0.323** | 0.517 |
| adversarial robustness | 1.000 | 1.000 |
| p50 latency | **28 s** | 72 s |
| wall clock | **25 min** | 4.6 h |

The 7B model is clearly better at the things a user cares about: it never
answered an out-of-scope question (abstention 0.750 → 1.000) and its answers are
22% closer to gold. It scores *worse* on faithfulness and hallucination — and the
spot-check above shows those flags are largely the citation-parser bug plus the
elaboration penalty, not fabrication. A stronger model writes longer answers,
which means more claims, which means more chances for an imperfect NLI check to
reject one.

**These numbers predate the citation fix and should be re-run.** They are kept
because they are what motivated the spot-check that found the bug.

## Threshold sweep on real models

`make sweep NAME=abstention` run in local mode (`llama3.2:3b`, `critic: nli`,
16-question subset, 9 grid points ≈ 100 min). These thresholds had only ever been
fitted against mock behavior — the last mock-tuned assumption in the pipeline.

| `support_threshold` | overAbst ↓ | ansF1 ↑ | faith ↑ | hallu ↓ | abstOK |
|---|---|---|---|---|---|
| **0.3** | **0.083** | **0.302** | 0.807 | 0.062 | 1.000 |
| 0.5 (shipped) | 0.167 | 0.295 | 0.818 | 0.062 | 1.000 |
| 0.7 | 0.250 | 0.269 | 0.849 | 0.062 | 1.000 |

Two findings, of very different strength.

**`nli_entail_threshold` is inert.** All three values (0.3 / 0.5 / 0.7) produce
byte-identical results at every support level. deberta's entailment scores are
saturated near 0 and 1 — the probes throughout this document read 0.000, 0.992,
0.998 — so nothing lands in the band the threshold moves through. The knob has no
purchase on this NLI model, and tuning effort should go elsewhere. This is the
sturdier of the two results: it is nine identical measurements, not a difference.

**`support_threshold` trades over-abstention against faithfulness, monotonically.**
Lowering it answers more questions: over-abstention halves, answer correctness
rises, hallucination stays flat at 0.062 and correct-abstention stays 1.000.
Faithfulness drifts down slightly because it now averages over more answered
questions.

*But 0.167 → 0.083 is exactly one record at n=12 answerable*, inside the
run-to-run noise documented above. The monotone trend across three levels
(2 records end to end) is better evidence than any single pair, but it is not
enough to move a shipped default. A full 62-question confirmation of
`support_threshold` 0.5 vs 0.3 is the deciding experiment.

### Best local configuration measured so far

`agent.critic: nli`, original prompt, severity metric with sentence-window
premises and clause-level claims. Every value below is read from the **same
artifact** (`eval/results/local_nli_severity.json`, the 23:51 run) against the
first local baseline (`local_llama32.json`) — an earlier version of this table
mixed rows from two different runs and overstated the result.

| | first baseline | **final config** |
|---|---|---|
| hallucination_rate | 0.313 | **0.063** |
| hallucination_rate_strict | 0.313 | 0.375 |
| over_abstention_rate | 0.333 | **0.167** |
| answer_correctness | 0.201 | **0.295** |
| faithfulness | 0.886 | 0.818 |
| correct_abstention / adv. robustness | 1.000 / 1.000 | **1.000 / 1.000** |
| recall@1 / MRR | 0.958 / 1.000 | 0.875 / 0.958 |
| p50 latency | 73 s | **34 s** |

Honest reading: hallucination down 80% relative, over-abstention halved, answer
correctness up 47%, latency down 2.2×, safety gates intact. Faithfulness and
recall@1 are *lower* — faithfulness because it now averages over more answered
questions, recall@1 because it is one record at n=12 and local runs are not
reproducible (see the caveat above). The intermediate run reached
over-abstention 0.083 and answer_correctness 0.323; that difference is one
record and should not be read as a regression.

### Measurement caveat introduced by this change

Routing eval faithfulness through `comp.judge` meant **claim extraction followed
`judge_model`**, so segmentation differed between runs with different judges and
`hallucination_rate`/`faithfulness` were comparable only within a fixed judge.

**Fixed.** When `eval.faithfulness_method: nli`, the metric now splits claims with
a deterministic sentence splitter and never calls an LLM — `critique_answer` takes
an explicit `claims` argument. The metric is now independent of every model in the
pipeline, so runs are comparable across judges and generators. Mock numbers were
unaffected (the mock already split deterministically), so no baseline moved.

## Next steps, in order

Revised after the experiment above. ~~Split the judge from the generator~~ was
step 1; it was implemented, measured, and did not work. The remaining steps
target generation and measurement, where the evidence now points.

Revised twice now. Two hypotheses were tested and both failed: ~~a stronger
judge~~ (no effect on the flag rate) and ~~a stricter answer prompt~~ (actively
worse). ~~Pin claim extraction~~ is done. What remains:

1. ~~Decide what the metric should count~~ and ~~clause-level decomposition~~ —
   both **done**. Remaining metric work is narrower: skip code-only claims (an
   NLI model has nothing to reason over in a code snippet), and handle anaphora
   broken by claim isolation ("This value…" loses its referent). Those are the
   two failure modes behind the single remaining false positive.
2. **Try a 7–8B instruct generator** (not judge — generator). Judge strength was
   ruled out; generation quality has not been tested, and it is the remaining
   pipeline-side variable. `qwen2.5:7b` is already pulled.
3. ~~Run `make calibrate` in local mode~~ — **done**, and it was the highest-value
   run of the whole exercise: it explained the over-abstention at source (3B judge
   recall 0.500) and corrected an earlier conclusion drawn from downstream rates.
   Direct measurement beat two rounds of inference.
4. ~~Re-sweep thresholds under local mode~~ — **done**, see the sweep section.
   `nli_entail_threshold` turned out inert on this NLI model;
   `support_threshold` shows a monotone trade-off whose winning point is within
   noise at n=16 and is being confirmed on the full gold set.
5. **Use `agent.critic: nli` for local mode** — the measured best config. Do *not*
   use a 7B judge here (strictly dominated at 7× the latency), and do not tighten
   the answer prompt (measured worse).

## Practical cost of local evaluation

~1.5–2 min per question through the full agentic loop on a 3B model:

| Scope | Time |
|---|---|
| 16-question stratified subset | ~31 min |
| Full 62-question gold set | ~2 h |
| 8-config ablation | ~16 h |

Which is why `mock` stays the default for CI and iteration, and why
`eval.subset` had to become slice-stratified — a fast subset that was 100% easy
questions would have shown `correct_abstention_rate` and
`adversarial_robustness` with no unanswerable or adversarial question behind
them.
