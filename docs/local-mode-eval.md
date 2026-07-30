# Measured: the pipeline on real open-weight models

**Date:** 2026-07-30 · **Config:** `mode: local` — `bge-small-en-v1.5` embeddings,
`bge-reranker-base`, `nli-deberta-v3-base`, generation + critic on
`llama3.2:3b` via Ollama · **Set:** 16-question stratified subset (10 easy /
2 multi-hop / 2 unanswerable / 2 adversarial) · **Pipeline:** as shipped
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

### Best local configuration measured so far

Run C: original prompt, `agent.critic: nli`, deterministic metric claims.
Versus the first local baseline (3B judge, LLM claims):

| | first baseline | **run C** |
|---|---|---|
| hallucination_rate | 0.313 | 0.375 |
| over_abstention_rate | 0.333 | **0.083** |
| answer_correctness | 0.201 | **0.323** |
| correct_abstention / adv. robustness | 1.000 / 1.000 | **1.000 / 1.000** |
| p50 latency | 73 s | **28 s** |

Four fewer correct answers thrown away, 60% higher answer correctness, 2.6×
faster, with the safety gates intact. The nominal hallucination rate is slightly
higher purely because it answers more questions — the per-answer flag rate went
*down* (0.625 → 0.545).

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

1. **Decide what the metric should count** — now the blocker, not a nice-to-have.
   "Correct, grounded, plus one unsupported aside" scores identically to a
   fabrication, and five configurations could not move the flag rate out of
   0.545–0.727. Either accept that as a small model's honest score, or give
   claim-level support a severity notion (flag only claims that *contradict*
   context; or weight by claim count rather than any-claim-fails). This redefines
   the headline metric, so it needs an explicit decision.
2. **Try a 7–8B instruct generator** (not judge — generator). Judge strength was
   ruled out; generation quality has not been tested, and it is the remaining
   pipeline-side variable. `qwen2.5:7b` is already pulled.
3. **Run `make calibrate` in local mode.** It has only ever graded the mock judge.
   It would measure judge/human agreement directly instead of inferring it from
   downstream rates — and given two failed inferences, direct measurement is
   overdue.
4. **Re-sweep thresholds under local mode** (`make sweep NAME=abstention`); the
   defaults were fitted on mock behavior. Budget ~10 min per NLI-critic run.
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
