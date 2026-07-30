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

### Measurement caveat introduced by this change

Routing eval faithfulness through `comp.judge` means **claim extraction now
follows `judge_model`**, so segmentation differs between runs with different
judges — `hallucination_rate` and `faithfulness` are strictly comparable only
within a fixed judge. The per-answer table above is the robust comparison: it
counts records, and the entailment scoring itself is the fixed deberta-NLI model
(`eval.faithfulness_method: nli`) in all three runs. Pinning claim extraction to
a deterministic sentence splitter when the metric is NLI-based would remove the
wrinkle; it would also shift the published mock numbers, so it is listed as a
next step rather than done silently here.

## Next steps, in order

Revised after the experiment above. ~~Split the judge from the generator~~ was
step 1; it was implemented, measured, and did not work. The remaining steps
target generation and measurement, where the evidence now points.

1. **Constrain generation, not grading.** The flagged answers are correct plus an
   unsupported explanatory clause. Tighten the answer prompt against
   elaboration beyond the retrieved context, and/or lower `llm.max_tokens` for
   local models. This is the only lever aimed at the actual cause, and it is
   directly measurable by the per-answer flag rate (currently ~0.63).
2. **Decide what the metric should count.** "Correct, grounded, plus one
   unsupported aside" is currently scored identically to a fabrication. If that
   is not the intent, claim-level support needs a severity notion — or the answer
   prompt needs to forbid asides (step 1). Either way this is a definition
   decision, not a bug.
3. **Pin claim extraction for the metric** to a deterministic splitter when
   `faithfulness_method: nli`, removing the LLM from the metric path entirely and
   making runs comparable across judges. Note this shifts the published mock
   numbers, so it needs a baseline regeneration.
4. **Run `make calibrate` in local mode.** Still worth doing: it has only ever
   graded the mock judge, and it would quantify the judge's agreement with humans
   rather than inferring it from downstream rates.
5. **Re-sweep thresholds under local mode** (`make sweep NAME=abstention`); the
   defaults were fitted on mock behavior. Budget: ~12 min per NLI-critic run.
6. **Use `agent.critic: nli` for local mode** — the measured best config (see the
   correction section). Do *not* use a 7B judge here: strictly dominated at 7× the
   latency.

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
