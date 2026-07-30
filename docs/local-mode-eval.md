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

So at 3B, **generation is fine and judging is not**. Both error rates are one
root cause.

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

## Next steps, in order

1. **Split the judge from the generator.** `llm.judge_model` already exists in
   config and is unused. Point the critic at a stronger model (7–8B, or an API
   model) while generating locally, and re-measure. This directly targets the
   root cause.
2. **Run `make calibrate` in local mode.** The judge-calibration harness exists
   and has only ever graded the mock judge. It reports accuracy and Cohen's κ
   against human labels — run it on the real judge and trust it with evidence.
3. **Re-sweep thresholds under local mode** (`make sweep NAME=abstention`), which
   the mock-tuned defaults were never validated for. Budget hours: one full
   local eval is ~30 min for 16 questions.
4. **Lean on the NLI cross-check** where the LLM judge is weak — `agent.critic:
   nli` is already selectable, and deberta-NLI is a purpose-built entailment
   model rather than a 3B generalist.

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
