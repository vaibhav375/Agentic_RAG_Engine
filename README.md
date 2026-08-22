# Self-Correcting Agentic RAG Engine + Evaluation Harness

<!-- The badge job on `main` regenerates both badge forms after every merge.
     The SVG is linked relatively so GitHub serves it directly — it renders
     whether the repo is public or private. If you'd rather use shields.io
     (public repos only, since shields fetches raw.githubusercontent
     anonymously), swap in:
     ![hallucination](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/vaibhav375/Agentic_RAG_Engine/main/eval/results/badge.json) -->
![hallucination](eval/results/badge.svg)

A production-style Retrieval-Augmented Generation engine that **improves its own
answers**: it retrieves with a hybrid dense+sparse pipeline, reranks with a
cross-encoder, generates a grounded answer with citations, then **critiques its
own answer and re-queries — or abstains — when the answer isn't supported** by
the retrieved context. The whole thing is wrapped in an **automated evaluation
harness** that measures faithfulness, answer relevance, context precision/recall,
and an explicit **hallucination rate**, and proves a **measured reduction in
hallucination** vs. a naive RAG baseline through an ablation study.

> **Headline (reproducible, `mock` mode):** hallucination rate **31.2% → 0.9%**
> (95% CI [0.0%, 2.8%]), correct abstention on unanswerable questions
> **0% → 91.7%**, and adversarial / prompt-injection robustness **0% → 100%**
> across the ablation, on a **109-question** technical-docs benchmark
> (easy · multi-hop · unanswerable · adversarial). Run `make ablation` to
> regenerate [`RESULTS.md`](RESULTS.md).
>
> These numbers were **0.0% and 100%** on the original 62-question set. Growing
> the set to 109 exposed two genuine residual failures — the perfect scores were
> partly an artifact of a small benchmark that the thresholds had been tuned
> against. The larger set is the honest one, and its confidence interval is
> narrow enough to mean something.
>
> **And on real open-weight models** (`qwen2.5:3b` + `bge-small`, **117**
> questions, no API keys): hallucination **0.85%** (95% CI [0%, 2.6%]) — **1 of
> 117 records flagged** — **correct abstention 100%** (12/12 out-of-scope
> questions declined), **adversarial robustness 100%** (18/18 on a slice
> deliberately hardened with subtle false premises, scored refutation-aware),
> faithfulness **0.928**, recall@1 **0.960**, MRR **0.994**, and
> `contradicted_claim_rate` **0.004**.
>
> Answer quality is reported two ways, because token-F1 measurably inverts on
> this data: **token-F1 0.365** against **semantic equivalence 0.773** (LLM judge,
> 75 answered questions). The judge is calibrated against hand labels at
> Cohen's κ 0.80, precision 1.000 — it never accepts an answer a human called
> wrong. Details in [`docs/local-mode-eval.md`](docs/local-mode-eval.md).
>
> Latency is deliberately **not** quoted here. Every timing measured on the 8 GB
> development machine is suspect — a full run drove it 12 GB into swap, and three
> separate "findings" turned out to be paging rather than pipeline behaviour.
> Quality metrics survive that contamination; timings do not.
>
> **Out-of-sample check.** Every threshold and prompt was tuned against this gold
> set, so `eval.split` holds out a stratified quarter that nothing was tuned on.
> The safety guarantees survive it intact — **abstention 1.000 and adversarial
> robustness 1.000 on the holdout** — while the generation metrics come in
> consistently lower (faithfulness 0.947 → 0.899, citation precision
> 0.826 → 0.738, recall@1 0.962 → 0.905). Quote the headline with that column
> beside it.
>
> Full analysis in [`docs/local-mode-eval.md`](docs/local-mode-eval.md). Every
> question the pipeline is measured on — with the answer it should give, or the
> reason it should refuse — is in
> [`docs/question-bank.md`](docs/question-bank.md).

## Why this is more than "a chatbot"

The value is in three things, in order: **(1) the measured eval/ablation
results, (2) the self-correction loop, (3) the production serving concerns**
(hybrid retrieval, reranking, semantic caching, latency/cost, Docker).

Beyond the base spec it implements the named techniques a modern RAG/agent
engineer is expected to know:

- **CRAG answerability gate** (Corrective RAG) — grades whether retrieved context
  can actually answer the query (IDF-weighted relevance) and **declines when it
  can't**. This is the main driver of the hallucination drop and of correct
  abstention on out-of-scope questions (91.7% — the one miss is analysed in
  `RESULTS.md`).
- **Self-correction loop with an NLI-cross-checked LLM judge** — the critic
  combines LLM-as-judge with a natural-language-inference entailment model,
  mitigating LLM-judge self-preference bias.
- **HyDE**, **multi-hop query decomposition**, and **MMR** — retrieval modes:
  embed a hypothetical answer; split multi-hop questions into sub-questions
  fused with RRF; and Maximal Marginal Relevance to diversify the final context.
- **Selective-prediction risk–coverage analysis** — evaluates the abstention gate
  with a risk–coverage curve + AUC; the CRAG signal reaches **max safe coverage =
  the answerable fraction** (answer every in-scope question at zero out-of-scope risk).
- **Prompt-injection input guardrail** — an explicit defense-in-depth layer that
  flags injection/jailbreak/exfiltration/false-premise inputs.
- **Query router / complexity classifier** — cheap queries skip the expensive
  loop; only complex ones pay for it (cost-aware design).
- **Contextual chunk enrichment** — Anthropic-style "contextual retrieval": each
  chunk gets a situating prefix before embedding.
- **Adversarial / prompt-injection robustness** — an 18-question slice of
  injection traps *and subtle false premises* (a plausible wrong default, a
  fabricated API name, an inverted guarantee). Scoring is **refutation-aware**:
  a grounded answer that never addresses the planted falsehood does not pass,
  because the reader still leaves believing it.
- **Judge calibration + experiment registry** — the critic is validated against
  human labels (accuracy, Cohen's κ), and every eval run is logged with git SHA +
  config hash for traceable experiment tracking.

## Runs offline out of the box

Every backend (embeddings, LLM, reranker, NLI) has a deterministic `mock`
implementation, so the full pipeline — including the eval harness — runs with
**no API keys and no network**. That's what makes CI reproduce exact numbers.
Flip `config.mode` to `local` (sentence-transformers + Ollama) or `api`
(OpenAI/Anthropic) to use real models; nothing else changes.

```bash
make install     # core deps only; mock mode works immediately
make test        # 182 unit + integration tests, no network
make demo        # ingest the corpus + answer one query
make ablation    # full ablation study -> RESULTS.md
```

## Architecture

```
INGESTION (offline)
  raw docs ─▶ loader ─▶ chunker (+contextual enrichment) ─▶ embedder ─▶ dense index
                                    └───────────────────────────────▶ BM25 (sparse)

QUERY (online)
  query ─▶ [semantic cache hit? ─▶ return]
            │ miss
            ▼   ┌── router: simple ─▶ direct path ──┐
        Hybrid Retrieve (dense + BM25) ─▶ RRF fuse ─▶ Cross-encoder Rerank ─▶ top-n
            │                                                                   │
            ▼                                                                   ▼
        Generate grounded answer + citations ─────────────────────────▶ Critic
                                                                          │
        ┌──────────── SELF-CORRECTION LOOP (≤ R iterations) ─────────────┘
        │  Critic (LLM-as-judge  ∧  NLI entailment): every claim supported?
        │   supported   ─▶ accept, return answer + citations
        │   unsupported ─▶ reformulate query / widen retrieval ─▶ retrieve again
        │   still unsupported after R ─▶ ABSTAIN (never fabricate)
        └────────────────────────────────────────────────────────────────
            ▼
        write to semantic cache ─▶ response (answer, citations, per-stage trace, cost)

EVAL (offline, repeatable)
  gold Q&A ─▶ run each config ─▶ faithfulness, answer-F1, context P/R,
              hallucination rate, abstention, latency, cost ─▶ RESULTS.md
```

## Repository layout

```
src/arag/
  common/     schemas, typed config loader (+env overrides), telemetry/cost
  providers/  embeddings · llm · rerank · nli  (each: mock + real backends)
  ingest/     load · chunk (strategies + contextual enrichment) · index (dense+BM25)
  retrieve/   hybrid (dense/sparse + RRF fusion) · rerank (cross-encoder)
  generate/   grounded answer + citations · prompts
  agent/      critic (LLM+NLI) · retrieval_grader (CRAG) · reformulate · router · graph
  cache/      semantic_cache (memory + redis, false-hit guarded)
  serve/      FastAPI api (/query, /health, /stats)
  engine.py   builds components + the single answer_query() entrypoint
  cli.py      arag ingest | query | serve | eval | calibrate-judge | history
eval/
  build_gold_set · metrics · run_eval · ablation · report · ci_gate
  pr_report (sticky PR comment: verdict, deltas, slices, trend)
  calibrate_judge · registry (experiment tracking) · selective · dashboard
data/
  corpus/*.md            authored, verifiable technical-docs corpus
  eval/gold_qa.jsonl     117 Qs: easy / multi-hop / unanswerable / adversarial
  eval/judge_calibration.jsonl   human labels for judge calibration
tests/  unit + integration (fixture corpus, mock mode)
```

## Quickstart

### 1. Mock mode (default — nothing to configure)

```bash
make install
make test
make demo
python -m arag.cli query "How do I return an HTTP error from a handler?" --trace
```

### 2. Reproduce the results

```bash
make ablation          # writes RESULTS.md + eval/results/*.json + a plot
```

### 3. Serve the API

```bash
make serve             # http://localhost:8000  (mock mode)
curl -s localhost:8000/health
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"query":"What is the default GZip minimum size?"}' | jq
```

### 3a. Playground — zero dependencies

`make serve` alone gives you a working UI at <http://localhost:8000>: ask a
question and watch the pipeline run — retrieve → CRAG grade → generate →
critique → accept or abstain. **Stages stream in as they complete** (SSE), so on
a local model you see `retrieve` land in half a second instead of watching a
blank panel for fourteen. Then the full answer arrives with the per-stage trace,
citations, retrieved context, and CRAG / guardrail / route badges. No Node, no
npm, no build step; it is one HTML file served by the API.

Try the built-in examples: an answerable question, an out-of-scope one (watch it
abstain), a prompt injection (watch the guardrail fire and the trace stop before
generation), and a multi-hop question. The feature checkboxes re-run the same
query with components on or off, which is a live ablation.

### 3b. Interactive web app (Next.js frontend)

An interactive **Playground** UI lets you ask questions and watch the pipeline
retrieve → grade (CRAG) → generate → self-correct → or abstain, with **live
feature toggles** (flip components and re-ask to see a live ablation), a
**streamed pipeline trace**, clickable citations, CRAG / guardrail / route
badges, and cost/latency.

```bash
make serve                      # backend on :8000 (self-ingests on first run)
cd frontend
cp .env.local.example .env.local
npm install && npm run dev      # UI on http://localhost:3000
```

Or the whole stack in one command (Redis + API + UI):

```bash
docker compose up               # UI on :3000, API on :8000
```

The UI drives these endpoints, added for the app: `POST /query` (+ per-request
feature flags), `POST /query/stream` (SSE), `GET /config`, `/corpus`,
`/chunk/{id}`, and `/eval/{ablation|selective|history|calibration}`.

### 4. Real models — open weights, no API key

`local` mode runs the whole pipeline on open-weight models with **no account, no
key, and no per-token cost**: `bge-small-en-v1.5` embeddings and
`bge-reranker-base` (sentence-transformers), `nli-deberta-v3-base` for the NLI
cross-check, and Ollama for generation.

```bash
make install-local
brew install ollama && ollama serve       # or the Ollama.app
ollama pull llama3.2:3b                   # ~2 GB
# config/config.yaml: mode: local, embeddings.provider: sentence_transformers,
#                     llm.provider: ollama, llm.ollama_model: llama3.2:3b
make ingest && make eval
```

**Pick an instruct model, not a reasoning model.** Measured on this pipeline
(M-series laptop, one query, full agentic path):

| Model | Wall clock | Output |
|---|---|---|
| `llama3.2:3b` (instruct) | **48 s** | clean grounded answer + citations |
| `qwen3:4b` (reasoning) | 229 s | narrates its reasoning into the answer text |

Reasoning models emit chain-of-thought that lands in the answer, wrecking
citation parsing and token-F1. `llm.think: false` asks Ollama to suppress it
(needs Ollama ≥ 0.9; set `null` for older servers) — but small reasoning models
often narrate anyway, so prefer an instruct model here. Qwen's own instruct
builds (`qwen2.5:3b`, `qwen2.5:7b`) work well.

Local generation is ~50× slower than mock, so use a **stratified subset** while
iterating — `eval.subset: 16` keeps every difficulty slice represented:

```bash
ARAG_EVAL__SUBSET=16 make eval
```

**Open-weight models too big for a laptop.** Kimi K2 (~1T params MoE), Qwen3-235B
and friends have open weights but need a host. `provider: openai` speaks the
OpenAI protocol to *any* compatible endpoint, so they need no new backend — just
a `base_url`, a model id, and whichever env var holds that host's key:

```yaml
llm:
  provider: openai
  base_url: https://api.moonshot.ai/v1     # Moonshot (Kimi)
  api_key_env: MOONSHOT_API_KEY
  model: kimi-k2-0711-preview
```

Same shape for OpenRouter, Together, Groq, or a self-hosted vLLM (which needs no
key at all — `base_url: http://localhost:8000/v1`). Note the distinction: the
*weights* are open, the *hosting* still costs money.

For OpenAI/Anthropic proper:

```bash
make install-api
cp .env.example .env             # add OPENAI_API_KEY / ANTHROPIC_API_KEY
# config/config.yaml: mode: api, llm.provider: openai, embeddings.provider: openai
```

Everything is config-driven — models, `k` values, thresholds, iteration cap,
chunk size — via `config/config.yaml`, overridable per-field with
`ARAG_SECTION__KEY` env vars (e.g. `ARAG_RETRIEVAL__USE_HYBRID=true`).

### 5. Full Docker stack (Redis + app)

```bash
docker compose up          # brings up redis and the API
```

## Evaluation methodology

`RESULTS.md` reports every metric **baseline → enhanced**, as an **ablation
table** (rows = pipeline configs, columns = metrics) so each component's
contribution is visible:

| Metric | What it measures |
|---|---|
| **Hallucination rate** (headline) | % of answers with ≥1 unsupported claim (unanswerable Qs: answering at all counts) |
| Faithfulness / groundedness | fraction of answer claims entailed by retrieved context |
| Answer correctness (token-F1) | overlap with the gold answer — **rough signal only**, see below |
| Context precision / recall | retrieval quality vs. gold supporting docs |
| Correct abstention | on unanswerable questions, does it decline instead of fabricating |
| Latency p50/p95 & cost/query | with/without cache and router |

**Retrieval vs. generation are scored separately** — `recall@k`, `precision@k`,
and `MRR` measure the retrieved pool; faithfulness/relevance/correctness measure
the answer given that pool. A regression then points at the half to fix instead
of a blended score that hides it.

**Retrieval is also scored at strict k.** `recall@k` with a generous k saturates
at 1.000 on a corpus this size — every ranking change looks like a no-op, which
is a benchmark that can't grade the thing it's measuring. `recall@1` / `recall@3`
(`eval.strict_ks`) sit next to it and do show the work: the hybrid row moves
recall@1 0.854 → 0.917 where `recall@k` showed a flat 1.000. They're also what
makes reranking's *cost* visible (see `RESULTS.md`) instead of invisible.

The gold set is hand-built with **easy, multi-hop, and unanswerable** questions;
`make ablation` / `python -m eval.build_gold_set` validate that every supporting
quote exists in the ingested corpus. Context precision/recall are **document-
level** so they stay meaningful even though chunking is itself an ablation
variable. Every metric is also reported **per slice** (a gain on easy questions
can't mask a drop on multi-hop ones) with a **95% bootstrap CI** on the
hallucination rate.

**Don't grade the loop with its own judge.** Eval faithfulness is measured with
an independent NLI model (`eval.faithfulness_method`), and the critic itself is
**calibrated against a human-labeled set** — `make calibrate` reports the judge's
accuracy and Cohen's κ vs. humans, so the grader is trusted with evidence, not
assumption.

Quality is enforced in CI: `eval/ci_gate.py` runs the full pipeline, **fails the
build if the hallucination rate regresses** past budget, and **diffs every
metric against a committed baseline** (`eval/results/ci_baseline.json`) — the
harness doubles as a per-metric regression gate for prompt/model changes.

### The eval report bot (MLOps quality gate)

A regression diff buried in CI logs is a diff nobody reads, so the gate posts its
verdict where engineers actually look. On every PR the workflow renders one
**sticky comment** (edited in place on each push, never duplicated) with the
pass/fail verdict, per-metric deltas vs. the committed baseline, the per-slice
breakdown, and a **sparkline trend** of the headline metrics over recent runs —
and `main` self-updates the hallucination badge above.

```
## 🟢 Eval Report — regression gate PASSED
hallucination 0.0% · faithfulness 1.000 · correct-abstention 1.000

| Metric                | Baseline | This PR |      Δ |    |
| hallucination_rate ↓  |    0.000 |   0.000 | +0.000 | ✅ |
| answer_correctness ↑  |    0.900 |   0.323 | -0.577 | ❌ |   ← flips the header to 🔴 and fails the check

hallucination_rate ↓  ██▂▁▁▁▁▁   0.306 → 0.000
faithfulness       ↑  ▁▁██████   0.742 → 1.000
```

`eval/ci_gate.py` and the comment renderer (`eval/pr_report.py`) share one
verdict function, so **what the comment shows is exactly what CI enforces**. The
trend survives ephemeral runners by carrying `history.jsonl` as a workflow
artifact. Budgets, tolerance, and baseline path live in `eval.ci_gate` in
`config/config.yaml`.

```bash
make report            # render the PR comment locally -> comment.md
make update-baseline   # re-baseline, then commit eval/results/ci_baseline.json in the same PR
```

When a change legitimately moves a metric, `make update-baseline` and committing
the new baseline in the same PR makes the shift explicit in review rather than
silent. Fork PRs get a read-only token, so the comment step is best-effort there
— the gate itself still runs and still fails the build.

```bash
make eval          # full pipeline over the gold set (per-slice + CIs)
make ablation      # component-by-component ablation -> RESULTS.md
make calibrate     # judge vs. human agreement (accuracy, Cohen's kappa)
make gate          # CI regression gate: budget + baseline diff
make report        # render the PR eval-report comment locally (comment.md)
make history       # experiment registry: recent runs (git sha + config hash)
make selective     # risk–coverage analysis of the abstention gate (AUC)
make sweep         # grid-sweep thresholds (NAME=abstention|crag|cache)
ARAG_EVAL__SPLIT=holdout make eval   # out-of-sample: the quarter nothing was tuned on
make dashboard     # self-contained HTML eval dashboard (open in a browser)
```

## Resume bullets (values filled from `RESULTS.md`)

- Built a self-correcting agentic RAG engine (Corrective-RAG answerability gate +
  NLI-cross-checked LLM-as-judge critic) that re-issues its own retrievals or
  abstains, cutting hallucination from **31.2%** to **0.9%**
  (95% CI [0.0%, 2.8%]) and reaching **91.7%** correct abstention
  and **100%** prompt-injection robustness on a 109-question benchmark.
- Implemented hybrid dense + sparse (BM25 + RRF) retrieval with cross-encoder
  reranking, HyDE, and multi-hop query decomposition, served behind FastAPI with
  a semantic cache and a full Dockerized stack.
- Developed an automated evaluation harness — retrieval (recall@k / MRR) vs.
  generation (faithfulness / hallucination) metrics, per-slice breakdowns,
  bootstrap CIs, judge calibration (Cohen's κ), and an experiment registry —
  wired into CI as a per-metric regression gate.

## Notes / limitations

- **The headline hallucination numbers are `mock`-mode results and do not
  reproduce on a small local model.** Measured on `llama3.2:3b` (16-question
  stratified subset): hallucination 0.313, over-abstention 0.333 — see
  [`docs/local-mode-eval.md`](docs/local-mode-eval.md). What *does* hold is
  abstention (2/2 unanswerable declined) and injection robustness (2/2), and
  retrieval genuinely improves with real embeddings (recall@1 0.833 → 0.958,
  MRR 0.913 → 1.000). Diagnosis: every flagged answer was hand-checked against
  the corpus and is factually correct — the model answers correctly and then adds
  an explanatory clause that isn't in the retrieved context, and one such clause
  flags the whole record. Two fixes were tried and **both failed**: a 7B judge
  left the per-answer flag rate unchanged (0.625 → 0.700) at 7× the latency, and a
  stricter answer prompt made it worse (0.545 → 0.727) by crowding out the
  original instructions on a 3B model. Across five configurations the flag rate
  never left 0.545–0.727, which points at the metric — flagging a whole record for
  one unsupported aside — rather than at the pipeline. Whether that *should* count
  as a hallucination is an open definition decision, documented not silently
  changed. `agent.critic: nli` is the measured best local config.
  This bullet previously read "the direction and mechanism hold in every mode";
  measurement disproved half of it.
- **`answer_correctness` is token-F1, and it inverted on a hand-check.** Seven
  answers the pipeline had discarded were recovered and graded against gold by
  hand: six correct, one wrong. Token-F1 ranked them close to backwards. The one
  *wrong* answer scored highest (0.632) because it matches gold's wording and
  swaps only the entity that decides it (`CORSMiddleware` for `GZipMiddleware`),
  while a fully *correct* answer scored 0.000 — "Routes are not authenticated by
  default" against gold "No; every route is public until you add a security
  dependency", same meaning, no shared content tokens. So small differences in
  this metric mean little, here or in `docs/local-mode-eval.md`. Two cheaper
  repairs were measured and both failed: NLI bidirectional entailment scores
  ~0.00 even for correct answers (it won't entail terse gold fragments like
  "True."), and embedding cosine puts the wrong answer at 0.826, above four of
  the six correct ones. Nothing comparing surface similarity catches a
  one-entity swap; a semantic verdict needs the LLM judge, for which the judge
  role and Cohen's κ calibration already exist.
- Mock embeddings are lexical, so hybrid/rerank precision gains are muted vs.
  neural embeddings — see the note in `RESULTS.md`.
- **Reranking measurably costs ranking quality in `mock` mode** (recall@1
  0.917 → 0.833): a lexical stand-in reranker carries less signal than the
  dense+sparse fusion it overwrites. `retrieval.rerank_fusion: rrf` fuses the two
  rankings and recovers it fully, but shifts the generator's context enough to
  degrade the CRAG gate (hallucination 0.000 → 0.032), so `replace` stays the
  default. Reported rather than hidden — it's an artifact of the mock reranker
  and the first thing to re-measure on real models.
- No fine-tuning, auth, or multi-tenancy — out of scope by design.

## Roadmap & how to get the best results

Concrete next steps, ordered by return-on-effort, with the knob that drives each:

1. **Run real models for the headline numbers.** `mode: api` with
   `embeddings.provider: openai` (or `local` with bge/e5). Neural embeddings make
   the hybrid/rerank rows move much more than the lexical mock shows, and a real
   NLI/LLM judge fixes the negation/quantity errors the calibration surfaces.
2. **Tune retrieval before touching prompts.** `k_dense`/`k_sparse`, `rrf_k`,
   and `rerank_top_n` are the highest-leverage knobs; watch `recall@1`/`recall@3`
   (ranking) and `recall@k` (pool quality) separately — the strict-k columns are
   the ones with headroom. Add a grid to `eval.sweeps` and run `make sweep`.
3. **Sweep chunking — it dominates quality.** `chunk_size`, `chunk_overlap`,
   `strategy`, and `contextual_enrichment` are ablation variables; add them as an
   `eval.sweeps` grid and `make sweep NAME=chunking` finds the sweet spot.
4. **Add HyDE / multi-query retrieval.** Generate a hypothetical answer (or 3
   query paraphrases) and retrieve on those — big recall win on vaguely-worded
   questions. Slots in next to `reformulate.py`.
5. **GraphRAG / multi-hop path.** A knowledge-graph or parent-document retriever
   for the multi-hop slice; report the multi-hop-slice delta (the harness already
   breaks metrics out by slice, so the win is directly measurable).
6. **Calibrate the abstention threshold.** `make sweep NAME=abstention` grids
   `agent.support_threshold` × `nli_entail_threshold` and reports the best point
   that doesn't regress safety — and on the current gold set those two are indeed
   already optimal. This bullet used to attribute the remaining over-abstention to
   "the CRAG gate and the gold set's phrasing". Measurement split that in two, and
   only part of it was CRAG:

   Of 14 over-abstentions, 13 had `recall_at_3` = 1.0 — retrieval had already
   found the answer. Isolating the two gates (`eval/experiments/over_abstention.py`)
   showed **5 are the CRAG gate declining before generating and 9 are the
   post-generation critic never accepting the answer**. Seven of the discarded
   answers were recovered and hand-checked against gold: **six were correct**, so
   this is real lost coverage, not a mislabelled metric.

   The two gates are not interchangeable. Removing the critic costs 4× the
   hallucination that removing CRAG does (0.205 vs 0.046 on the enriched subset),
   while `correct_abstention` stays 1.000 even with CRAG entirely off — the critic
   refuses the unanswerable on its own. **The critic is the load-bearing safety
   gate.**

   The CRAG half is a threshold artifact: `incorrect_threshold: 0.51` was tuned
   in-sample, and the five declined questions score 0.497 / 0.483 / 0.441 / 0.288 /
   0.267 — three miss by under 0.03. Chosen on the dev split alone the boundary is
   0.267. Measured over the full gold set, 0.25 recovers all five and halves
   over-abstention on **both** splits (dev 0.1364 → 0.0758, holdout 0.1429 →
   0.0952), at a cost of hallucination 0.0085 → 0.0256 and adversarial robustness
   1.000 → 0.944. **Adopted, then reverted** — the holdout is why. On dev it looks
   like a clear win, but dev is where the threshold was chosen; out-of-sample it
   recovered **1 question and cost 1 hallucination** (dev 9/66 → 5/66 over-abstained
   but 0 → 1 hallucinated; holdout 3/21 → 2/21 and 1 → 2). That is not worth
   tripling the headline metric, so `incorrect_threshold` stays **0.51**.

   Two things worth keeping from the attempt. It is **mode-sensitive**: how far the
   gate can relax depends on the critic behind it, and in `mock` the same value
   costs `correct_abstention` 0.917 → 0.667 for *no* coverage gain (over-abstention
   is already flat by 0.45), because mock's lexical critic cannot catch what the
   gate lets through. And 0.45 is an untested middle that would recover the two
   highest-scoring of the five — queued in `eval/experiments/crag_threshold.py`.

   Replacing the gate's IDF-coverage signal with the cross-encoder reranker (free,
   already computed) was tested and **refuted**: it is worse at every safety level
   (0.172 vs 0.460 false abstentions at 90% of unanswerable declined). Adversarial
   questions score 0.96–0.99 on a cross-encoder because it measures relevance, and
   an out-of-scope question about a covered topic is still relevant.
7. **Harden the semantic cache.** `make sweep NAME=cache` grids
   `cache.similarity_threshold`; watch `cache_false_hits` (already measured) to
   push hit-rate up without quality loss.
8. **Scale the gold set from logs.** The single biggest quality lever is more,
   realer eval data — every production failure becomes a permanent gold case, and
   a held-out split prevents overfitting prompts to the examples you re-read.
9. **Judge robustness.** Use a different model for the judge than the generator,
   add self-consistency (majority vote over N judge samples), and expand
   `judge_calibration.jsonl` — track Cohen's κ as the trust metric over time.

## License

MIT.
