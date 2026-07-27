# Self-Correcting Agentic RAG Engine + Evaluation Harness

<!-- The badge job on `main` regenerates eval/results/badge.json each merge.
     shields.io fetches it anonymously, so the badge only renders once this
     repo is public; re-point the URL if you fork it. -->
![hallucination](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/vaibhav375/Agentic_RAG_Engine/main/eval/results/badge.json)

A production-style Retrieval-Augmented Generation engine that **improves its own
answers**: it retrieves with a hybrid dense+sparse pipeline, reranks with a
cross-encoder, generates a grounded answer with citations, then **critiques its
own answer and re-queries — or abstains — when the answer isn't supported** by
the retrieved context. The whole thing is wrapped in an **automated evaluation
harness** that measures faithfulness, answer relevance, context precision/recall,
and an explicit **hallucination rate**, and proves a **measured reduction in
hallucination** vs. a naive RAG baseline through an ablation study.

> **Headline (reproducible, `mock` mode):** hallucination rate **30.6% → 0.0%**,
> correct abstention on unanswerable questions **0% → 100%**, and adversarial /
> prompt-injection robustness **0% → 100%** across the ablation, on a 62-question
> technical-docs benchmark (easy · multi-hop · unanswerable · adversarial). Run
> `make ablation` to regenerate [`RESULTS.md`](RESULTS.md).

## Why this is more than "a chatbot"

The value is in three things, in order: **(1) the measured eval/ablation
results, (2) the self-correction loop, (3) the production serving concerns**
(hybrid retrieval, reranking, semantic caching, latency/cost, Docker).

Beyond the base spec it implements the named techniques a modern RAG/agent
engineer is expected to know:

- **CRAG answerability gate** (Corrective RAG) — grades whether retrieved context
  can actually answer the query (IDF-weighted relevance) and **declines when it
  can't**. This is what drives hallucination to 0 and abstention to 100% on
  out-of-scope questions.
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
- **Adversarial / prompt-injection robustness** — a dedicated eval slice of
  injection and false-premise traps, with a robustness metric (100% on the set).
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
make test        # 19 unit + integration tests, no network
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
  eval/gold_qa.jsonl     62 Qs: easy / multi-hop / unanswerable / adversarial
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

Or the whole stack in one command (Qdrant + Redis + API + UI):

```bash
docker compose up               # UI on :3000, API on :8000
```

The UI drives these endpoints, added for the app: `POST /query` (+ per-request
feature flags), `POST /query/stream` (SSE), `GET /config`, `/corpus`,
`/chunk/{id}`, and `/eval/{ablation|selective|history|calibration}`.

### 4. Real models

```bash
make install-api                 # or: make install-local
cp .env.example .env             # add OPENAI_API_KEY / ANTHROPIC_API_KEY
# edit config/config.yaml: mode: api, llm.provider: openai, embeddings.provider: openai
make ingest && make eval
```

Everything is config-driven — models, `k` values, thresholds, iteration cap,
chunk size — via `config/config.yaml`, overridable per-field with
`ARAG_SECTION__KEY` env vars (e.g. `ARAG_RETRIEVAL__USE_HYBRID=true`).

### 5. Full Docker stack (Qdrant + Redis + app)

```bash
docker compose up          # brings up qdrant, redis, and the API
```

## Evaluation methodology

`RESULTS.md` reports every metric **baseline → enhanced**, as an **ablation
table** (rows = pipeline configs, columns = metrics) so each component's
contribution is visible:

| Metric | What it measures |
|---|---|
| **Hallucination rate** (headline) | % of answers with ≥1 unsupported claim (unanswerable Qs: answering at all counts) |
| Faithfulness / groundedness | fraction of answer claims entailed by retrieved context |
| Answer correctness (token-F1) | overlap with the gold answer |
| Context precision / recall | retrieval quality vs. gold supporting docs |
| Correct abstention | on unanswerable questions, does it decline instead of fabricating |
| Latency p50/p95 & cost/query | with/without cache and router |

**Retrieval vs. generation are scored separately** — `recall@k`, `precision@k`,
and `MRR` measure the retrieved pool; faithfulness/relevance/correctness measure
the answer given that pool. A regression then points at the half to fix instead
of a blended score that hides it.

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
make dashboard     # self-contained HTML eval dashboard (open in a browser)
```

## Resume bullets (values filled from `RESULTS.md`)

- Built a self-correcting agentic RAG engine (Corrective-RAG answerability gate +
  NLI-cross-checked LLM-as-judge critic) that re-issues its own retrievals or
  abstains, cutting hallucination from **30.6%** to **0.0%** and reaching **100%**
  correct abstention and **100%** prompt-injection robustness on a 62-question
  benchmark.
- Implemented hybrid dense + sparse (BM25 + RRF) retrieval with cross-encoder
  reranking, HyDE, and multi-hop query decomposition, served behind FastAPI with
  a semantic cache and a full Dockerized stack.
- Developed an automated evaluation harness — retrieval (recall@k / MRR) vs.
  generation (faithfulness / hallucination) metrics, per-slice breakdowns,
  bootstrap CIs, judge calibration (Cohen's κ), and an experiment registry —
  wired into CI as a per-metric regression gate.

## Notes / limitations

- `mock` mode simulates a naive generator that fabricates on weak context so the
  eval measures a real hallucination signal; absolute numbers shift with real
  models, but the *direction and mechanism* (self-correction ↓ hallucination,
  abstention on unanswerables) hold in every mode.
- Mock embeddings are lexical, so hybrid/rerank precision gains are muted vs.
  neural embeddings — see the note in `RESULTS.md`.
- No fine-tuning, auth, or multi-tenancy — out of scope by design.

## Roadmap & how to get the best results

Concrete next steps, ordered by return-on-effort, with the knob that drives each:

1. **Run real models for the headline numbers.** `mode: api` with
   `embeddings.provider: openai` (or `local` with bge/e5). Neural embeddings make
   the hybrid/rerank rows move much more than the lexical mock shows, and a real
   NLI/LLM judge fixes the negation/quantity errors the calibration surfaces.
2. **Tune retrieval before touching prompts.** `k_dense`/`k_sparse`, `rrf_k`,
   and `rerank_top_n` are the highest-leverage knobs; watch `recall@k` (pool
   quality) and `MRR` (ranking) separately. Rerank helps precision most when the
   fused pool is large (raise `fetch_k`) and `rerank_top_n` is small.
3. **Sweep chunking — it dominates quality.** `chunk_size`, `chunk_overlap`,
   `strategy`, and `contextual_enrichment` are ablation variables; a small script
   that grids them and re-runs `make eval` finds the sweet spot fast.
4. **Add HyDE / multi-query retrieval.** Generate a hypothetical answer (or 3
   query paraphrases) and retrieve on those — big recall win on vaguely-worded
   questions. Slots in next to `reformulate.py`.
5. **GraphRAG / multi-hop path.** A knowledge-graph or parent-document retriever
   for the multi-hop slice; report the multi-hop-slice delta (the harness already
   breaks metrics out by slice, so the win is directly measurable).
6. **Calibrate the abstention threshold.** `agent.support_threshold` and
   `nli_entail_threshold` trade hallucination against over-abstention; sweep them
   and pick the point on the curve your use-case wants (the ablation already
   reports both `hallucination_rate` and `over_abstention_rate`).
7. **Harden the semantic cache.** Sweep `cache.similarity_threshold` and watch
   `cache_false_hits` (already measured) to push hit-rate up without quality loss.
8. **Scale the gold set from logs.** The single biggest quality lever is more,
   realer eval data — every production failure becomes a permanent gold case, and
   a held-out split prevents overfitting prompts to the examples you re-read.
9. **Judge robustness.** Use a different model for the judge than the generator,
   add self-consistency (majority vote over N judge samples), and expand
   `judge_calibration.jsonl` — track Cohen's κ as the trust metric over time.

## License

MIT.
