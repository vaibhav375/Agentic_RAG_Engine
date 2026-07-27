# Working conventions

- Work phase by phase; commit at each boundary (`phaseN-complete`). Never skip
  Phase 1 (gold set) or Phase 6 (eval/ablation).
- Every enhancement must be **measured against the baseline** — no unquantified
  claims. After each phase, run `make eval` and record the numbers.
- All tunables (models, k-values, thresholds, iteration cap, chunk size) live in
  `config/config.yaml`. Nothing hard-coded in source.
- Three runtime modes via `config.mode`: `mock` (deterministic, no keys, used by
  tests/CI/demo), `local` (sentence-transformers + Ollama), `api` (OpenAI/Anthropic).
  Every backend has a `mock` fallback so the pipeline is always runnable offline.
- Prefer the local/mock backend during development; run API evals deliberately to
  control cost.
- Confirm current model availability before pinning IDs (`config.yaml` holds them).
- State assumptions in comments/README when a tradeoff is ambiguous rather than
  stalling.

## Package layout

Installable package `arag` lives under `src/arag/`. Import as `from arag.retrieve.hybrid import ...`.

## Fast local loop

```
make install        # core only — mock mode works immediately
make test           # unit + integration, no network
make demo           # ingest + one query in mock mode
```

## Real models

```
make install-local  # or make install-api
# edit config.mode -> local|api, set provider fields + keys in .env
make ingest && make eval
```
