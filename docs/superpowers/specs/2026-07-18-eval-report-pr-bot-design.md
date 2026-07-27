# Design: Eval Report PR Bot + Metric Trend

**Date:** 2026-07-18
**Status:** Implemented 2026-07-27 (`eval/pr_report.py`, `eval/ci_gate.py`,
`.github/workflows/ci.yml`, `tests/unit/test_pr_report.py`). The GitHub-side
behavior — sticky comment, history artifact, badge commit — is unverified until
the repo has a GitHub remote (see §8 note below).
**Author:** Vaibhav (with Claude)

## 1. Problem & goal

The engine already computes a full per-metric regression diff on every PR
([`eval/ci_gate.py`](../../../eval/ci_gate.py), wired into
[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)) — but the diff is
buried in CI logs where nobody reads it. The eval harness is this project's
strongest asset and it is currently invisible in the one place engineers actually
look: the pull request.

**Goal:** turn the existing eval signal into a visible, GitHub-native workflow
artifact — a bot that posts (and updates) a single sticky PR comment showing the
pass/fail verdict, per-metric deltas vs. a committed baseline, the per-slice
breakdown, and a trend of the headline metrics over recent runs; plus a
self-updating README badge for the current hallucination rate.

This showcases **AI/ML integrated into a real dev workflow (MLOps quality gate)** —
distinct from an agent/MCP-tooling story. It is a build on top of the existing
harness, not new ML.

### Non-goals (YAGNI)

- No new metrics or model changes; reuse `run_eval`'s summary verbatim.
- No external eval SaaS (W&B/Langfuse) — first-party GitHub only.
- No auto-opening PRs to bump the baseline in phase 1 (a local make target +
  optional `workflow_dispatch` covers it).
- Fork-PR write access is explicitly out of scope (see §7).

## 2. What the user sees

On every PR, one sticky comment (updates in place on new pushes):

```
## 🟢 Eval Report — regression gate PASSED

Full agentic pipeline over the 62-question gold set (mock mode, deterministic).
hallucination **0.0%** · faithfulness **1.000** · correct-abstention **1.000**

### Metrics vs. baseline (tolerance 0.05)
| Metric | Baseline | This PR | Δ | |
|---|---|---|---|---|
| hallucination_rate ↓ | 0.000 | 0.000 | +0.000 | ✅ |
| faithfulness ↑ | 1.000 | 1.000 | +0.000 | ✅ |
| ... | | | | |

### Per-slice (full pipeline)
| Slice | n | hallu | faith | recall@k | ans.F1 | robust |
| easy | 40 | 0.000 | 1.000 | 1.000 | 0.326 | - |
| ... | | | | | | |

### Trend (last 8 recorded runs)
hallucination ↓  ▆▆▆▃▁▁▁▁   0.306 → 0.000
faithfulness  ↑  ▃▃▃▆██████ 0.742 → 1.000

<sub>commit `a1b2c3d` · config `239e1859` · [dashboard artifact](…) · run #123</sub>
<!-- arag-eval-report -->
```

A failing gate flips the header to `## 🔴 Eval Report — regression gate FAILED`,
lists the offending rows with ❌, and the CI job exits non-zero (native failing
check). The README gains a badge: `hallucination | 0.0%` (green).

## 3. Architecture

Three seams, all reusing existing code:

```
run_eval(cfg)  ──► summary dict  (unchanged; already exists)
      │
      ├─► ci_gate.evaluate_gate(summary, baseline, budgets, tol)  ← REFACTORED OUT
      │        returns (ok: bool, rows: [MetricRow])
      │
      ├─► registry.load_history() / compare_runs()  ← REUSED for trend
      │
      └─► pr_report.render_comment(...)  ← NEW: pure fn, str out, sticky marker
                │
   ci.yml step writes comment.md ──► github-script posts/updates sticky comment
                                      + uploads history artifact + dashboard.html
                                      + (main only) writes & commits badge.json
```

### 3.1 New module: `eval/pr_report.py`

Pure, deterministic, no network. Reuses `report._fmt`, `registry._TRACKED`,
`registry.load_history`, `registry.compare_runs`.

- `sparkline(values: list[float]) -> str` — 8-level unicode block sparkline
  (`▁▂▃▄▅▆▇█`), normalized over min/max. Zero dependencies.
- `render_metric_table(rows) -> str` — from `evaluate_gate`'s structured rows,
  with ↑/↓ direction glyphs and ✅/❌ flags.
- `render_trend(history, metrics=("hallucination_rate","faithfulness"), n=8) -> str`
  — sparkline + `first → last` per tracked metric; omitted with a one-line note
  if `< 2` runs recorded.
- `render_comment(summary, baseline, rows, ok, history, links) -> str` — assembles
  header (verdict + one-line KPIs), metric table, per-slice table (via a small
  local variant of `report._slice_table` fed `summary["by_slice"]`), trend, and a
  footer with `git_sha`, `config_hash`, dashboard link, run URL. Ends with the
  hidden marker `<!-- arag-eval-report -->`.

### 3.2 Refactor: `eval/ci_gate.py`

Extract the inline diff loop into a reusable function so the CLI gate and the PR
report share one source of truth:

```python
@dataclass
class MetricRow:
    name: str; baseline: float | None; current: float
    delta: float; higher_is_better: bool; regressed: bool

def evaluate_gate(summary, baseline, budgets, tolerance) -> tuple[bool, list[MetricRow]]:
    ...  # existing higher/lower-is-better logic, now returning structured rows
```

- `main()` keeps its current stdout behavior (backwards compatible) by rendering
  `MetricRow`s.
- New flags: `--report PATH` (write the markdown comment body via
  `pr_report.render_comment`) and `--github-output` (append `gate_passed=true|false`
  and `hallucination=<pct>` to `$GITHUB_OUTPUT` for the badge/comment steps).
- Budgets/tolerance now **default from config** (`eval.ci_gate.*`, see §4), CLI
  args override — honoring the repo's "nothing hard-coded" convention.

### 3.3 Trend persistence (`history.jsonl` across ephemeral runners)

`registry.record_run` already appends every run to `eval/results/history.jsonl`.
CI runners are ephemeral, so the workflow carries it between runs:

1. `actions/cache` restores prior history into `eval/results/` before the eval step
   (`key: eval-history-<run_id>`, `restore-keys: eval-history-`).
2. Eval appends this run's line (existing behavior).
3. The cache's post-step saves it under this run's unique key.

First run (no prior cache) → trend shows "first recorded run." No commit to the
repo is required for the trend.

> **Revised during implementation.** This section originally specified
> `download-artifact`/`upload-artifact`. That does not work:
> `actions/download-artifact@v4` only resolves artifacts within the *current*
> run, so every run restored nothing and the trend was permanently stuck at
> "first recorded run" (observed on PR #1). `actions/cache` is the first-party
> mechanism for carrying a file *between* runs, and its restore-keys fall back to
> the default branch's cache, which is what makes PR runs show main's trend.
> History is still uploaded inside the `eval-dashboard` artifact for download.

### 3.4 README badge (self-updating, dependency-free)

A shields.io **endpoint** badge reads a committed JSON file:

```
![hallucination](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/<user>/<repo>/main/eval/results/badge.json)
```

A **main-branch-only** job writes `eval/results/badge.json`
(`{schemaVersion:1,label:"hallucination",message:"0.0%",color:"brightgreen"}` —
green ≤5%, yellow ≤15%, red otherwise) and commits it with
`[skip ci]`. PRs never touch it, so the badge always reflects merged `main`.

### 3.5 Workflow: `.github/workflows/ci.yml`

- Job-level `permissions: { contents: write, pull-requests: write }`.
- After the existing lint/test/gold steps, replace the bare gate step with:
  restore history artifact → `python -m eval.ci_gate --report comment.md
  --github-output` → `python -m eval.dashboard` → upload `dashboard.html` +
  `eval-history` artifacts → `actions/github-script` posts/updates the sticky
  comment (find by marker, edit if present else create).
- New `badge` job: `if: github.ref == 'refs/heads/main'` → run gate, write
  `badge.json`, commit.
- The gate's non-zero exit still fails the check (native status).

### 3.6 Baseline update ergonomics

- `make update-baseline` → `python -m eval.ci_gate --update-baseline` (flag already
  exists) then the author commits `eval/results/ci_baseline.json` in the same PR
  that legitimately moves a metric. The review diff makes the change explicit.
- Optional `workflow_dispatch: update-baseline` job that runs the same command and
  commits to a branch — documented but not required for phase 1.

## 4. Config (new `eval.ci_gate` block in `config/config.yaml`)

```yaml
eval:
  ci_gate:
    max_hallucination: 0.15
    min_correct_abstention: 0.5
    tolerance: 0.05
    baseline_path: eval/results/ci_baseline.json
    trend_runs: 8
```

`ci_gate.py` reads these as defaults; existing CLI flags override.

## 5. Files touched

| File | Change |
|---|---|
| `eval/pr_report.py` | **new** — sparkline, tables, `render_comment` |
| `eval/ci_gate.py` | refactor `evaluate_gate`; add `--report`/`--github-output`; config defaults |
| `config/config.yaml` | **new** `eval.ci_gate` block |
| `.github/workflows/ci.yml` | permissions, history artifact, comment step, badge job, dashboard artifact |
| `eval/results/badge.json` | **new** — generated + committed by main job |
| `Makefile` | `report`, `update-baseline` targets |
| `README.md` | badge at top; short "CI eval bot" subsection with a screenshot |
| `tests/unit/test_pr_report.py` | **new** |

## 6. Testing (all mock mode, no network — matches repo CI-reproducibility rule)

Unit (`tests/unit/test_pr_report.py`):
- `sparkline` — monotonic input → ascending blocks; flat input → single level; empty → "".
- `evaluate_gate` — synthetic summaries for: clean pass, one regressed
  higher-is-better metric (flag + `ok=False`), one regressed lower-is-better
  metric, no-baseline (absolute budgets only, no REGRESSION rows).
- `render_comment` — asserts the sticky marker is present, the verdict header
  matches `ok`, a seeded regression renders ❌ on the right row, and the per-slice
  table appears when `by_slice` is populated.
- `render_trend` — `<2` runs → note, not a sparkline; `≥2` → `first → last` string.

Manual verification before "done":
- `make report` renders `comment.md` locally from the committed baseline + a fresh
  eval; eyeball it.
- Push a branch that seeds a deliberate regression (e.g. raise a threshold) and
  confirm the comment flips to FAILED and the check fails; revert.

## 7. Edge cases & risks

- **No baseline** → comment shows a "baseline not set — run `make update-baseline`"
  note; gate still enforces absolute budgets; no REGRESSION rows.
- **No history** → trend section shows "first recorded run."
- **Fork PRs**: `GITHUB_TOKEN` is read-only on PRs from forks, so the comment-post
  step will fail. Mitigation: `continue-on-error: true` on the comment step + a
  workflow notice; the gate itself (the enforcement) still runs. We deliberately do
  **not** use `pull_request_target` (it would run with a write token against
  untrusted PR code — a security footgun). Documented as a known limitation.
- **Badge race** (two main merges): the badge job commits with `[skip ci]`; last
  write wins, which is correct (badge reflects latest main).
- **Sticky-comment dupes**: always find-by-marker before create; edit in place.

## 8. Success criteria

1. Opening a PR produces exactly one Eval Report comment; a second push edits it in
   place (no duplicates).
2. A seeded regression flips the verdict to FAILED **and** fails the CI check.
3. The README badge on `main` shows the current hallucination rate and recolors by
   threshold.
4. The trend renders from `history.jsonl` restored via artifact across runs.
5. All new tests pass in mock mode with no network; `make report` works locally.
6. No change to any existing metric value — this is pure surfacing.

### Verification status (2026-07-27)

All six criteria exercised against `vaibhav375/Agentic_RAG_Engine` (private).

1. **One comment, edited in place** — ✅ PR #1: the first push created the
   comment; the second edited it. Comment count stayed at 1.
2. **Seeded regression fails loudly** — ✅ raising the committed baseline's
   `answer_correctness` to 0.900 produced a 🔴 FAILED header, ❌ on that row
   only (the other 8 stayed ✅), and a red `test-and-eval` check. Reverting
   flipped the same comment to 🟢.
3. **Badge** — partially: `badge.json` is generated and committed correctly
   (`0.0%`, brightgreen), and the badge job correctly skips the commit when the
   value is unchanged. The shields.io image itself cannot render while the repo
   is private — shields fetches `raw.githubusercontent.com` anonymously. It will
   render as-is the moment the repo goes public.
4. **Trend across runs** — ✅ after the §3.3 fix, run #5 restored the prior
   run's history and rendered `last 2 recorded runs`. See the revision note in
   §3.3: the artifact-based design in the original spec could never have worked.
5. **Tests/local** — ✅ 18 new unit tests (54 total), ruff clean, `make report`
   renders `comment.md` locally.
6. **No metric moved** — ✅ gate output identical to pre-refactor.

Deviations from the design: the badge is written by `ci_gate --badge PATH`
rather than a separate script, so the badge job reuses one eval run; and §3.3's
artifact mechanism was replaced by `actions/cache` (see the note there).

One caveat worth recording: `mrr` came out 0.912 locally (macOS) vs. 0.913 on
the Linux runner. Mock mode is deterministic *per platform*, not bit-identical
across them — well inside the 0.05 tolerance, but the "CI reproduces these exact
numbers" claim in `RESULTS.md` should be read as ±0.001 cross-platform.
