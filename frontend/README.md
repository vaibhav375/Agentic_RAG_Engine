# Agentic RAG — demo UI

Watch the engine decide whether it is *allowed* to answer.

Most RAG demos are a chat box. This one is built around the pipeline's actual
distinguishing behaviour: retrieving passages, judging whether they can answer the
question, writing an answer only if they can, and **refusing** when they can't.
Refusal is presented as a considered outcome, not an error — on the 117-question
benchmark the engine declines all 12 out-of-scope questions rather than guessing.

What you can see:

- **The pipeline ledger** fills live from the SSE stream, so on a slow local model
  you watch each stage land instead of staring at a blank panel. When the
  answerability gate declines, the ledger visibly stops *before* `generate` — the
  engine never wrote anything.
- **The ruling** — ANSWERED or DECLINED, with the numbers behind it: answerability
  score, fraction of claims the sources support, how many passes it took.
- **Citation → source linking.** Hover a citation and the passage it came from
  lights up in the evidence column. That link is the claim a chat transcript
  cannot show you.
- **Rewritten queries.** When the first answer isn't fully supported the engine
  broadens the question and searches again; those rewrites are listed.
- **Feature toggles** — turn off the answerability gate or self-correction and
  watch the safety behaviour disappear.

The four example questions are taken from the graded benchmark and labelled with
the behaviour they actually produce. (An earlier version offered a "should
abstain" example that does not abstain — its wording overlaps the corpus enough to
score 0.69 on the gate.)

## Stack

Next.js 14 (App Router) · TypeScript · Tailwind · SSE streaming. No data-fetching
or charting libraries — the page has one request path and doesn't need them.

## Run

1. Start the backend (from the repo root):

   ```bash
   make serve          # FastAPI on :8000 (mock mode, self-ingests on first run)
   ```

2. Start the frontend:

   ```bash
   cd frontend
   cp .env.local.example .env.local     # points at http://localhost:8000
   npm install
   npm run dev                          # http://localhost:3000
   ```

## What each control does

- **Pipeline toggles** (right panel) map 1:1 to the engine's config flags and are
  sent per-request, so flipping one and re-asking re-runs the same query through a
  different pipeline — a live ablation. Presets: **Baseline** (everything off) and
  **Full pipeline**.
- **Sample chips**: a factual question, a multi-hop question, an out-of-scope
  question (watch CRAG abstain), and a prompt-injection attempt (watch the
  guardrail flag it).

## Talks to

- `POST /query` and `POST /query/stream` (SSE) — answer + trace + grade + flags
- `GET /config`, `GET /corpus`, `GET /chunk/{id}`, `GET /eval/{name}`

Set `NEXT_PUBLIC_API_URL` to point at a deployed backend for production.
