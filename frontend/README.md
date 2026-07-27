# Agentic RAG — Frontend (Next.js)

Interactive playground for the Self-Correcting Agentic RAG engine. Ask a question
and watch the pipeline retrieve → grade (CRAG) → generate → self-correct → or
abstain, with live feature toggles, a streamed pipeline trace, clickable
citations, CRAG/guardrail/route badges, and cost/latency.

## Stack

Next.js 14 (App Router) · TypeScript · Tailwind · TanStack Query · SSE streaming.

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
