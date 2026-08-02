"""FastAPI serving layer for the web UI.

    POST /query          -> full answer + citations + trace + CRAG grade + guardrail flags
    POST /query/stream    -> Server-Sent Events: one event per pipeline stage, then the answer
    GET  /health          -> readiness + index/config summary
    GET  /config          -> current pipeline flags (UI default toggle state)
    GET  /stats           -> cache stats
    GET  /corpus          -> documents + chunk counts
    GET  /chunk/{id}      -> a single chunk (for clickable citations)
    GET  /eval/{name}     -> eval artifacts (ablation|selective|history|calibration|gold)

Per-request `flags` let the UI toggle pipeline components live (hybrid, rerank,
HyDE, MMR, decompose, agent, CRAG, router, cache) on the already-loaded index —
so the frontend can demo the ablation interactively without re-ingesting.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from arag.common.config import load_config
from arag.common.schemas import Answer
from arag.engine import Components, answer_query, build_components

# Flags that only change online pipeline behavior (safe to toggle per-request on
# the same index). Chunking/embedding changes are excluded — they need re-ingest.
_FLAG_MAP = {
    "useHybrid": "retrieval.use_hybrid",
    "useRerank": "retrieval.use_rerank",
    "useHyde": "retrieval.use_hyde",
    "useMmr": "retrieval.use_mmr",
    "useDecompose": "retrieval.use_decompose",
    "agent": "agent.enabled",
    "crag": "agent.crag.enabled",
    "router": "agent.router.enabled",
    "cache": "cache.enabled",
}


class QueryRequest(BaseModel):
    query: str
    flags: dict[str, bool] | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    abstained: bool
    iterations: int
    route: str
    from_cache: bool
    retrieval_grade: str | None = None
    retrieval_grade_score: float | None = None
    input_flags: list[str] = []
    citations: list[dict] = []
    contexts: list[dict] = []
    trace: list[dict] = []
    cost_usd: float = 0.0
    tokens: dict = {}
    support_fraction: float | None = None


def _answer_to_response(ans: Answer) -> QueryResponse:
    return QueryResponse(
        query=ans.query,
        answer=ans.answer,
        abstained=ans.abstained,
        iterations=ans.iterations,
        route=ans.route,
        from_cache=ans.from_cache,
        retrieval_grade=ans.retrieval_grade,
        retrieval_grade_score=ans.retrieval_grade_score,
        input_flags=ans.input_flags,
        citations=[c.model_dump() for c in ans.citations],
        contexts=[
            {
                "chunk_id": rc.chunk.chunk_id,
                "doc_id": rc.chunk.doc_id,
                "section": rc.chunk.section,
                "score": rc.score,
                "source": rc.source,
                "ranks": rc.ranks,
                "text": rc.chunk.text,
            }
            for rc in ans.contexts
        ],
        trace=[s.model_dump() for s in ans.trace],
        cost_usd=ans.cost_usd,
        tokens=ans.tokens,
        support_fraction=ans.critique.support_fraction if ans.critique else None,
    )


def create_app(config_path: str = "config/config.yaml") -> FastAPI:
    cfg = load_config(os.environ.get("ARAG_CONFIG", config_path))
    app = FastAPI(title="Agentic RAG Engine", version="0.1.0")

    origins = os.environ.get("ARAG_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Components] = {}

    def base_components() -> Components:
        if "components" not in state:
            try:
                state["components"] = build_components(cfg)
            except FileNotFoundError:
                # No index yet -> ingest on first use so the server is self-starting.
                from arag.ingest.index import build_index

                store = build_index(cfg)
                state["components"] = build_components(cfg, store=store)
        return state["components"]

    def components_for(flags: dict[str, bool] | None) -> Components:
        """Return base components, or a variant with per-request flag overrides
        applied on the SAME loaded index (no re-ingest)."""
        base = base_components()
        if not flags:
            return base
        overrides: dict = {}
        for key, dotted in _FLAG_MAP.items():
            if key in flags:
                overrides[dotted] = bool(flags[key])
        if overrides.get("agent.enabled"):
            overrides.setdefault("agent.critic", "both")
        variant_cfg = cfg.with_overrides(overrides)
        return build_components(variant_cfg, store=base.store)

    @app.on_event("startup")
    def _startup() -> None:
        base_components()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def playground() -> str:
        """Zero-dependency playground served by the API itself.

        The Next.js app under frontend/ is the richer UI, but it needs Node and
        an npm install. This single file needs neither: `make serve` and open
        the root URL. It is the fastest way to watch the pipeline actually run —
        retrieve, CRAG grade, generate, critique, abstain — with the per-stage
        trace and citations.
        """
        return (Path(__file__).parent / "playground.html").read_text()

    @app.get("/health")
    def health() -> dict:
        comp = state.get("components")
        return {
            "status": "ok" if comp is not None else "starting",
            "mode": cfg.get("mode"),
            "index": comp.store.meta if comp else None,
        }

    @app.get("/config")
    def config() -> dict:
        return {
            "mode": cfg.get("mode"),
            "flags": {key: bool(cfg.get(dotted, False)) for key, dotted in _FLAG_MAP.items()},
        }

    @app.get("/stats")
    def stats() -> dict:
        comp = base_components()
        return {"cache": comp.cache.stats() if comp.cache else "disabled"}

    @app.get("/corpus")
    def corpus() -> dict:
        comp = base_components()
        docs: dict[str, int] = {}
        for c in comp.store.chunks:
            docs[c.doc_id] = docs.get(c.doc_id, 0) + 1
        return {"n_docs": len(docs), "n_chunks": len(comp.store.chunks),
                "docs": [{"doc_id": k, "chunks": v} for k, v in sorted(docs.items())]}

    @app.get("/chunk/{chunk_id}")
    def chunk(chunk_id: str) -> dict:
        comp = base_components()
        c = comp.store.get(chunk_id)
        if c is None:
            raise HTTPException(404, "chunk not found")
        return c.model_dump()

    @app.get("/eval/{name}", response_model=None)
    def eval_artifact(name: str):
        allowed = {
            "ablation": "eval/results/ablation.json",
            "selective": "eval/results/selective.json",
            "history": "eval/results/history.jsonl",
            "gold": cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl"),
        }
        if name == "calibration":
            from eval.calibrate_judge import calibrate

            return calibrate(cfg)
        if name not in allowed:
            raise HTTPException(404, f"unknown artifact '{name}'")
        p = Path(allowed[name])
        if not p.exists():
            return {"available": False, "hint": "run `make ablation` first"}
        if p.suffix == ".jsonl":
            return {"items": [json.loads(x) for x in p.read_text().splitlines() if x.strip()]}
        return json.loads(p.read_text())

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        comp = components_for(req.flags)
        return _answer_to_response(answer_query(comp, req.query))

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest):
        comp = components_for(req.flags)

        async def gen():
            yield _sse("status", {"message": "running pipeline"})
            ans = await asyncio.to_thread(answer_query, comp, req.query)
            # Reveal the pipeline stage-by-stage from the recorded trace.
            for s in ans.trace:
                yield _sse("stage", s.model_dump())
                await asyncio.sleep(0.12)
            yield _sse("answer", _answer_to_response(ans).model_dump())
            yield _sse("done", {})

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
