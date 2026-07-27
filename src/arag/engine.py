"""Core engine: builds pipeline components from config and exposes the single
query entrypoint `answer_query` used by the CLI, the API, and the eval harness.

The pipeline behavior is entirely config-driven, which is what makes the ablation
study trivial — flip `retrieval.use_hybrid`, `retrieval.use_rerank`,
`agent.enabled`, `cache.enabled` and re-run.
"""

from __future__ import annotations

from dataclasses import dataclass

from arag.common.schemas import Answer, RetrievedChunk
from arag.common.telemetry import Trace
from arag.generate.answer import generate_answer
from arag.ingest.index import Store, load_store
from arag.providers import make_llm, make_nli, make_reranker
from arag.providers.base import Embedder, LanguageModel, NLIModel, Reranker
from arag.retrieve.hybrid import fuse_many as _fuse
from arag.retrieve.hybrid import retrieve as retrieve_candidates
from arag.retrieve.rerank import rerank as rerank_candidates


@dataclass
class Components:
    cfg: object
    store: Store
    llm: LanguageModel
    reranker: Reranker
    nli: NLIModel
    embedder: Embedder
    cache: object | None = None


def build_components(cfg, store: Store | None = None) -> Components:
    store = store or load_store(cfg)
    comp = Components(
        cfg=cfg,
        store=store,
        llm=make_llm(cfg),
        reranker=make_reranker(cfg),
        nli=make_nli(cfg),
        embedder=store.embedder,
    )
    if bool(cfg.get("cache.enabled", False)):
        from arag.cache.semantic_cache import build_cache

        comp.cache = build_cache(cfg, comp.embedder)
    return comp


def retrieve_contexts(
    comp: Components,
    retrieval_query: str,
    trace: Trace | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """Retrieve -> (rerank). Returns (full_ranked_list, top_n_contexts) without
    generating, so an answerability gate can inspect the context first."""
    cfg = comp.cfg
    trace = trace or Trace()
    r = cfg.retrieval
    top_n = int(r.get("rerank_top_n", 5))
    use_hyde = bool(r.get("use_hyde", False))
    use_decompose = bool(r.get("use_decompose", False))

    with trace.stage("retrieve", query=retrieval_query, hyde=use_hyde, decompose=use_decompose):
        # Multi-hop: split into sub-questions, retrieve each, fuse the pools.
        subqueries = [retrieval_query]
        if use_decompose:
            subqueries = comp.llm.decompose(retrieval_query) or [retrieval_query]

        pools = []
        for sq in subqueries:
            dense_text = comp.llm.hypothetical_document(sq) if use_hyde else None
            pools.append(retrieve_candidates(comp.store, sq, cfg, dense_text=dense_text))
        candidates: list[RetrievedChunk] = (
            pools[0] if len(pools) == 1 else _fuse(pools, int(r.get("rrf_k", 60)))
        )

    if bool(r.get("use_rerank", False)):
        with trace.stage("rerank", candidates=len(candidates)):
            ranked = rerank_candidates(
                comp.reranker,
                retrieval_query,
                candidates,
                top_n=None,
                fusion=str(r.get("rerank_fusion", "replace")),
                rrf_k=int(r.get("rrf_k", 60)),
            )
    else:
        ranked = candidates

    if bool(r.get("use_mmr", False)):
        from arag.retrieve.mmr import mmr_select

        with trace.stage("mmr", candidates=len(ranked)):
            contexts = mmr_select(comp.store, retrieval_query, ranked, top_n,
                                  lambda_=float(r.get("mmr_lambda", 0.7)))
    else:
        contexts = ranked[:top_n]
    return ranked, contexts


def generate_from(
    comp: Components,
    answer_query: str,
    ranked: list[RetrievedChunk],
    contexts: list[RetrievedChunk],
    trace: Trace | None = None,
) -> Answer:
    trace = trace or Trace()
    with trace.stage("generate", contexts=len(contexts)):
        ans = generate_answer(comp.llm, answer_query, contexts)
    ans.retrieved = ranked  # full ranked list for retrieval-quality metrics
    return ans


def run_retrieval_and_generation(
    comp: Components,
    retrieval_query: str,
    answer_query: str | None = None,
    trace: Trace | None = None,
) -> Answer:
    """One retrieve -> (rerank) -> generate pass. `retrieval_query` drives search;
    `answer_query` (default = retrieval_query) is what the answer must address."""
    answer_query = answer_query or retrieval_query
    trace = trace or Trace()
    ranked, contexts = retrieve_contexts(comp, retrieval_query, trace=trace)
    return generate_from(comp, answer_query, ranked, contexts, trace=trace)


def answer_query(comp: Components, query: str) -> Answer:
    """Full online path: guardrail -> cache -> route -> (self-correction | single pass) -> cache write."""
    cfg = comp.cfg
    trace = Trace()

    # 0. Input guardrail (defense-in-depth; flags injection/jailbreak attempts).
    from arag.agent.guardrails import scan_input

    input_flags = scan_input(query)

    # 1. Semantic cache
    if comp.cache is not None:
        with trace.stage("cache_lookup"):
            hit = comp.cache.get(query)
        if hit is not None:
            hit.from_cache = True
            hit.trace = trace.stages
            hit.input_flags = input_flags
            return hit

    # 2. Agentic self-correction (optionally gated by the router)
    if bool(cfg.get("agent.enabled", False)):
        from arag.agent.graph import run_self_correction
        from arag.agent.router import classify_query

        route = "full"
        max_iter = int(cfg.get("agent.max_iterations", 2))
        if bool(cfg.get("agent.router.enabled", False)):
            with trace.stage("route"):
                label = classify_query(comp.llm, query, cfg)
            if label == "simple":
                route, max_iter = "direct", 0  # critique once, but no retrieval retries
        ans = run_self_correction(comp, query, trace=trace, max_iter=max_iter)
        ans.route = route
    else:
        ans = run_retrieval_and_generation(comp, query, trace=trace)
        ans.route = "direct"

    ans.input_flags = input_flags
    ans.trace = trace.stages
    ans.cost_usd = trace.cost_usd
    ans.tokens = trace.tokens

    # 3. Cache write (skip abstentions — never cache a non-answer)
    if comp.cache is not None and not ans.abstained:
        comp.cache.put(query, ans)

    return ans
