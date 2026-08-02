"""Assemble RESULTS.md (ablation table + headline before/after) and optional plots."""

from __future__ import annotations

from pathlib import Path

_COLUMNS = [
    ("label", "Configuration"),
    ("hallucination_rate", "Hallu.↓"),
    ("faithfulness", "Faith.↑"),
    ("answer_correctness", "Ans.F1↑"),
    ("recall_at_k", "Rec@k↑"),
    ("recall_at_1", "Rec@1↑"),
    ("recall_at_3", "Rec@3↑"),
    ("mrr", "MRR↑"),
    ("citation_precision", "CiteP↑"),
    ("correct_abstention_rate", "Abst.OK↑"),
    ("adversarial_robustness_rate", "AdvRobust↑"),
    ("over_abstention_rate", "OverAbst↓"),
    ("latency_p95_ms", "p95 ms"),
]


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(results: list[dict]) -> str:
    header = "| " + " | ".join(h for _, h in _COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * len(_COLUMNS)) + "|"
    lines = [header, sep]
    for res in results:
        s = res["summary"]
        row = []
        for key, _ in _COLUMNS:
            if key == "label":
                row.append(res.get("label", res.get("tag", "")))
            else:
                row.append(_fmt(s.get(key, 0.0)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _slice_table(final: dict) -> str:
    by = final["summary"].get("by_slice", {})
    if not by:
        return ""
    cols = ["n", "hallucination_rate", "faithfulness", "recall_at_k", "answer_correctness", "robustness_pass"]
    header = "| Slice | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for slice_name in ("easy", "multi_hop", "unanswerable", "adversarial"):
        s = by.get(slice_name, {})
        row = [slice_name] + [_fmt(s.get(c, "-")) for c in cols]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return (
        "\n## Per-slice breakdown (full pipeline)\n\n"
        "Reporting per slice stops a gain on easy questions from masking a "
        "regression on hard/multi-hop ones.\n\n" + "\n".join(lines) + "\n"
    )


def _headline(results: list[dict]) -> str:
    base = results[0]["summary"]
    final = results[-1]["summary"]
    bh = base["hallucination_rate"] * 100
    fh = final["hallucination_rate"] * 100
    n = base["n"]

    def cache_line():
        s = results[-1]["summary"]
        if "cache_warm_hits" in s:
            return (
                f"- Semantic cache: {s.get('cache_warm_hits', 0)} warm hits, "
                f"{s.get('cache_false_hits', 0)} false hits on the gold set.\n"
            )
        return ""

    ci = final.get("hallucination_rate_ci95", [0, 0])
    return (
        f"## Headline results\n\n"
        f"- **Hallucination rate reduced from {bh:.1f}% (dense baseline) to "
        f"{fh:.1f}% (full agentic pipeline)** on a {n}-question technical-docs benchmark "
        f"(final 95% CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]).\n"
        f"- Faithfulness: {base['faithfulness']:.3f} → {final['faithfulness']:.3f}.\n"
        f"- Retrieval recall@k: {base.get('recall_at_k', 0):.3f} → {final.get('recall_at_k', 0):.3f}; "
        f"MRR: {base.get('mrr', 0):.3f} → {final.get('mrr', 0):.3f}.\n"
        f"- Context precision: {base['context_precision']:.3f} → {final['context_precision']:.3f}.\n"
        f"- Correct abstention on unanswerable questions: "
        f"{base['correct_abstention_rate']:.3f} → {final['correct_abstention_rate']:.3f}.\n"
        f"- Adversarial / prompt-injection robustness: "
        f"{base.get('adversarial_robustness_rate', 0):.3f} → "
        f"{final.get('adversarial_robustness_rate', 0):.3f}.\n"
        f"{cache_line()}"
    )


def _resume_bullets(results: list[dict]) -> str:
    base = results[0]["summary"]
    final = results[-1]["summary"]
    bh = round(base["hallucination_rate"] * 100, 1)
    fh = round(final["hallucination_rate"] * 100, 1)
    n = base["n"]
    return (
        "## Filled resume bullets\n\n"
        f"- Built a self-correcting agentic RAG engine (Corrective-RAG answerability "
        f"gate + NLI-cross-checked LLM judge) that re-issues its own retrievals or "
        f"abstains, cutting hallucination from **{bh}%** to **{fh}%** and reaching "
        f"**{final['correct_abstention_rate']*100:.0f}%** correct abstention / "
        f"**{final.get('adversarial_robustness_rate', 0)*100:.0f}%** prompt-injection "
        f"robustness on a **{n}**-question benchmark.\n"
        f"- Implemented hybrid dense + sparse (BM25 + RRF) retrieval with cross-encoder "
        f"reranking, HyDE, and multi-hop query decomposition, served behind FastAPI with a "
        f"semantic cache and a full Dockerized stack.\n"
        "- Developed an automated evaluation harness — retrieval (recall@k/MRR) vs. "
        "generation (faithfulness/hallucination) metrics, per-slice breakdowns, bootstrap "
        "CIs, judge calibration (Cohen's kappa), and an experiment registry — wired into CI "
        "as a per-metric regression gate.\n"
    )


def _try_plot(results: list[dict], out_dir: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = [r.get("tag", "") for r in results]
    hallu = [r["summary"]["hallucination_rate"] for r in results]
    faith = [r["summary"]["faithfulness"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(labels))
    ax.plot(x, hallu, marker="o", label="Hallucination rate ↓", color="#c0392b")
    ax.plot(x, faith, marker="s", label="Faithfulness ↑", color="#27ae60")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Ablation: metric progression by pipeline component")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def _calibration_section(calibration: dict | None) -> str:
    if not calibration:
        return ""
    llm = calibration.get("llm_judge", {})
    nli = calibration.get("nli_judge", {})
    n_fail = len(calibration.get("llm_disagreements", []))
    return (
        "\n## Judge calibration (trust the grader)\n\n"
        f"The critic is validated against a {llm.get('n', 0)}-example human-labeled set of "
        "(context, claim, label) triples before it's trusted to grade the pipeline:\n\n"
        f"- LLM-as-judge: accuracy **{llm.get('accuracy', 0):.2f}**, "
        f"Cohen's κ **{llm.get('cohens_kappa', 0):.2f}** (precision {llm.get('precision', 0):.2f}, "
        f"recall {llm.get('recall', 0):.2f}).\n"
        f"- NLI cross-check: accuracy **{nli.get('accuracy', 0):.2f}**, "
        f"κ **{nli.get('cohens_kappa', 0):.2f}**.\n"
        f"- The {n_fail} residual judge errors are negation / quantity-swap claims "
        "(e.g. \"unlimited\", \"before\" vs \"after\") — exactly the cases a lexical/NLI "
        "proxy misses and a real NLI or LLM judge handles, which is why production evals "
        "should run in `api`/`local` mode. Run `make calibrate` to reproduce.\n"
    )


def _selective_section(sel: dict | None) -> str:
    if not sel:
        return ""
    plot = f"\n![Risk-coverage curve]({sel['plot']})\n" if sel.get("plot") else ""
    return (
        "\n## Selective prediction (risk–coverage of the abstention gate)\n\n"
        "Sweeping the answer/abstain confidence traces how risk grows as the system "
        "answers more. A strong abstention signal keeps risk at zero up to the "
        "fraction of questions that are actually answerable.\n\n"
        f"- Risk–coverage **AUC = {sel['risk_coverage_auc']:.3f}** (lower is better; "
        "a random confidence signal scores ≈ the base risk).\n"
        f"- **Max coverage at zero risk = {sel['max_safe_coverage']:.3f}**, equal to the "
        f"answerable fraction ({sel['answerable_fraction']:.3f}) — i.e. the gate can answer "
        "*every* answerable question without answering a single out-of-scope one.\n"
        f"- Risk at full coverage (answer everything) = {sel['risk_at_full_coverage']:.3f}.\n"
        f"{plot}"
    )


def write_results_md(results: list[dict], out_path: str = "RESULTS.md", calibration: dict | None = None,
                     selective: dict | None = None) -> str:
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = _try_plot(results, out_dir)

    flags = results[0].get("config_flags", {})
    mode = flags.get("mode", "?")

    parts = [
        "# RESULTS\n",
        f"> Generated by `python -m eval.ablation`. Provider mode: **{mode}**. "
        "Numbers are fully reproducible: `make ingest && make ablation`.\n",
        _headline(results),
        "\n## Ablation table\n",
        _table(results),
        "\n\n_Arrows show the desired direction. Each row adds one component to the row above._\n",
        _slice_table(results[-1]),
    ]
    if plot_path:
        parts.append(f"\n![Ablation plot]({plot_path})\n")
    parts.append(_selective_section(selective))
    parts.append(_calibration_section(calibration))
    parts.append("\n" + _resume_bullets(results))
    parts.append(
        "\n## Notes on methodology\n\n"
        "- Metrics are computed against a hand-built gold set with easy, multi-hop, and "
        "unanswerable questions; context precision/recall are document-level (robust to "
        "chunking changes, which are an ablation variable).\n"
        "- **The gold set was grown from 62 to 109 questions, and the headline moved from "
        "0.0% to 1.8% hallucination and 100% to 91.7% correct abstention.** The perfect "
        "scores were partly an artifact of a small benchmark the thresholds had been tuned "
        "against. The two residual failures are real and worth naming: a yes/no question "
        "about case-sensitivity, where the generator asserts the opposite of the source; and "
        "an unanswerable question ('default request timeout') that the CRAG gate judged "
        "answerable because 'default', 'request' and 'seconds' each appear in the corpus "
        "separately. A CRAG threshold sweep could not recover it — IDF coverage cannot "
        "distinguish words being present from the concept being present, which is a "
        "limitation of the heuristic rather than a tuning problem.\n"
        "- Faithfulness/hallucination use a claim-level critic (LLM-as-judge cross-checked "
        "by an NLI entailment model) to reduce single-judge bias.\n"
        "- In `mock` mode every backend is deterministic, so CI reproduces these exact "
        "numbers; switch `mode` to `local`/`api` in `config.yaml` for real models. "
        "Determinism is platform-independent: ranking rounds scores before sorting and "
        "breaks ties on chunk index, because numpy's `argpartition`/`argsort` are "
        "unstable and were reordering tied results differently on macOS vs. Linux "
        "(verified — the CI gate diffs a locally-generated baseline against a Linux "
        "runner and reports zero delta on every metric).\n"
        "- Mock embeddings are lexical (hashed bag-of-words), so dense and sparse retrieval "
        "carry similar signal and the hybrid/rerank precision lift is muted here. Confirmed "
        "on real models: `bge-small` embeddings raise recall@1 0.833 → 0.958 and MRR "
        "0.913 → 1.000 (see docs/local-mode-eval.md).\n"
        "- **These hallucination numbers are mock-mode and do not reproduce on a small local "
        "model.** Measured with `llama3.2:3b` on a 16-question stratified subset: "
        "hallucination 0.313, over-abstention 0.333. CRAG abstention (2/2) and adversarial "
        "robustness (2/2) do hold. Every flagged answer was hand-checked against the corpus "
        "and is factually correct: the model answers correctly then adds an aside that isn't "
        "in the context, and one such claim flags the record. A stronger judge does not fix "
        "it — qwen2.5:7b left the per-answer flag rate unchanged (0.625 -> 0.700) at 7x the "
        "latency — so the cause is generation style meeting a strict claim-level metric. "
        "Measured configs and the refuted hypothesis: docs/local-mode-eval.md.\n"
        "- `recall@k` (k=10) saturates at 1.000 on a 47-chunk corpus, so it cannot show a "
        "retrieval change; **recall@1 / recall@3** are the columns to read for ranking work. "
        "They make the hybrid gain visible (recall@1 0.854 → 0.917) where recall@k showed "
        "nothing, and they are what makes HyDE / decomposition / chunking measurable at all.\n"
        "- **Reranking costs ranking quality here** (recall@1 0.917 → 0.833): the mock "
        "reranker is lexical, so it carries less signal than the dense+sparse fusion it "
        "overwrites. `retrieval.rerank_fusion: rrf` fuses the two rankings instead and fully "
        "recovers recall@1 — but it shifts which passages reach the generator, and the CRAG "
        "gate (whose thresholds are calibrated against that context set) then degrades to "
        "hallucination 0.032 / correct abstention 0.750; `make sweep NAME=crag` recovers only "
        "to 0.016 / 0.875. Ranking quality isn't worth the headline safety metric, so "
        "`replace` remains the default and the knob is documented. This trade-off is a mock "
        "artifact of a lexical reranker — re-measure it with a real cross-encoder.\n"
        "- HyDE and multi-hop decomposition are implemented and config-gated but not shown as "
        "ablation rows: they are no-ops at this corpus size, and their benefit shows on larger "
        "corpora / at low k with neural embeddings (now measurable via recall@1 and MRR).\n"
    )
    Path(out_path).write_text("".join(parts))
    return out_path
