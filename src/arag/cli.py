"""Command-line interface: `arag ingest | query | serve | eval | validate-gold`."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel

from arag.common.config import load_config

app = typer.Typer(add_completion=False, help="Agentic RAG Engine CLI")
console = Console()


@app.command()
def ingest(config: str = typer.Option("config/config.yaml", "--config", "-c")):
    """Load the corpus, chunk, embed, and build dense + sparse indexes."""
    from arag.ingest.index import build_index

    cfg = load_config(config)
    store = build_index(cfg)
    console.print(
        Panel.fit(
            f"[green]Indexed[/green] {store.meta['n_chunks']} chunks "
            f"from {store.meta['n_docs']} docs (dim={store.meta['dim']}, "
            f"embeddings={store.meta['embeddings_provider']})",
            title="ingest",
        )
    )


@app.command()
def query(
    question: str,
    config: str = typer.Option("config/config.yaml", "--config", "-c"),
    show_trace: bool = typer.Option(False, "--trace"),
):
    """Answer a single question with the current config."""
    from arag.engine import answer_query, build_components

    cfg = load_config(config)
    comp = build_components(cfg)
    ans = answer_query(comp, question)

    body = ans.answer if not ans.abstained else f"[yellow](abstained)[/yellow] {ans.answer}"
    console.print(Panel(body, title=f"answer  (route={ans.route}, iters={ans.iterations}, cache={ans.from_cache})"))
    if ans.citations:
        console.print("[bold]Citations:[/bold]")
        for c in ans.citations:
            console.print(f"  • [{c.chunk_id}] {c.doc_id} :: {c.section or '-'}")
    if ans.critique:
        console.print(f"[dim]support_fraction={ans.critique.support_fraction}[/dim]")
    if show_trace:
        for s in ans.trace:
            console.print(f"  [dim]{s.stage:14s} {s.ms:8.2f} ms[/dim]")


@app.command()
def serve(
    config: str = typer.Option("config/config.yaml", "--config", "-c"),
    host: str | None = None,
    port: int | None = None,
):
    """Run the FastAPI server."""
    import uvicorn

    from arag.serve.api import create_app

    cfg = load_config(config)
    app_ = create_app(config)
    uvicorn.run(
        app_,
        host=host or cfg.get("serve.host", "0.0.0.0"),
        port=port or int(cfg.get("serve.port", 8000)),
    )


@app.command("validate-gold")
def validate_gold(config: str = typer.Option("config/config.yaml", "--config", "-c")):
    """Validate the gold Q&A set against the ingested corpus."""
    from eval.build_gold_set import validate_gold_set

    cfg = load_config(config)
    report = validate_gold_set(cfg)
    console.print(json.dumps(report, indent=2))
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command("calibrate-judge")
def calibrate_judge(config: str = typer.Option("config/config.yaml", "--config", "-c")):
    """Measure the critic/judge against a human-labeled set (accuracy + kappa)."""
    from eval.calibrate_judge import calibrate

    cfg = load_config(config)
    report = calibrate(cfg)
    console.print(json.dumps({k: v for k, v in report.items() if k != "llm_disagreements"}, indent=2))


@app.command("selective")
def selective(config: str = typer.Option("config/config.yaml", "--config", "-c")):
    """Risk–coverage analysis of the abstention gate (selective prediction)."""
    from eval.selective import run_selective

    cfg = load_config(config)
    s = run_selective(cfg)
    console.print(json.dumps({k: v for k, v in s.items() if k != "points"}, indent=2))


@app.command("dashboard")
def dashboard(config: str = typer.Option("config/config.yaml", "--config", "-c")):
    """Generate a self-contained HTML evaluation dashboard."""
    from eval.dashboard import build

    cfg = load_config(config)
    from pathlib import Path

    out = Path("eval/results/dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(cfg))
    console.print(f"Wrote {out}")


@app.command("history")
def history(compare: str = typer.Option("", "--compare", help="Two run tags: 'tagA,tagB'")):
    """Show recent eval runs (experiment registry), or diff two by tag."""
    from eval.registry import compare_runs, load_history

    hist = load_history()
    if not hist:
        console.print("No runs recorded yet. Run `make eval` or `make ablation`.")
        return
    if compare:
        a_tag, b_tag = (compare.split(",") + [""])[:2]
        by_tag = {h["tag"]: h for h in hist}
        if a_tag in by_tag and b_tag in by_tag:
            console.print(json.dumps(compare_runs(by_tag[a_tag], by_tag[b_tag]), indent=2))
        else:
            console.print("Unknown tag(s).")
        return
    for h in hist[-10:]:
        m = h["metrics"]
        console.print(
            f"{h['ts']}  [cyan]{h['git_sha']}[/cyan] cfg:{h['config_hash']}  "
            f"[bold]{h['tag']}[/bold]  hallu={m.get('hallucination_rate')} "
            f"faith={m.get('faithfulness')} advRobust={m.get('adversarial_robustness_rate')}"
        )


@app.command("eval")
def eval_cmd(
    config: str = typer.Option("config/config.yaml", "--config", "-c"),
    subset: int = typer.Option(0, "--subset", help="Evaluate only first N questions (0 = all)"),
    tag: str = typer.Option("current", "--tag"),
):
    """Run the eval harness over the gold set for the current config."""
    from eval.run_eval import run_eval

    cfg = load_config(config)
    if subset:
        cfg = cfg.with_overrides({"eval.subset": subset})
    result = run_eval(cfg, tag=tag)
    console.print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    app()
