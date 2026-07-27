"""Generate a self-contained HTML evaluation dashboard from the eval artifacts.

Reads eval/results/{ablation,selective}.json + history.jsonl (+ live judge
calibration) and writes a single dashboard.html with the headline metrics, the
ablation table, per-slice breakdown, the risk–coverage and ablation plots
(embedded as base64 so the file is fully portable), and the run history.
No server, no CDN — open it in any browser.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from arag.common.config import load_config

RESULTS = Path("eval/results")


def _b64_img(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border:1px solid #eee;border-radius:8px"/>'


def _table(headers, rows) -> str:
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def build(cfg) -> str:
    ablation = json.loads((RESULTS / "ablation.json").read_text()) if (RESULTS / "ablation.json").exists() else []
    selective = json.loads((RESULTS / "selective.json").read_text()) if (RESULTS / "selective.json").exists() else {}

    base = ablation[0]["summary"] if ablation else {}
    final = ablation[-1]["summary"] if ablation else {}

    # Ablation table
    cols = [("label", "Configuration"), ("hallucination_rate", "Hallu↓"), ("faithfulness", "Faith↑"),
            ("recall_at_k", "Rec@k↑"), ("mrr", "MRR↑"), ("citation_precision", "CiteP↑"),
            ("correct_abstention_rate", "Abst.OK↑"), ("adversarial_robustness_rate", "AdvRobust↑"),
            ("over_abstention_rate", "OverAbst↓")]
    rows = []
    for run in ablation:
        s = run["summary"]
        row = [run.get("label", run.get("tag", ""))]
        for key, _ in cols[1:]:
            v = s.get(key, 0.0)
            row.append(f"{v:.3f}" if isinstance(v, int | float) else v)
        rows.append(row)
    ablation_table = _table([c[1] for c in cols], rows)

    # Per-slice
    by = final.get("by_slice", {})
    slice_rows = []
    for name in ("easy", "multi_hop", "unanswerable", "adversarial"):
        sl = by.get(name, {})
        slice_rows.append([name, sl.get("n", "-"), f"{sl.get('hallucination_rate', 0):.3f}",
                           f"{sl.get('faithfulness', 0):.3f}",
                           f"{sl.get('robustness_pass', '-')}"])
    slice_table = _table(["Slice", "n", "Hallu", "Faith", "Robust"], slice_rows)

    # Calibration (live)
    try:
        from eval.calibrate_judge import calibrate

        cal = calibrate(cfg)
        cal_html = (f"LLM-judge accuracy <b>{cal['llm_judge']['accuracy']:.2f}</b>, "
                    f"Cohen's κ <b>{cal['llm_judge']['cohens_kappa']:.2f}</b>")
    except Exception:
        cal_html = "n/a"

    # History
    hist_rows = []
    hpath = RESULTS / "history.jsonl"
    if hpath.exists():
        for line in hpath.read_text().splitlines()[-12:]:
            h = json.loads(line)
            m = h["metrics"]
            hist_rows.append([h["ts"], h["git_sha"], h["config_hash"], h["tag"],
                              f"{m.get('hallucination_rate')}", f"{m.get('faithfulness')}"])
    history_table = _table(["ts", "git", "cfg", "tag", "hallu", "faith"], hist_rows) if hist_rows else "<i>no runs</i>"

    def card(label, value):
        return f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>'

    cards = "".join([
        card("Hallucination (base→final)", f"{base.get('hallucination_rate',0)*100:.1f}% → {final.get('hallucination_rate',0)*100:.1f}%"),
        card("Correct abstention", f"{final.get('correct_abstention_rate',0)*100:.0f}%"),
        card("Adversarial robustness", f"{final.get('adversarial_robustness_rate',0)*100:.0f}%"),
        card("Faithfulness", f"{final.get('faithfulness',0):.2f}"),
        card("Risk–coverage AUC", f"{selective.get('risk_coverage_auc','-')}"),
        card("Max safe coverage", f"{selective.get('max_safe_coverage','-')}"),
    ])

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Agentic RAG — Eval Dashboard</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1116;color:#e6e6e6}}
 .wrap{{max-width:1000px;margin:0 auto;padding:28px}}
 h1{{font-size:22px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #2a2f3a;padding-bottom:6px}}
 .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
 .card{{background:#171a21;border:1px solid #2a2f3a;border-radius:10px;padding:14px 16px;min-width:150px}}
 .card .v{{font-size:20px;font-weight:700;color:#7ee0a2}} .card .l{{font-size:12px;color:#9aa4b2;margin-top:4px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{border:1px solid #2a2f3a;padding:6px 8px;text-align:center}} th{{background:#171a21}}
 td:first-child,th:first-child{{text-align:left}}
 .muted{{color:#9aa4b2;font-size:12px}}
</style></head><body><div class="wrap">
 <h1>Self-Correcting Agentic RAG — Evaluation Dashboard</h1>
 <div class="muted">Deterministic mock mode · regenerate with <code>make ablation &amp;&amp; make dashboard</code></div>
 <div class="cards">{cards}</div>
 <h2>Ablation (each row adds one component)</h2>{ablation_table}
 <h2>Per-slice breakdown</h2>{slice_table}
 <h2>Selective prediction</h2>
 <p class="muted">Risk–coverage of the abstention gate — risk stays 0 up to the answerable fraction.</p>
 {_b64_img(RESULTS / "risk_coverage.png")}
 <h2>Ablation trend</h2>{_b64_img(RESULTS / "ablation.png")}
 <h2>Judge calibration</h2><p>{cal_html}</p>
 <h2>Run history (experiment registry)</h2>{history_table}
</div></body></html>"""


def main() -> int:
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
    html = build(cfg)
    out = RESULTS / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
