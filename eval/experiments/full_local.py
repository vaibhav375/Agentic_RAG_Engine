"""Definitive local numbers on the full 109-question gold set.

Uses the measured-best config: qwen2.5:3b (same quality as 7b at 3x the speed),
critic: nli, clause claim extraction, hybrid + rerank(replace) + CRAG.

The previous full-set run predates the claim-extraction default and the citation
fixes, so its numbers are pessimistic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from eval.run_eval import run_eval  # noqa: E402

MODEL = "qwen2.5:3b"
OUT = Path("eval/results/full_local_progress.json")


def main() -> int:
    import httpx

    if OUT.exists():
        print("already complete:", OUT.read_text()[:200])
        return 0
    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})
    cfg = load_config("config/config.yaml").with_overrides({
        "mode": "local", "embeddings.provider": "sentence_transformers",
        "llm.provider": "ollama", "llm.ollama_model": MODEL,
        "vector_store.persist_dir": ".arag_index_local",
        "eval.subset": None,
        "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
        "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
    })
    t0 = time.time()
    d = run_eval(cfg, tag="full_local_qwen3b")
    s = d["summary"]
    abst = sum(1 for r in d["detail"] if r["abstained"])
    answered = len(d["detail"]) - abst
    flagged = sum(1 for r in d["detail"] if r["metrics"].get("hallucination", 0) > 0)
    keys = ["n", "hallucination_rate", "hallucination_rate_ci95",
            "hallucination_rate_strict", "faithfulness", "citation_precision",
            "answer_correctness", "over_abstention_rate", "correct_abstention_rate",
            "adversarial_robustness_rate", "recall_at_1", "recall_at_k", "mrr",
            "latency_p50_ms"]
    out = {k: s.get(k) for k in keys}
    out.update({"answered": answered, "flagged": flagged,
                "by_slice": s.get("by_slice"),
                "wall_min": round((time.time() - t0) / 60, 1)})
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
