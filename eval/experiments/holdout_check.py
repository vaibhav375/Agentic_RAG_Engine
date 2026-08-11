"""How much of the reported performance is in-sample?

Thresholds, prompts, the CRAG gate and the claim splitter were all tuned while
looking at the whole gold set. The dev/holdout gap is the honest estimate of how
much that inflated the numbers.

SUPERSEDED by `make eval`, which now reports dev against holdout by default and
does it in one pass (the splits partition the gold set, so both halves share a
single index build instead of paying for two). Prefer that; this script is kept
because `local_dev.json` / `local_holdout.json` are cited in
docs/local-mode-eval.md and rerunning it must keep reproducing those files.

Do not add metrics here — add them to `run_eval_split_report`. Two
implementations of the same number drift, and then neither can be trusted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from eval.run_eval import run_eval  # noqa: E402

OUT = Path("eval/results/holdout_progress.json")
MODEL = "qwen2.5:3b"
KEYS = ["n", "hallucination_rate", "hallucination_rate_ci95", "faithfulness",
        "citation_precision", "answer_correctness", "over_abstention_rate",
        "correct_abstention_rate", "adversarial_robustness_rate", "recall_at_1", "mrr"]


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    cfg = load_config("config/config.yaml")
    for split in ["holdout", "dev"]:          # holdout first: the number that matters
        if split in done:
            print(f"skip (done): {split}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> {split}", flush=True)
        s = run_eval(cfg.with_overrides({
            "mode": "local", "embeddings.provider": "sentence_transformers",
            "llm.provider": "ollama", "llm.ollama_model": MODEL,
            "vector_store.persist_dir": ".arag_index_local",
            "eval.split": split,
            "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
            "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
        }), tag=f"local_{split}")["summary"]
        done[split] = {k: s.get(k) for k in KEYS}
        done[split]["wall_min"] = round((time.time() - t0) / 60, 1)
        OUT.write_text(json.dumps(done, indent=2))
        print(json.dumps(done[split], indent=2), flush=True)

    if len(done) == 2:
        print("\n=== in-sample vs out-of-sample ===", flush=True)
        cols = ["hallucination_rate", "faithfulness", "citation_precision",
                "answer_correctness", "correct_abstention_rate",
                "adversarial_robustness_rate", "recall_at_1"]
        print(f"{'split':9s} " + " ".join(f"{c[:13]:>14}" for c in cols), flush=True)
        for k in ("dev", "holdout"):
            print(f"{k:9s} " + " ".join(f"{done[k].get(c, 0):>14.3f}" for c in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
