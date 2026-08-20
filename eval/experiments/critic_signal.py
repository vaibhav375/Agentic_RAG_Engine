"""Can an LLM critic rescue the answers the NLI critic wrongly refuses?

Nine of the 14 over-abstentions are the post-generation critic never accepting an
answer whose retrieved context was correct — and 6 of 7 such answers, recovered
and hand-checked, were right. NLI is weak exactly where these fail: negation and
paraphrase ("Routes are not authenticated by default" against a premise saying
"every route is public").

Disabling the critic is not the fix — measured, it costs hallucination 0.205
against 0.046 for disabling CRAG, because the critic is the load-bearing safety
gate. So the question is whether a *different signal* accepts the correct answers
without also blessing the unsupported ones.

ONE MODEL PER RUN, deliberately. Setting llm.judge_model to 7b while generating
on 3b would leave two models resident and make Ollama swap per call — the exact
thrashing that once produced a phantom "7b costs 11x more" conclusion. Both arms
therefore run entirely on one model and vary only `agent.critic`, so the critic
signal is the single variable. If the 3B judge proves too weak, the follow-up is
an all-7b run, not a mixed one.

Arms:
  nli   the shipped critic (baseline, re-measured under current code)
  llm   the LLM judge deciding claim support

Success is not "recovers the nine". A critic that accepts everything recovers all
nine and destroys the safety guarantee, so the refusal slices ride along in both
arms and `hallucination_rate` is read next to the recovery count.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.experiments._harness import run_arm  # noqa: E402

OUT = Path("eval/results/critic_signal_progress.json")
SOURCE = Path("eval/results/full_local_hard.json")
MODEL = "qwen2.5:3b"

BASE = {
    "mode": "local", "embeddings.provider": "sentence_transformers",
    "llm.provider": "ollama", "llm.ollama_model": MODEL,
    "vector_store.persist_dir": ".arag_index_local",
    "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
    "agent.enabled": True, "agent.crag.enabled": True,
}
ARMS = [("nli", "nli"), ("llm", "llm")]
KEYS = ["n", "hallucination_rate", "faithfulness", "citation_precision",
        "answer_correctness", "answer_equivalence", "over_abstention_rate",
        "correct_abstention_rate", "adversarial_robustness_rate", "latency_p50_ms"]


def _targets() -> tuple[list[str], dict[str, str]]:
    d = json.loads(SOURCE.read_text())
    over = [r for r in d["detail"] if r["metrics"].get("over_abstention")]
    cause = {
        r["id"]: ("crag_pre_generation" if r["iterations"] == 1 else "critic_exhausted")
        for r in over
    }
    refusal = [r["id"] for r in d["detail"] if r["difficulty"] in ("unanswerable", "adversarial")]
    return sorted({*cause, *refusal}), cause


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})

    ids, cause = _targets()
    n_critic = sum(1 for v in cause.values() if v == "critic_exhausted")
    print(f"{len(ids)} questions/arm: {n_critic} critic-exhausted, "
          f"{len(cause) - n_critic} CRAG-declined, {len(ids) - len(cause)} refusal cases",
          flush=True)

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, critic in ARMS:
        if name in done:
            print(f"skip (done): {name}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> critic={critic}", flush=True)
        d = run_arm({**BASE, "agent.critic": critic, "eval.only_ids": ids},
                    tag=f"critic_{name}")
        rows = {r["id"]: r for r in d["detail"]}
        s = d["summary"]

        # Did the critic-exhausted cases come back, and were they actually right?
        # answer_equivalence is the semantic grade; token-F1 is kept beside it
        # because it is the series every earlier number sits on.
        crit_ids = [i for i, c in cause.items() if c == "critic_exhausted" and i in rows]
        recovered = [i for i in crit_ids if not rows[i]["abstained"]]
        refusal_ids = [i for i in rows if i not in cause]
        done[name] = {
            "critic": critic,
            **{k: s.get(k) for k in KEYS},
            "critic_exhausted_recovered": f"{len(recovered)}/{len(crit_ids)}",
            "recovered_equivalent": sum(
                1 for i in recovered if rows[i]["metrics"].get("answer_equivalence")
            ),
            "refusals_kept": sum(1 for i in refusal_ids if rows[i]["abstained"]),
            "refusals_total": len(refusal_ids),
            "refusals_hallucinated": sum(
                1 for i in refusal_ids if rows[i]["metrics"].get("hallucination")
            ),
            "wall_min": round((time.time() - t0) / 60, 1),
        }
        OUT.write_text(json.dumps(done, indent=2))
        print(json.dumps(done[name], indent=2), flush=True)

    if len(done) == len(ARMS):
        print("\n=== critic signal: recovery vs safety ===", flush=True)
        cols = ["critic_exhausted_recovered", "recovered_equivalent", "hallucination_rate",
                "correct_abstention_rate", "adversarial_robustness_rate",
                "over_abstention_rate", "refusals_hallucinated", "wall_min"]
        print(f"{'critic':8s} " + " ".join(f"{c[:20]:>22}" for c in cols), flush=True)
        for name, _ in ARMS:
            r = done[name]
            print(f"{name:8s} " + " ".join(f"{r.get(c)!s:>22}" for c in cols), flush=True)
        print("\nA critic that accepts everything recovers all nine and destroys the"
              "\nguarantee — read the recovery count next to hallucination_rate.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
