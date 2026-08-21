"""Can an LLM judge do the two semantic jobs NLI and token-F1 cannot?

Two defects share one candidate fix, so they are validated before anything is
built — the pattern that has killed four hypotheses cheaply on this project
already (stronger NLI, crag.mode: llm, GraphRAG, the reranker as a CRAG signal).

PART A — answer equivalence. `answer_correctness` is token-F1 and it inverted on
a hand-check: the one wrong answer scored highest (0.632, matching gold's wording
and swapping only CORSMiddleware for GZipMiddleware) and a correct answer scored
0.000 ("Routes are not authenticated by default" vs "No; every route is public").
NLI bidirectional entailment scored ~0.00 for everything; embedding cosine put the
wrong answer above four correct ones. A judge is the remaining option.

PART B — claim support. Nine of the 14 over-abstentions are the critic refusing
answers whose context was correct; 6 of 7 recovered answers were right. NLI is
weak on negation and paraphrase, which is exactly the shape of these. If the judge
marks them supported *without* also blessing the genuinely unsupported ones, an
LLM fallback for unentailed claims fixes over-abstention without the 0.205
hallucination that disabling the critic costs.

Sensitivity alone is worthless here: a judge that says "supported" to everything
recovers all nine and destroys the safety guarantee. So both parts measure against
known-wrong cases too — m06, e14 and e57 are hand-verified wrong answers, and e14
in particular is a subtle inversion ("ensures get_db runs only once per request"
where the corpus says it forces it to run every time).

Ground truth is hand-verified and recorded in docs/local-mode-eval.md; n is small
(9 labels for part A) and stated rather than smoothed over.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from arag.providers.llm import make_llm  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
OUT = Path(f"eval/results/judge_validation_{MODEL.replace(':','_')}.json")

# The shipped prompt, imported rather than copied: a second copy of it would
# drift from the one actually used, and then this validation would be measuring
# something the pipeline does not run.
from arag.generate.prompts import EQUIVALENCE_SYSTEM, EQUIVALENCE_USER  # noqa: E402

# Hand-verified labels. The six correct ones were recovered from the pipeline's
# own discarded answers and checked against gold one by one; the three wrong ones
# are documented failures.
LABELS = {
    # Correct answers, hand-checked against gold one by one.
    "e26": True, "e39": True, "m04": True, "e50": True, "e70": True, "m10": True,
    # Correct-but-elaborated. Added after the first prompt rejected all three:
    # each contains the gold answer verbatim and then adds accurate detail, and a
    # judge that penalises elaboration understates quality across the whole set.
    # The original 9 labels were mostly terse, which is why they missed this.
    "e10": True, "e18": True, "e22": True,
    # Wrong answers. m06 is the case token-F1 scored highest of all.
    "m06": False,   # names CORSMiddleware where gold says GZipMiddleware
    "e14": False,   # inverts use_cache=False
    "e57": False,   # unsupported claim about a generic 500 response
}


def _answers() -> dict[str, dict]:
    """Predicted answers for the labelled ids, from the runs that produced them."""
    out: dict[str, dict] = {}
    for path in ("eval/results/oa_no_post_abstain.json", "eval/results/crag_t051_shipped.json",
                 "eval/results/full_local_hard.json"):
        for r in json.loads(Path(path).read_text())["detail"]:
            if r["id"] in LABELS and r["id"] not in out and (r["predicted"] or "").strip():
                out[r["id"]] = r
    return out


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})

    cfg = load_config("config/config.yaml").with_overrides(
        {"mode": "local", "llm.provider": "ollama", "llm.ollama_model": MODEL}
    )
    llm = make_llm(cfg)
    rows = _answers()
    missing = set(LABELS) - set(rows)
    if missing:
        print(f"warning: no saved answer for {sorted(missing)} — excluded", flush=True)

    results = []
    for qid, truth in LABELS.items():
        if qid not in rows:
            continue
        r = rows[qid]
        raw = llm._complete(
            EQUIVALENCE_SYSTEM,
            EQUIVALENCE_USER.format(
                question=r["question"], gold=r["gold_answer"], candidate=r["predicted"]
            ),
        )
        try:
            verdict = bool(json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["equivalent"])
            parsed = True
        except Exception:
            verdict, parsed = False, False
        results.append({
            "id": qid, "truth": truth, "judge": verdict, "parsed": parsed,
            "token_f1": r["metrics"].get("answer_correctness"),
            "raw": raw.strip()[:200],
        })
        print(f"  {qid}: truth={truth!s:5s} judge={verdict!s:5s} "
              f"{'OK ' if verdict == truth else 'MISS'} tokenF1={r['metrics'].get('answer_correctness')}",
              flush=True)

    ok = sum(1 for x in results if x["judge"] == x["truth"])
    tp = sum(1 for x in results if x["truth"] and x["judge"])
    fp = sum(1 for x in results if not x["truth"] and x["judge"])
    n_pos = sum(1 for x in results if x["truth"])
    n_neg = len(results) - n_pos
    unparsed = sum(1 for x in results if not x["parsed"])

    print(f"\nagreement with hand labels: {ok}/{len(results)}", flush=True)
    print(f"  correct answers accepted : {tp}/{n_pos}", flush=True)
    print(f"  WRONG answers accepted   : {fp}/{n_neg}   <- must be 0 to be usable", flush=True)
    if unparsed:
        print(f"  unparseable judge replies: {unparsed}", flush=True)
    print("\ntoken-F1 for comparison ranked these close to backwards; the bar here is"
          "\nnot 'better than token-F1' but 'never accepts a wrong answer'.", flush=True)

    OUT.write_text(json.dumps(
        {"model": MODEL, "n": len(results), "agreement": ok,
         "correct_accepted": f"{tp}/{n_pos}", "wrong_accepted": f"{fp}/{n_neg}",
         "unparsed": unparsed, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
