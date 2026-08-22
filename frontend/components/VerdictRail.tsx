"use client";

import type { Stage } from "@/lib/api";

/**
 * The page's signature: a ledger that fills as the pipeline runs and ends in a
 * ruling. Stages stamp in from the SSE stream, so on a slow local model you watch
 * the engine work rather than staring at a blank panel.
 *
 * Abstention is rendered as a considered outcome, not an error. That is the whole
 * thesis of this pipeline — declining to answer is the feature — so DECLINED gets
 * the same typographic weight as ANSWERED, in amber rather than red.
 */

const LABELS: Record<string, string> = {
  guardrail: "screen input",
  retrieve: "retrieve",
  rerank: "rerank",
  crag_grade: "grade context",
  generate: "generate",
  critique: "critique",
  reformulate: "reformulate",
  cache_lookup: "cache",
};

function metaLine(s: Stage): string {
  const m = s.meta || {};
  const bits: string[] = [];
  for (const k of ["contexts", "candidates", "chunks", "claims"]) {
    if (m[k] !== undefined && m[k] !== null) bits.push(`${k} ${m[k]}`);
  }
  // The loop counts iterations from zero internally; a reader counts passes from
  // one, and "iteration 0" reads like a bug.
  if (typeof m.iteration === "number") bits.push(`pass ${m.iteration + 1}`);
  return bits.join(" · ");
}

export function VerdictRail({
  stages,
  running,
  verdict,
}: {
  stages: Stage[];
  running: boolean;
  verdict: null | {
    abstained: boolean;
    grade: string | null;
    gradeScore: number | null;
    support: number | null;
    iterations: number;
    flags: string[];
  };
}) {
  return (
    <aside className="lg:sticky lg:top-6 self-start">
      <h2 className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink3">
        Pipeline
      </h2>

      <ol className="mt-3 border-l border-rule">
        {stages.length === 0 && !running && (
          <li className="pl-4 py-2 text-sm text-ink3">Ask something to start.</li>
        )}

        {stages.map((s, i) => (
          <li key={`${s.stage}-${i}`} className="stamp relative pl-4 py-[7px]">
            <span
              aria-hidden
              className="absolute -left-[3px] top-[15px] h-[5px] w-[5px] rounded-full bg-ink2"
            />
            <div className="flex items-baseline justify-between gap-3">
              <span className="font-mono text-[13px] tracking-tight text-ink">
                {LABELS[s.stage] ?? s.stage}
              </span>
              <span className="font-mono text-[11px] tabular-nums text-ink3">
                {s.ms < 1 ? "<1" : Math.round(s.ms)} ms
              </span>
            </div>
            {metaLine(s) && (
              <div className="font-mono text-[11px] text-ink3">{metaLine(s)}</div>
            )}
          </li>
        ))}

        {running && (
          <li className="relative pl-4 py-[7px]">
            <span className="running block h-[2px] w-24 rounded" />
          </li>
        )}
      </ol>

      {verdict && (
        <div className="stamp mt-5 border-t border-rule pt-4">
          <div
            className={`font-mono text-[26px] leading-none tracking-tightest ${
              verdict.abstained ? "text-declined" : "text-grounded"
            }`}
          >
            {verdict.abstained ? "DECLINED" : "ANSWERED"}
          </div>
          <p className="mt-2 text-[13px] leading-snug text-ink2">
            {verdict.abstained
              ? "The engine could not ground an answer in the retrieved sources, so it refused rather than guessing."
              : "Every claim below was checked against the retrieved sources before this answer was released."}
          </p>

          <dl className="mt-3 space-y-1 font-mono text-[11px] text-ink2">
            {verdict.grade && (
              <div className="flex justify-between gap-2">
                <dt className="text-ink3">answerability</dt>
                <dd className="tabular-nums">
                  {verdict.grade}
                  {verdict.gradeScore != null && ` ${verdict.gradeScore.toFixed(2)}`}
                </dd>
              </div>
            )}
            {verdict.support != null && (
              <div className="flex justify-between gap-2">
                <dt className="text-ink3">claims supported</dt>
                <dd className="tabular-nums">{(verdict.support * 100).toFixed(0)}%</dd>
              </div>
            )}
            <div className="flex justify-between gap-2">
              <dt className="text-ink3">passes</dt>
              <dd className="tabular-nums">{verdict.iterations}</dd>
            </div>
            {verdict.flags.length > 0 && (
              <div className="flex justify-between gap-2">
                <dt className="text-ink3">guardrail</dt>
                <dd className="text-contra">{verdict.flags.join(", ")}</dd>
              </div>
            )}
          </dl>
        </div>
      )}
    </aside>
  );
}
