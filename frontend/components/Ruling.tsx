"use client";

import type { QueryResponse } from "@/lib/api";

/**
 * The answer, presented as a ruling with its citations attached.
 *
 * Citation brackets are interactive: hovering one illuminates the source chunk in
 * the evidence column. That link is the pipeline's actual claim — this sentence
 * came from that passage — and it is the one thing a reader cannot verify from a
 * chat transcript.
 *
 * When the engine declined, this panel shows the reason rather than an empty box.
 */
export function Ruling({
  data,
  onHoverCitation,
  active,
}: {
  data: QueryResponse;
  onHoverCitation: (chunkId: string | null) => void;
  active: string | null;
}) {
  const cited = data.citations ?? [];

  return (
    <section className="rounded-sm border border-rule bg-card p-5 sm:p-6">
      {data.abstained ? (
        <>
          <p className="text-[15px] leading-relaxed text-ink2">{data.answer}</p>
          <p className="mt-3 border-l-2 border-declined pl-3 text-[13px] leading-snug text-ink2">
            This is the intended behaviour. On the 117-question benchmark the engine
            declines every one of the 12 out-of-scope questions rather than
            fabricating an answer.
          </p>
        </>
      ) : (
        <p className="whitespace-pre-wrap text-[17px] leading-[1.6] text-ink">
          {data.answer}
        </p>
      )}

      {cited.length > 0 && (
        <div className="mt-5 border-t border-rule pt-4">
          <h3 className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink3">
            Grounded in
          </h3>
          <ul className="mt-2 flex flex-wrap gap-2">
            {cited.map((c, i) => (
              <li key={`${c.chunk_id}-${i}`}>
                <button
                  onMouseEnter={() => onHoverCitation(c.chunk_id)}
                  onMouseLeave={() => onHoverCitation(null)}
                  onFocus={() => onHoverCitation(c.chunk_id)}
                  onBlur={() => onHoverCitation(null)}
                  className={`font-mono text-[12px] rounded-sm border px-2 py-1 transition-colors ${
                    active === c.chunk_id
                      ? "border-grounded bg-grounded text-card"
                      : "border-rule bg-paper text-ink2 hover:border-grounded hover:text-grounded"
                  }`}
                >
                  [{c.chunk_id}]
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {data.reformulations?.length > 0 && (
        <div className="mt-5 border-t border-rule pt-4">
          <h3 className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink3">
            It rewrote the question {data.reformulations.length}×
          </h3>
          <p className="mt-1 text-[13px] leading-snug text-ink2">
            The first answer wasn&rsquo;t fully supported, so the engine broadened the
            query and searched again.
          </p>
          <ol className="mt-2 space-y-1">
            {data.reformulations.map((q, i) => (
              <li key={i} className="font-mono text-[12px] leading-snug text-ink2">
                <span className="text-ink3">{i + 1} →</span> {q}
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
