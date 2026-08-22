"use client";

import type { Ctx } from "@/lib/api";

/**
 * What the generator was actually shown, in rank order.
 *
 * A chunk lights up when its citation is hovered in the ruling, which is the
 * point of showing it at all: you can see that the sentence came from this
 * passage and not from the model's memory. Retrieval rank is kept visible because
 * a cited chunk sitting at rank 5 is a different story from one at rank 1.
 */
export function Evidence({
  contexts,
  active,
  citedIds,
}: {
  contexts: Ctx[];
  active: string | null;
  citedIds: Set<string>;
}) {
  if (!contexts?.length) return null;

  return (
    <section className="mt-6">
      <h2 className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink3">
        Sources shown to the generator ({contexts.length})
      </h2>

      <ul className="mt-3 space-y-2">
        {contexts.map((c, i) => {
          const isActive = active === c.chunk_id;
          const wasCited = citedIds.has(c.chunk_id);
          return (
            <li
              key={c.chunk_id}
              className={`rounded-sm border p-3 transition-colors ${
                isActive
                  ? "border-grounded bg-card ring-1 ring-grounded"
                  : wasCited
                    ? "border-rule bg-card"
                    : "border-rule/60 bg-transparent"
              }`}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-mono text-[12px] text-ink">
                  <span className="text-ink3">{String(i + 1).padStart(2, "0")}</span>{" "}
                  {c.chunk_id}
                </span>
                <span className="font-mono text-[11px] tabular-nums text-ink3">
                  {wasCited && (
                    <span className="mr-2 text-grounded">cited</span>
                  )}
                  {c.score?.toFixed(3)}
                </span>
              </div>
              {c.section && (
                <div className="font-mono text-[11px] text-ink3">{c.section}</div>
              )}
              <p className="mt-1 text-[13px] leading-snug text-ink2">
                {c.text?.slice(0, 260)}
                {c.text && c.text.length > 260 ? "…" : ""}
              </p>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
