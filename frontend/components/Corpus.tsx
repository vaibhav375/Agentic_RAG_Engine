"use client";

import { useEffect, useState } from "react";
import { API_URL } from "@/lib/api";

type Doc = {
  doc_id: string;
  title: string;
  chunks: number;
  sections: string[];
  /** A question from the graded benchmark that this document answers. */
  example?: string;
};

/**
 * What the engine actually knows.
 *
 * This pipeline declines anything outside its corpus, so without a table of
 * contents the correct behaviour reads as brokenness — a visitor asks about
 * something reasonable, gets refused, and concludes the demo is broken rather
 * than that the question was out of scope.
 *
 * Section headings are also the practical answer to "how do I phrase it". The
 * answerability gate scores how much of a question's distinctive vocabulary
 * appears in the retrieved text, so a question built from these words lands and
 * a vague one ("how does caching work") scores 0.53 and gets declined.
 */
export function Corpus({ onPick }: { onPick: (seed: string) => void }) {
  const [docs, setDocs] = useState<Doc[] | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/corpus`)
      .then((r) => r.json())
      .then((d) => setDocs(d.docs ?? []))
      .catch(() => setDocs([]));
  }, []);

  if (!docs?.length) return null;

  return (
    <details className="mt-4 rounded-sm border border-rule">
      <summary className="cursor-pointer list-none px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-ink2 hover:text-ink">
        What it knows — {docs.length} documents ▾
      </summary>

      <div className="border-t border-rule px-3 py-3">
        <p className="mb-3 text-[12px] leading-snug text-ink2">
          Anything outside this list gets declined, by design. Phrasing matters
          more than it should: the answerability gate scores how much of your
          wording appears in the sources, so &ldquo;how does caching work&rdquo;
          scores 0.53 and is refused while &ldquo;what does the cache decorator
          do&rdquo; scores 1.00. Each topic below offers a question from the graded
          benchmark that is known to work.
        </p>

        <ul className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
          {docs.map((d) => (
            <li key={d.doc_id}>
              <div className="font-mono text-[12px] text-ink">{d.title}</div>
              <div className="text-[11px] leading-snug text-ink3">
                {d.sections.join(" · ")}
              </div>
              {d.example && (
                <button
                  onClick={() => onPick(d.example!)}
                  className="mt-1 text-left text-[12px] leading-snug text-ink2 underline decoration-rule underline-offset-2 hover:text-grounded hover:decoration-grounded"
                >
                  {d.example}
                </button>
              )}
            </li>
          ))}
        </ul>

      </div>
    </details>
  );
}
