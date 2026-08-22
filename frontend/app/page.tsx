"use client";

import { useEffect, useMemo, useState } from "react";
import { AskBar } from "@/components/AskBar";
import { Corpus } from "@/components/Corpus";
import { Evidence } from "@/components/Evidence";
import { Ruling } from "@/components/Ruling";
import { VerdictRail } from "@/components/VerdictRail";
import { API_URL, type Flags, type QueryResponse, type Stage, streamQuery } from "@/lib/api";

/**
 * The pipeline's defining behaviour is refusing to answer when it cannot ground a
 * claim, so the page is built around the decision rather than around a chat log:
 * a live ledger of what the engine did, the ruling it reached, and the passages it
 * is standing on.
 *
 * Every feature is on by default. The previous demo shipped with the config's
 * defaults, which have hybrid search, the answerability gate and self-correction
 * switched off — so the first thing a visitor saw was the naive baseline this
 * project exists to improve on.
 */
const DEFAULT_FLAGS: Flags = {
  useHybrid: true,
  useRerank: true,
  useHyde: false,
  useMmr: false,
  useDecompose: false,
  agent: true,
  crag: true,
  router: false,
  cache: false,
};

type Health = {
  mode: string;
  index?: { n_chunks?: number; n_docs?: number; embeddings_provider?: string };
};

export default function Page() {
  const [q, setQ] = useState("How do you declare a path parameter in Breeze?");
  const [flags, setFlags] = useState<Flags>(DEFAULT_FLAGS);
  const [stages, setStages] = useState<Stage[]>([]);
  const [data, setData] = useState<QueryResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function ask(override?: string) {
    const question = (override ?? q).trim();
    if (!question) return;
    setQ(question);
    setBusy(true);
    setError(null);
    setStages([]);
    setData(null);
    setHovered(null);
    try {
      await streamQuery(question, flags, (s) => setStages((prev) => [...prev, s]), setData);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const citedIds = useMemo(
    () => new Set((data?.citations ?? []).map((c) => c.chunk_id)),
    [data],
  );

  return (
    <main className="mx-auto max-w-6xl px-5 py-8 sm:py-12">
      <header className="border-b border-rule pb-5">
        <h1 className="font-mono text-[15px] tracking-tightest text-ink">
          Self-correcting RAG
        </h1>
        <p className="mt-1 max-w-xl text-[13px] leading-snug text-ink2">
          Ask the Breeze documentation something. Watch the engine retrieve, judge
          whether the passages can answer you, write a grounded answer — and refuse
          when they can&rsquo;t.
        </p>
        {health && (
          <p className="mt-2 font-mono text-[11px] text-ink3">
            {health.mode} mode · {health.index?.n_chunks ?? "?"} chunks from{" "}
            {health.index?.n_docs ?? "?"} documents · {health.index?.embeddings_provider}
          </p>
        )}
        {health === null && (
          <p className="mt-2 font-mono text-[11px] text-contra">
            Backend unreachable at {API_URL} — start it with{" "}
            <code>make serve</code>.
          </p>
        )}
      </header>

      <div className="mt-6 grid gap-8 lg:grid-cols-[220px_1fr]">
        <VerdictRail
          stages={stages}
          running={busy}
          verdict={
            data
              ? {
                  abstained: data.abstained,
                  grade: data.retrieval_grade,
                  gradeScore: data.retrieval_grade_score,
                  support: data.support_fraction,
                  iterations: data.iterations,
                  flags: data.input_flags ?? [],
                }
              : null
          }
        />

        <div>
          <AskBar
            value={q}
            onChange={setQ}
            onAsk={ask}
            busy={busy}
            flags={flags}
            onFlags={setFlags}
          />

          {/* Ask the benchmark question verbatim. An earlier version wrapped the
              topic in "What does the documentation say about X?", which the gate
              declined every time — "documentation" and "say" are rare words absent
              from the corpus, so they drag lexical coverage down. */}
          <Corpus onPick={(question) => ask(question)} />

          {error && (
            <p className="mt-5 rounded-sm border border-contra bg-card p-3 font-mono text-[12px] text-contra">
              {error}
            </p>
          )}

          {data && (
            <div className="mt-6">
              <Ruling data={data} onHoverCitation={setHovered} active={hovered} />
              <Evidence
                contexts={data.contexts ?? []}
                active={hovered}
                citedIds={citedIds}
              />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
