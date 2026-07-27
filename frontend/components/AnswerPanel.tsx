"use client";

import { Badge, Card, CardBody, CardHeader } from "@/components/ui/primitives";
import { API_URL, type QueryResponse } from "@/lib/api";
import { useState } from "react";

function gradeTone(g: string | null) {
  if (g === "correct") return "good";
  if (g === "ambiguous") return "warn";
  if (g === "incorrect") return "bad";
  return "default";
}

export function AnswerPanel({ res }: { res: QueryResponse }) {
  const [openChunk, setOpenChunk] = useState<string | null>(null);
  const [chunkText, setChunkText] = useState<string>("");

  async function showChunk(id: string) {
    if (openChunk === id) {
      setOpenChunk(null);
      return;
    }
    setOpenChunk(id);
    const r = await fetch(`${API_URL}/chunk/${encodeURIComponent(id)}`);
    const d = await r.json();
    setChunkText(d.text || "");
  }

  return (
    <Card>
      <CardHeader className="flex flex-wrap items-center gap-2">
        <span>Answer</span>
        <div className="ml-auto flex flex-wrap gap-1.5">
          {res.abstained ? <Badge tone="warn">abstained</Badge> : <Badge tone="good">answered</Badge>}
          <Badge tone="info">route: {res.route}</Badge>
          <Badge>iters: {res.iterations}</Badge>
          {res.retrieval_grade && (
            <Badge tone={gradeTone(res.retrieval_grade) as any}>
              CRAG: {res.retrieval_grade}
              {res.retrieval_grade_score != null && ` (${res.retrieval_grade_score.toFixed(2)})`}
            </Badge>
          )}
          {res.from_cache && <Badge tone="info">cache hit</Badge>}
          {res.support_fraction != null && <Badge>support: {res.support_fraction.toFixed(2)}</Badge>}
        </div>
      </CardHeader>
      <CardBody className="space-y-4">
        {res.input_flags.length > 0 && (
          <div className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
            ⚠ Guardrail: input flagged as {res.input_flags.join(", ")}
          </div>
        )}

        <p className={res.abstained ? "text-warn" : "text-[15px] leading-relaxed"}>{res.answer}</p>

        {res.citations.length > 0 && (
          <div>
            <div className="mb-1 text-xs uppercase tracking-wide text-muted">Citations</div>
            <div className="flex flex-wrap gap-2">
              {res.citations.map((c) => (
                <button
                  key={c.chunk_id}
                  onClick={() => showChunk(c.chunk_id)}
                  className="rounded-md border border-edge px-2 py-1 text-xs hover:bg-edge/40"
                >
                  [{c.chunk_id}] {c.doc_id}
                  {c.section ? ` · ${c.section}` : ""}
                </button>
              ))}
            </div>
            {openChunk && (
              <div className="mt-2 rounded-lg border border-edge bg-bg p-3 text-sm text-muted">
                <div className="mb-1 text-xs text-accent">{openChunk}</div>
                {chunkText}
              </div>
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-4 border-t border-edge pt-3 text-xs text-muted">
          <span>cost: ${res.cost_usd.toFixed(6)}</span>
          <span>tokens: {res.tokens?.in ?? 0} in / {res.tokens?.out ?? 0} out</span>
          <span>contexts: {res.contexts.length}</span>
        </div>
      </CardBody>
    </Card>
  );
}
