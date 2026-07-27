"use client";

import { Card, CardBody, CardHeader, Badge } from "@/components/ui/primitives";
import type { Ctx } from "@/lib/api";

export function ContextsPanel({ contexts }: { contexts: Ctx[] }) {
  return (
    <Card>
      <CardHeader>Retrieved context · {contexts.length} passages</CardHeader>
      <CardBody className="space-y-3">
        {contexts.length === 0 && <div className="text-sm text-muted">No context yet.</div>}
        {contexts.map((c, i) => (
          <div key={c.chunk_id} className="rounded-lg border border-edge bg-bg p-3">
            <div className="mb-1 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-accent">#{i + 1}</span>
              <span className="text-muted">{c.chunk_id}</span>
              <Badge>{c.source}</Badge>
              <span className="text-muted">score {c.score.toFixed(4)}</span>
              {Object.entries(c.ranks).map(([k, v]) => (
                <span key={k} className="text-muted">
                  {k}:{v}
                </span>
              ))}
            </div>
            <div className="line-clamp-3 text-sm text-muted">{c.text}</div>
          </div>
        ))}
      </CardBody>
    </Card>
  );
}
