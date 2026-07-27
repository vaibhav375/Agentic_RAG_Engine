"use client";

import { Card, CardBody, CardHeader } from "@/components/ui/primitives";
import type { Stage } from "@/lib/api";

const STAGE_COLORS: Record<string, string> = {
  cache_lookup: "#6b7280",
  route: "#8e44ad",
  retrieve: "#3b82f6",
  rerank: "#0ea5e9",
  mmr: "#06b6d4",
  crag_grade: "#e5c07b",
  generate: "#7ee0a2",
  critique: "#e06c75",
  reformulate: "#f59e0b",
};

export function TraceTimeline({ stages }: { stages: Stage[] }) {
  const total = stages.reduce((a, s) => a + s.ms, 0) || 1;
  return (
    <Card>
      <CardHeader>Pipeline trace · {total.toFixed(1)} ms</CardHeader>
      <CardBody className="space-y-2">
        {stages.length === 0 && <div className="text-sm text-muted">Run a query to see the trace.</div>}
        {stages.map((s, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="w-28 shrink-0 text-xs text-muted">{s.stage}</div>
            <div className="h-3 flex-1 overflow-hidden rounded bg-edge/40">
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.max(4, (s.ms / total) * 100)}%`,
                  background: STAGE_COLORS[s.stage] || "#7ee0a2",
                }}
              />
            </div>
            <div className="w-16 shrink-0 text-right text-xs tabular-nums text-muted">
              {s.ms.toFixed(2)}ms
            </div>
          </div>
        ))}
      </CardBody>
    </Card>
  );
}
