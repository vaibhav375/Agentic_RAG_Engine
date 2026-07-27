"use client";

import { Card, CardBody, CardHeader, Switch, Button } from "@/components/ui/primitives";
import type { Flags } from "@/lib/api";

const PRESETS: Record<string, Partial<Flags>> = {
  Baseline: { useHybrid: false, useRerank: false, useHyde: false, useMmr: false, useDecompose: false, agent: false, crag: false, router: false, cache: false },
  "Full pipeline": { useHybrid: true, useRerank: true, useHyde: false, useMmr: false, useDecompose: false, agent: true, crag: true, router: true, cache: true },
};

const ROWS: { key: keyof Flags; label: string; hint: string }[] = [
  { key: "useHybrid", label: "Hybrid retrieval", hint: "dense + BM25 · RRF" },
  { key: "useRerank", label: "Cross-encoder rerank", hint: "precision" },
  { key: "useMmr", label: "MMR", hint: "diversity" },
  { key: "useHyde", label: "HyDE", hint: "hypothetical doc" },
  { key: "useDecompose", label: "Query decomposition", hint: "multi-hop" },
  { key: "agent", label: "Self-correction loop", hint: "critic + retry" },
  { key: "crag", label: "CRAG answerability gate", hint: "abstain if OOS" },
  { key: "router", label: "Query router", hint: "cost-aware" },
  { key: "cache", label: "Semantic cache", hint: "latency" },
];

export function FeatureToggles({
  flags,
  setFlags,
}: {
  flags: Flags;
  setFlags: (f: Flags) => void;
}) {
  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <span>Pipeline</span>
        <div className="flex gap-2">
          {Object.keys(PRESETS).map((name) => (
            <Button
              key={name}
              variant="ghost"
              className="px-2 py-1 text-xs"
              onClick={() => setFlags({ ...flags, ...PRESETS[name] } as Flags)}
            >
              {name}
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardBody className="py-2">
        {ROWS.map((r) => (
          <Switch
            key={r.key}
            label={r.label}
            hint={r.hint}
            checked={flags[r.key]}
            onChange={(v) => setFlags({ ...flags, [r.key]: v })}
          />
        ))}
      </CardBody>
    </Card>
  );
}
