// Typed client for the Agentic RAG FastAPI backend.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Flags = {
  useHybrid: boolean;
  useRerank: boolean;
  useHyde: boolean;
  useMmr: boolean;
  useDecompose: boolean;
  agent: boolean;
  crag: boolean;
  router: boolean;
  cache: boolean;
};

export type Citation = {
  chunk_id: string;
  doc_id: string;
  section: string | null;
  quote?: string | null;
};

export type Ctx = {
  chunk_id: string;
  doc_id: string;
  section: string | null;
  score: number;
  source: string;
  ranks: Record<string, number>;
  text: string;
};

export type Stage = { stage: string; ms: number; meta: Record<string, unknown> };

export type QueryResponse = {
  query: string;
  answer: string;
  abstained: boolean;
  iterations: number;
  route: string;
  from_cache: boolean;
  retrieval_grade: string | null;
  retrieval_grade_score: number | null;
  input_flags: string[];
  citations: Citation[];
  contexts: Ctx[];
  trace: Stage[];
  cost_usd: number;
  tokens: Record<string, number>;
  support_fraction: number | null;
};

export async function getConfig(): Promise<{ mode: string; flags: Flags }> {
  const r = await fetch(`${API_URL}/config`);
  return r.json();
}

export async function getCorpus() {
  const r = await fetch(`${API_URL}/corpus`);
  return r.json() as Promise<{
    n_docs: number;
    n_chunks: number;
    docs: { doc_id: string; chunks: number }[];
  }>;
}

export async function runQuery(query: string, flags: Flags): Promise<QueryResponse> {
  const r = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, flags }),
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

// Streaming: POST SSE, parse `event:`/`data:` frames, invoke callbacks as the
// pipeline reveals each stage, then the final answer.
export async function streamQuery(
  query: string,
  flags: Flags,
  onStage: (s: Stage) => void,
  onAnswer: (a: QueryResponse) => void,
): Promise<void> {
  const r = await fetch(`${API_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, flags }),
  });
  if (!r.body) throw new Error("no stream body");
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split("\n\n");
    buf = frames.pop() || "";
    for (const frame of frames) {
      const ev = /event:\s*(.*)/.exec(frame)?.[1]?.trim();
      const dataLine = /data:\s*([\s\S]*)/.exec(frame)?.[1];
      if (!ev || !dataLine) continue;
      const data = JSON.parse(dataLine);
      if (ev === "stage") onStage(data as Stage);
      else if (ev === "answer") onAnswer(data as QueryResponse);
    }
  }
}
