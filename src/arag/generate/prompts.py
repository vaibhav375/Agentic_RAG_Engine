"""Prompt templates for the real LLM backends. The mock backend implements the
same tasks procedurally and does not use these strings."""

ABSTAIN_PHRASE = "INSUFFICIENT_CONTEXT"

ANSWER_SYSTEM = """You are a precise technical documentation assistant. Answer the \
user's question using ONLY the numbered context passages provided. Rules:
- Ground every sentence in the passages. Do not use outside knowledge.
- After each sentence, cite the passage id(s) you used like [c3] or [c1][c4].
- If the passages do not contain enough information to answer, reply with exactly:
  {abstain}
Keep the answer concise (1-4 sentences)."""

ANSWER_USER = """Question: {question}

Context passages:
{context_block}

Answer (grounded, with [id] citations):"""

CLAIMS_SYSTEM = """Break the following answer into a JSON list of atomic factual \
claims. Each claim should be a single verifiable statement. Return ONLY a JSON \
array of strings."""

JUDGE_SYSTEM = """You are a strict fact-checker. Decide whether the CLAIM is fully \
supported by the CONTEXT. Answer with a JSON object:
{"supported": true|false, "confidence": 0.0-1.0, "reason": "<short>"}
A claim is supported only if the context directly entails it. If the context is \
silent or only partially covers it, it is NOT supported."""

JUDGE_USER = """CONTEXT:
{context}

CLAIM: {claim}

JSON:"""

REFORMULATE_SYSTEM = """You rewrite a search query to retrieve missing evidence. \
Given the original question, what information was missing, and previous queries \
tried, produce ONE improved search query (keywords allowed). Return only the query."""

REFORMULATE_USER = """Original question: {question}
Missing information: {missing}
Previously tried: {prior}

Improved query:"""

ROUTER_SYSTEM = """Classify the question's retrieval difficulty. Reply with one \
word: "simple" (a single fact lookup) or "complex" (needs multi-hop reasoning, \
comparison, or synthesis across sources)."""
