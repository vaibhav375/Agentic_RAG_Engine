"""Input guardrail: detect prompt-injection / jailbreak / false-premise attempts.

This is a lightweight, explicit safety layer at the front door. It does not, by
itself, decide the answer (the CRAG gate + grounded critic already prevent the
model from acting on injected instructions or planted falsehoods), but it *flags*
adversarial inputs so they surface in the trace / API response and can be
rate-limited, logged, or routed to stricter handling — the defense-in-depth
pattern expected around LLM apps.
"""

from __future__ import annotations

import re

# Patterns: instruction-override, role/jailbreak, exfiltration, and false-premise
# priming ("as we established", "everyone knows") used to smuggle assumptions.
_PATTERNS: list[tuple[str, str]] = [
    ("instruction_override", r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instruction|prompt|rule|above|previous)\b"),
    ("role_jailbreak", r"\b(you are now|act as|developer mode|unrestricted mode|jailbreak|DAN)\b"),
    ("exfiltration", r"\b(reveal|print|show|leak|expose)\b.{0,30}\b(system prompt|api key|secret|password|token|credentials)\b"),
    ("false_premise", r"\b(as we established|as established earlier|everyone knows|the docs? clearly|obviously|confirm that)\b"),
]

_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _PATTERNS]


def scan_input(query: str) -> list[str]:
    """Return the names of any injection patterns matched in the query."""
    return [name for name, rx in _COMPILED if rx.search(query)]
