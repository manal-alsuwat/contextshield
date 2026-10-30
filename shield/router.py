"""Router — exactly 3 rules (FIXED).

R1  input_tokens > T                 -> run B (relevance selection)
R2  any chunk has source=untrusted   -> run A (injection scan) + D (wrap)
R3  regex PII pre-scan finds >= 1 hit-> run C (PII redaction)
No rule fires -> pass-through with only the cheap regex pre-scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pii import regex_prescan
from .tokens import count_tokens


@dataclass
class RouterDecision:
    run_A: bool = False
    run_B: bool = False
    run_C: bool = False
    run_D: bool = False
    rules_fired: list[str] = field(default_factory=list)
    input_tokens: int = 0
    prescan_hits: int = 0

    def as_dict(self) -> dict:
        return {
            "rules_fired": list(self.rules_fired),
            "run_A": self.run_A,
            "run_B": self.run_B,
            "run_C": self.run_C,
            "run_D": self.run_D,
            "input_tokens": self.input_tokens,
            "prescan_hits": self.prescan_hits,
        }


def route(question: str, chunks: list[dict], T: int = 2000) -> RouterDecision:
    d = RouterDecision()
    d.input_tokens = count_tokens(question) + sum(count_tokens(c["text"]) for c in chunks)

    # R1
    if d.input_tokens > T:
        d.run_B = True
        d.rules_fired.append("R1")
    # R2
    if any(c.get("source") == "untrusted" for c in chunks):
        d.run_A = True
        d.run_D = True
        d.rules_fired.append("R2")
    # R3 — cheap regex pre-scan over question + all chunk text (no spaCy, no Presidio)
    d.prescan_hits = regex_prescan(question + "\n" + "\n".join(c["text"] for c in chunks))
    if d.prescan_hits >= 1:
        d.run_C = True
        d.rules_fired.append("R3")
    return d
