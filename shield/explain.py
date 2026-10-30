"""Human-readable explanations and lightweight helpers shared by the demo and the CLI.

No ML imports here: the cached demo mode and the CLI help text must stay light.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .tokens import count_tokens

TARGET_CHUNK_TOKENS = 150


def explain_router(decision: dict | None, T: int = 2000) -> list[str]:
    """Turn a router decision dict (ShieldResult.router) into plain sentences."""
    if not decision:
        return ["No router: fixed pipeline (every configured stage runs)."]
    fired = set(decision.get("rules_fired", []))
    n = decision.get("input_tokens", 0)
    out = []
    if "R1" in fired:
        out.append(f"R1 fired: {n:,} input tokens > T={T:,} → relevance selection (B) runs.")
    else:
        out.append(f"R1 not fired: {n:,} input tokens ≤ T={T:,} → the whole document fits the budget, selection (B) is skipped on purpose — no tokens to save, no embedding cost.")
    if "R2" in fired:
        out.append("R2 fired: at least one chunk is untrusted → injection scan (A) + untrusted wrapping (D) run.")
    else:
        out.append("R2 not fired: no untrusted chunk → no injection scan, nothing to wrap.")
    hits = decision.get("prescan_hits", 0)
    if "R3" in fired:
        out.append(f"R3 fired: regex PII pre-scan found {hits} hit(s) → PII redaction (C) runs on the outbound text.")
    else:
        out.append("R3 not fired: regex PII pre-scan found nothing → redaction (C) skipped (no spaCy/Presidio cost).")
    if not fired:
        out.append("No rule fired → pass-through: the context goes out unchanged except for the cheap pre-scan.")
    return out


def chunk_document(text: str, source: str = "untrusted", kind: str = "document", target_tokens: int = TARGET_CHUNK_TOKENS) -> list[dict]:
    """Split free text into ~150-token chunks on paragraph / sentence boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur, cur_tok = "", 0
    for p in paras:
        pt = count_tokens(p)
        if pt > target_tokens * 1.5:  # long paragraph: split on sentences
            for s in re.split(r"(?<=[.!?؟])\s+", p):
                st = count_tokens(s)
                if cur and cur_tok + st > target_tokens:
                    chunks.append(cur.strip())
                    cur, cur_tok = "", 0
                cur += (" " if cur else "") + s
                cur_tok += st
            continue
        if cur and cur_tok + pt > target_tokens:
            chunks.append(cur.strip())
            cur, cur_tok = "", 0
        cur += ("\n\n" if cur else "") + p
        cur_tok += pt
    if cur.strip():
        chunks.append(cur.strip())
    return [{"chunk_id": f"c{i + 1:02d}", "source": source, "kind": kind, "text": c} for i, c in enumerate(chunks)]


def load_context_file(path: str | Path, default_source: str = "untrusted") -> list[dict]:
    """.json: a task file ({"context": [...]}) or a list of {text, source?}; anything else: plain text."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(raw)
        items = data["context"] if isinstance(data, dict) and "context" in data else data
        out = []
        for i, it in enumerate(items):
            if isinstance(it, str):
                it = {"text": it}
            out.append({"chunk_id": it.get("chunk_id", f"c{i + 1:02d}"), "source": it.get("source", default_source), "kind": it.get("kind", "document"), "text": it["text"]})
        return out
    return chunk_document(raw, source=default_source)
