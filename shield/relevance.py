"""Stage B — relevance selection.

* Chunking: dataset chunks are used as-is when <= MAX_CHUNK_TOKENS (200);
  larger ones are split sentence-boundary aware into sub-chunks whose ids keep
  the parent id (c07 -> c07.1, c07.2). MiniLM max is 256 tokens; never exceed.
* Embeddings: sentence-transformers/all-MiniLM-L6-v2, computed once per text
  and cached (in-memory + on-disk npz keyed by sha1 of the text).
* Score = cosine(query, chunk). Rank descending.
* Selection = token budget (default 2500) + minimum 2 chunks.
* Must-keep: any chunk containing an entity string from the question
  (IDs, names, dates, amounts) is kept, plus its +/-1 neighbours; must-keep
  chunks are charged to the budget first.
* Log per chunk: {chunk_id, score, selected, must_keep_reason}.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np

from .tokens import count_tokens, split_sentences

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBED_DIR = Path(os.environ.get("SHIELD_MODEL_DIR", Path(__file__).resolve().parent.parent / "models")) / "all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS = 200
MIN_CHUNK_TOKENS = 150
CACHE_DIR = Path(os.environ.get("SHIELD_CACHE_DIR", Path(__file__).resolve().parent.parent / "cache")) / "embeddings"

_ENTITY_PATTERNS = [
    ("id", re.compile(r"\b[A-Z]{2,5}-\d{3,8}\b")),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("amount", re.compile(r"\b(?:SAR|USD|\$)\s?\d[\d,]*(?:\.\d+)?\b|\b\d{3,}(?:\.\d+)?\b")),
    ("name", re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")),
    ("quoted", re.compile(r"[\"“']([^\"”']{3,40})[\"”']")),
]
_STOP_NAMES = {"What Is", "When Was", "How Many", "Which Team", "Who Is", "How Much"}


def question_entities(question: str) -> list[tuple[str, str]]:
    ents = []
    for kind, pat in _ENTITY_PATTERNS:
        for m in pat.finditer(question):
            val = m.group(1) if m.groups() else m.group(0)
            if kind == "name" and val in _STOP_NAMES:
                continue
            ents.append((kind, val))
    return ents


def split_chunk(chunk: dict) -> list[dict]:
    """Split a chunk > MAX_CHUNK_TOKENS at sentence boundaries."""
    parent = chunk.get("parent_id", chunk["chunk_id"])
    if count_tokens(chunk["text"]) <= MAX_CHUNK_TOKENS:
        return [{**chunk, "parent_id": parent}]
    # paragraph boundaries (newlines) are preserved so stage A can still excise a single paragraph
    pieces: list[tuple[str, str]] = []  # (sentence, separator before it)
    for pi, para in enumerate(re.split(r"\n+", chunk["text"])):
        for si, s in enumerate(split_sentences(para)):
            pieces.append((s, ("\n" if si == 0 and pi > 0 else " ") if pieces else ""))
    parts, cur, cur_tok = [], "", 0
    for s, sep in pieces:
        st = count_tokens(s)
        if cur and cur_tok + st > MAX_CHUNK_TOKENS:
            parts.append(cur)
            cur, cur_tok, sep = "", 0, ""
        cur += sep + s
        cur_tok += st
    if cur:
        parts.append(cur)
    return [
        {**chunk, "chunk_id": f"{chunk['chunk_id']}.{i + 1}", "parent_id": parent, "text": p}
        for i, p in enumerate(parts)
    ]


class Embedder:
    _instance = None

    def __init__(self):
        # Prefer the copy shipped in the repo (models/all-MiniLM-L6-v2, Apache-2.0): no download, no token.
        local = LOCAL_EMBED_DIR if (LOCAL_EMBED_DIR / "config.json").exists() else None
        if local is not None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        self.model = SentenceTransformer(str(local) if local else EMBED_MODEL, device="cpu")
        self.model.max_seq_length = 256
        self.load_s = time.perf_counter() - t0
        self._mem: dict[str, np.ndarray] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        todo = []
        for i, k in enumerate(keys):
            if k in self._mem:
                out[i] = self._mem[k]
                continue
            f = CACHE_DIR / f"{k}.npy"
            if f.exists():
                v = np.load(f)
                self._mem[k] = v
                out[i] = v
            else:
                todo.append(i)
        if todo:
            vecs = self.model.encode([texts[i] for i in todo], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            for i, v in zip(todo, vecs):
                v = np.asarray(v, dtype=np.float32)
                self._mem[keys[i]] = v
                np.save(CACHE_DIR / f"{keys[i]}.npy", v)
                out[i] = v
        return np.stack(out)  # type: ignore[arg-type]


def select_relevant(question: str, chunks: list[dict], budget_tokens: int = 2500, min_chunks: int = 2) -> tuple[list[dict], list[dict], dict]:
    """Return (selected chunks in original order, per-chunk log, info)."""
    t0 = time.perf_counter()
    units: list[dict] = []
    for c in chunks:
        units.extend(split_chunk(c))
    emb = Embedder.get()
    qv = emb.encode([question])[0]
    cv = emb.encode([u["text"] for u in units])
    scores = cv @ qv
    tok = [count_tokens(u["text"]) for u in units]

    ents = question_entities(question)
    must: dict[int, str] = {}
    for i, u in enumerate(units):
        for kind, val in ents:
            if val in u["text"]:
                must.setdefault(i, f"{kind}:{val}")
                for j in (i - 1, i + 1):
                    if 0 <= j < len(units):
                        must.setdefault(j, f"neighbor_of:{u['chunk_id']}")
                break

    selected = set()
    used = 0
    for i in sorted(must, key=lambda k: -scores[k]):
        selected.add(i)
        used += tok[i]
    for i in np.argsort(-scores):
        i = int(i)
        if i in selected:
            continue
        if used + tok[i] <= budget_tokens or len(selected) < min_chunks:
            selected.add(i)
            used += tok[i]

    log = [
        {"chunk_id": u["chunk_id"], "score": round(float(scores[i]), 4), "tokens": tok[i], "selected": i in selected, "must_keep_reason": must.get(i)}
        for i, u in enumerate(units)
    ]
    out = [units[i] for i in range(len(units)) if i in selected]
    info = {
        "n_units": len(units),
        "n_selected": len(out),
        "selected_tokens": used,
        "budget_tokens": budget_tokens,
        "entities": [f"{k}:{v}" for k, v in ents],
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }
    return out, log, info


def naive_truncate(chunks: list[dict], budget_tokens: int = 2500) -> list[dict]:
    """E6 baseline: keep chunks in original order until the budget is used."""
    out, used = [], 0
    for c in chunks:
        t = count_tokens(c["text"])
        if used + t > budget_tokens:
            remain = budget_tokens - used
            if remain > 20:
                from .tokens import truncate_to_tokens

                out.append({**c, "text": truncate_to_tokens(c["text"], remain), "truncated": True})
            break
        out.append(c)
        used += t
    return out
