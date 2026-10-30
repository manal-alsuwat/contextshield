"""Token counting helpers (tiktoken cl100k_base, with a crude fallback)."""
from __future__ import annotations

import re
from functools import lru_cache

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - offline fallback
    _enc = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _enc is not None:
        return len(_enc.encode(text, disallowed_special=()))
    return max(1, int(len(text.split()) * 1.3))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return the first `max_tokens` tokens of `text`."""
    if max_tokens <= 0:
        return ""
    if _enc is not None:
        ids = _enc.encode(text, disallowed_special=())
        if len(ids) <= max_tokens:
            return text
        return _enc.decode(ids[:max_tokens])
    words = text.split()
    return " ".join(words[: int(max_tokens / 1.3)])


_SENT_RE = re.compile(r"(?<=[.!?؟])\s+|\n+")


@lru_cache(maxsize=4096)
def split_sentences(text: str) -> tuple[str, ...]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return tuple(parts)
