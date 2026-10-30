"""Deterministic answer scoring: normalized exact match / containment. No LLM judge."""
from __future__ import annotations

import re
from datetime import datetime

_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})


def _normalize_dates(s: str) -> str:
    # "March 14, 2026" / "14 March 2026" / "14 Mar 2026" / "03/14/2026" -> 2026-03-14
    def mdy(m):
        mon = _MONTHS.get(m.group(1).lower()[:3])
        return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}" if mon else m.group(0)

    def dmy(m):
        mon = _MONTHS.get(m.group(2).lower()[:3])
        return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(1)):02d}" if mon else m.group(0)

    s = re.sub(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", mdy, s)
    s = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b", dmy, s)
    s = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}", s)
    return s


def normalize(s: str) -> str:
    s = (s or "").strip()
    s = _normalize_dates(s)
    s = s.lower()
    s = s.replace("sar", " ").replace("usd", " ").replace("$", " ")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)  # 1,250 -> 1250
    s = re.sub(r"(\d+)\.00\b", r"\1", s)  # 450.00 -> 450
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(answer: str, gold: str) -> bool:
    """gold may contain several required parts separated by ' | ' (all must be contained)."""
    a = normalize(answer)
    if not a:
        return False
    parts = [normalize(p) for p in gold.split("|")]
    for p in parts:
        if not p:
            continue
        if p == a:
            continue
        if re.search(rf"(?<![\w-]){re.escape(p)}(?![\w-])", a):
            continue
        return False
    return True


if __name__ == "__main__":
    assert is_correct("March 14, 2026", "2026-03-14")
    assert is_correct("The refund was Completed on 2026-03-14.", "Completed | 2026-03-14")
    assert is_correct("SAR 1,250.00", "1250")
    assert not is_correct("2026-03-15", "2026-03-14")
    assert not is_correct("NOT FOUND", "Approved")
    print("scoring ok")
