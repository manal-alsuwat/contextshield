"""Stage C — PII redaction (Microsoft Presidio + custom regex recognizers).

FIXED scope (6 types, stated explicitly in every output):
  1. Saudi national ID       (10 digits, starts with 1 or 2)          -> <ID_n>
  2. Saudi mobile numbers    (05XXXXXXXX, +9665XXXXXXXX, 009665XXXXXXXX,
                              5XXXXXXXX)                              -> <PHONE_n>
  3. Email addresses                                                  -> <EMAIL_n>
  4. IBAN (SA + 22 chars)                                             -> <IBAN_n>
  5. Payment card numbers (Luhn-valid)                                -> <CARD_n>
  6. English person names (spaCy en_core_web_sm NER via Presidio)     -> <PERSON_n>

The re-identification map stays in memory on the local machine only; it is
never written to logs, result files or the cloud.

`regex_prescan` is the cheap router check (R3): regex only, no spaCy.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("shield.pii")

PII_TYPES = ["ID", "PHONE", "EMAIL", "IBAN", "CARD", "PERSON"]

RE_SA_ID = r"\b[12]\d{9}\b"
RE_SA_PHONE = r"(?:\+966|00966)[\s-]?5\d{8}\b|\b05\d{8}\b|(?<![\d+])5\d{8}\b"
RE_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
RE_IBAN = r"\bSA\d{2}[A-Z0-9]{20}\b"
RE_CARD = r"\b(?:\d[ -]?){12,18}\d\b"
# cheap name cue used only by the pre-scan (R3); real PERSON detection is spaCy NER
RE_NAME_CUE = r"\b(?:Dear|Mr\.?|Ms\.?|Mrs\.?|Customer|Name|Contact|Attn|Cardholder|Account holder|Requester|Regards,)\s*:?\s*[A-Z][a-z]+ [A-Z][a-z]+\b"

_PRESCAN = [re.compile(p) for p in (RE_SA_ID, RE_SA_PHONE, RE_EMAIL, RE_IBAN, RE_NAME_CUE)]
_CARD = re.compile(RE_CARD)


def luhn_ok(s: str) -> bool:
    digits = [int(ch) for ch in s if ch.isdigit()]
    if len(digits) < 13:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def regex_prescan(text: str) -> int:
    """Number of cheap regex hits (R3). No spaCy, no Presidio."""
    n = sum(len(p.findall(text)) for p in _PRESCAN)
    n += sum(1 for m in _CARD.finditer(text) if luhn_ok(m.group(0)))
    return n


# --- Presidio engine ----------------------------------------------------------
_ENGINE = None


def get_analyzer():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import CreditCardRecognizer, EmailRecognizer, SpacyRecognizer

    t0 = time.perf_counter()
    # Prefer the spaCy model shipped in the repo (models/en_core_web_sm, MIT); fall back to the installed package.
    local = Path(os.environ.get("SHIELD_MODEL_DIR", Path(__file__).resolve().parent.parent / "models")) / "en_core_web_sm"
    spacy_model = str(local) if (local / "meta.json").exists() else "en_core_web_sm"
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    })
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.add_recognizer(PatternRecognizer(supported_entity="SA_ID", patterns=[Pattern("sa_id", RE_SA_ID, 0.9)]))
    registry.add_recognizer(PatternRecognizer(supported_entity="SA_PHONE", patterns=[Pattern("sa_phone", RE_SA_PHONE, 0.9)]))
    registry.add_recognizer(PatternRecognizer(supported_entity="SA_IBAN", patterns=[Pattern("sa_iban", RE_IBAN, 0.95)]))
    registry.add_recognizer(EmailRecognizer())
    registry.add_recognizer(CreditCardRecognizer())  # Luhn-validated
    registry.add_recognizer(SpacyRecognizer(supported_entities=["PERSON"]))
    _ENGINE = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
    log.info("Presidio analyzer ready in %.1fs", time.perf_counter() - t0)
    return _ENGINE


_ENTITY_MAP = {"SA_ID": "ID", "SA_PHONE": "PHONE", "EMAIL_ADDRESS": "EMAIL", "SA_IBAN": "IBAN", "CREDIT_CARD": "CARD", "PERSON": "PERSON"}


@dataclass
class Redactor:
    """Consistent typed placeholders across all texts of one request."""

    _map: dict[str, str] = field(default_factory=dict)  # original -> placeholder (LOCAL ONLY)
    _counters: dict[str, int] = field(default_factory=lambda: {t: 0 for t in PII_TYPES})
    counts: dict[str, int] = field(default_factory=lambda: {t: 0 for t in PII_TYPES})
    spans: list[dict] = field(default_factory=list)  # typed, value-free log
    _span_values: list[str] = field(default_factory=list)  # LOCAL ONLY (metrics), never persisted
    ms: float = 0.0

    def _placeholder(self, typ: str, value: str) -> str:
        key = f"{typ}:{value}"
        if key not in self._map:
            self._counters[typ] += 1
            self._map[key] = f"<{typ}_{self._counters[typ]}>"
        return self._map[key]

    def redact(self, text: str, label: str = "") -> str:
        if not text:
            return text
        t0 = time.perf_counter()
        analyzer = get_analyzer()
        results = analyzer.analyze(text=text, language="en", entities=list(_ENTITY_MAP), score_threshold=0.4)
        results = [r for r in results if r.entity_type in _ENTITY_MAP]
        # person names: require >= 2 tokens and letters only (reduces NER noise)
        kept = []
        for r in results:
            val = text[r.start:r.end]
            if r.entity_type == "PERSON" and not re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+", val.strip()):
                continue
            kept.append(r)
        kept.sort(key=lambda r: (r.start, -(r.end - r.start)))
        out, pos, last_end = [], 0, -1
        for r in kept:
            if r.start < last_end:
                continue  # overlap -> keep the earlier/longer span
            typ = _ENTITY_MAP[r.entity_type]
            val = text[r.start:r.end]
            out.append(text[pos:r.start])
            out.append(self._placeholder(typ, val))
            self.counts[typ] += 1
            self.spans.append({"where": label, "type": typ, "len": r.end - r.start})
            self._span_values.append(val)
            pos = r.end
            last_end = r.end
        out.append(text[pos:])
        self.ms += (time.perf_counter() - t0) * 1000
        return "".join(out)

    def originals(self) -> list[str]:
        """Original values seen (used ONLY for local metric computation; never persisted)."""
        return [k.split(":", 1)[1] for k in self._map]

    @property
    def total(self) -> int:
        return sum(self.counts.values())
