"""ContextShield AI — zero-LLM local context preprocessing layer.

Stages:
  A  injection scan   (shield.injection)
  B  relevance select (shield.relevance)
  C  PII redaction    (shield.pii)
  D  untrusted wrap   (shield.wrap)
Router (shield.router) decides which stages run; shield.pipeline wires them.

The pipeline is imported lazily so that light helpers (shield.explain, shield.tokens)
can be used without pulling numpy/psutil/ML dependencies (cached demo mode, CLI help).
"""
from __future__ import annotations

__all__ = ["ContextShield", "ShieldConfig", "ShieldResult"]


def __getattr__(name: str):
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
