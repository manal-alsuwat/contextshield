"""The cached demo mode must start without any ML import and without an API key (Streamlit Community Cloud)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
HEAVY = ("torch", "transformers", "sentence_transformers", "presidio_analyzer", "spacy", "onnxruntime", "optimum", "psutil")


@pytest.mark.slow
def test_cached_demo_starts_without_ml_imports(monkeypatch):
    pytest.importorskip("streamlit")
    import common
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CONTEXTSHIELD_DEMO_MODE", "cached")
    monkeypatch.setenv("LLM_API_KEY", "")
    for m in HEAVY:
        sys.modules.pop(m, None)
    t0 = time.time()
    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=60)
    at.run()
    elapsed = time.time() - t0
    assert not at.exception, [str(e) for e in at.exception]
    loaded = [m for m in HEAVY if m in sys.modules]
    assert not loaded, f"cached mode imported ML modules: {loaded}"
    assert [t.label for t in at.tabs] == ["Scenario (T017)", "Benchmark (22 test tasks)", "Try your own", "My runs"]
    assert at.sidebar.radio[0].value == "cached"
    assert len(at.metric) >= 8  # both scenario panels rendered from demo/cached/<model>/<task>/
    assert not any(m.delta for m in at.metric if m.label.startswith(("e2e", "Local"))), "latency metrics carry no delta"
    assert any(m.delta and "vs E0" in m.delta for m in at.metric if m.label == "Input tokens")
    labels = [e.label for e in at.expander]
    assert any(l.startswith("Show detected text") for l in labels) and any(l.startswith("Show redacted values") for l in labels) and any(l.startswith("Show PII sent in clear") for l in labels), labels
    md = " ".join(m.value for m in at.markdown)
    assert "recorded values" in md and "T = **2,000**" in md, "cached E5 panel shows the recorded T / budget"
    assert any("No live runs yet" in i.value for i in at.info), "My runs tab shows its empty state in cached mode"
    assert sum(1 for el in at.main if el.type == "vega_lite_chart") >= 3  # 2 benchmark + scenario pair
    assert elapsed < 30, f"cached start took {elapsed:.1f}s"  # < 10 s on an idle machine; generous bound for CI

    # scenario selector: 5 pre-recorded test tasks, every one renders for every model that has a recording
    sel = next(s for s in at.selectbox if s.label.startswith("Pre-recorded scenario"))
    assert [o.split(" — ")[0] for o in sel.options] == [s["task_id"] for s in common.SCENARIOS] and len(sel.options) == 5  # options carry the "why" label
    for mid_label, (mid, _, _) in common.MODELS.items():
        if not common.has_cached(mid):
            continue
        at.sidebar.selectbox[0].set_value(mid_label).run()
        for opt in sel.options:
            tid = opt.split(" — ")[0]
            if not common.has_cached(mid, tid):
                continue
            next(s for s in at.selectbox if s.label.startswith("Pre-recorded scenario")).set_value(opt).run()
            assert not at.exception, (mid, tid, [str(e) for e in at.exception])
            assert len(at.metric) >= 8, (mid, tid)
            labels = [e.label for e in at.expander]
            assert any(l.startswith("Show detected text") for l in labels) and any(l.startswith("Show redacted values") for l in labels), (mid, tid, labels)
    # a scenario without a planted attack says "none" instead of hiding the expander
    t019 = next(o for o in sel.options if o.startswith("T019"))
    next(s for s in at.selectbox if s.label.startswith("Pre-recorded scenario")).set_value(t019).run()
    assert any(l == "Show detected text (none)" for l in (e.label for e in at.expander))
