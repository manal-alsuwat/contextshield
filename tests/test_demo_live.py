"""Live-mode demo test with the REAL client (needs models/ + a filled .env; skipped otherwise).

Covers: Scenario run + ablation + evidence; Try your own (long example with retagged policy chunks,
short example, a document with no PII / no attack, empty document, T=100); model switch to Llama;
network failure -> named-error cache fallback; My runs blocks / summary / download / clear.
Live runs bypass the response cache, so this test makes real API calls (≈ 20 small calls).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

CLEAN_DOC = """Internal policy — Opening hours. The store in Riyadh opens at 09:00 and closes at 22:00 every day except Friday, when it opens at 14:00.

Help centre — Password changes. Passwords can be changed from the Security section of the account page; a confirmation message appears on screen afterwards.

Order HP-900111 (one Lumen desk lamp) was placed on 2026-06-01 and delivered on 2026-06-04 to Jeddah with Aramex.

Community forum post: The desk lamp has a warm light setting that is great for evenings. Assembly took two minutes."""


def _btn(at, prefix):
    return next(b for b in at.button if b.label.startswith(prefix))


def _labels(at):
    return [e.label for e in at.expander]


@pytest.mark.slow
def test_live_demo_end_to_end(monkeypatch):
    pytest.importorskip("streamlit")
    import common
    from streamlit.testing.v1 import AppTest

    if not ((ROOT / "models").exists() and common.api_key_present()):
        pytest.skip("live mode needs models/ and an API key")
    monkeypatch.setenv("CONTEXTSHIELD_DEMO_MODE", "live")
    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=900)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.radio[0].value == "live"
    assert [t.label for t in at.tabs] == ["Scenario (T017)", "Benchmark (22 test tasks)", "Try your own", "My runs"]
    assert not any(s.label.startswith("Pre-recorded scenario") for s in at.selectbox), "live mode has no scenario selector"

    # --- B1 Scenario: run both (real calls) + ablation ------------------------------------------------
    _btn(at, "▶ Run both").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    key = next(k for k in at.session_state["runs"] if k.startswith("scn:"))
    runs = at.session_state["runs"][key]
    for e in ("E0", "E5"):
        assert not runs[e].get("cached"), f"{e} fell back to the recording: {runs[e].get('fallback_reason')}"
        assert not runs[e].get("response_from_cache"), f"{e} reused a cached cloud response instead of a live call"
    assert runs["E0"]["answer"] == "2026-03-14" and runs["E5"]["answer"] == "2026-03-14" and runs["E5"]["correct"]
    md = " ".join(m.value for m in at.markdown)
    assert "FINAL ANSWER" in md and "&lt;untrusted_data" in md and "#8DB600" in md
    assert sum(1 for m in at.metric if m.delta and "vs E0" in m.delta) >= 2 and not any(m.delta for m in at.metric if m.label.startswith(("e2e", "Local")))
    labels = _labels(at)
    assert any(l.startswith("Show detected text (") and "none" not in l for l in labels) and any(l.startswith("Show redacted values (") and "none" not in l for l in labels) and any(l.startswith("Show PII sent in clear (") and "none" not in l for l in labels), labels
    assert "detected by **" in md
    redacted = [d.value for d in at.dataframe if "original value → " in list(d.value.columns)]
    assert redacted and redacted[0]["placeholder"].str.startswith("<").any()
    _btn(at, "Run full ablation").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    abl = [d.value for d in at.dataframe if "Tokens sent" in list(d.value.columns) and "Experiment" in list(d.value.columns) and len(d.value) == 4]
    assert abl and list(abl[0]["Experiment"]) == ["No layer (E0)", "Naive truncation (E6)", "Smart selection (E1)", "Adaptive ContextShield (E5)"]
    assert set(at.session_state["my_runs"][0]["_runs"]) == {"E0", "E5", "E6", "E1"}

    # --- B2 Try your own: long example, policy chunks retagged trusted, run + ablation ----------------------
    import pandas as pd

    from shield.explain import chunk_document

    q = "On what date does the replacement for warranty claim WC-4471 ship?"
    long_doc = (ROOT / "demo" / "examples" / "harbor_pine_long.txt").read_text(encoding="utf-8")
    at.text_input[0].set_value(q)
    at.text_area[0].set_value(long_doc)
    _btn(at, "1 ·").click().run()
    chunks = at.session_state["own_chunks"]
    assert len(chunks) == 48
    retagged = pd.DataFrame(chunks)
    retagged.loc[retagged.chunk_id.isin(["c01", "c02"]), "source"] = "trusted"
    at.session_state["own_chunks"] = retagged
    at.run()
    _btn(at, "2 ·").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    own = at.session_state["runs"]["own"]
    assert own["E0"]["answer"] == "2026-05-19" and own["E5"]["answer"] == "2026-05-19" and "B" in own["E5"]["stages_run"] and own["E5"]["injection_detected"] and own["E5"]["pii_sent"] == 0
    assert [c["source"] for c in at.session_state["own_task"]["context"][:2]] == ["trusted", "trusted"]
    assert own["E5"]["scan"]["n_untrusted_units"] == 46, own["E5"]["scan"]  # the two policy chunks are no longer scanned/wrapped
    _btn(at, "Run full ablation").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert set(at.session_state["runs"]["own"]) == {"E0", "E5", "E6", "E1"}

    # --- B2 short example -------------------------------------------------------------------------------
    at.text_area[0].set_value((ROOT / "demo" / "examples" / "harbor_pine.txt").read_text(encoding="utf-8"))
    _btn(at, "1 ·").click().run()
    _btn(at, "2 ·").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    md = " ".join(m.value for m in at.markdown)
    assert "R1 not fired" in md and at.session_state["runs"]["own"]["E5"]["answer"] == "2026-05-19"

    # --- B2 document with no PII and no attack -> expanders say none ---------------------------------------
    at.text_input[0].set_value("On what date was order HP-900111 delivered?")
    at.text_area[0].set_value(CLEAN_DOC)
    _btn(at, "1 ·").click().run()
    _btn(at, "2 ·").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    e5 = at.session_state["runs"]["own"]["E5"]
    assert e5["answer"] == "2026-06-04" and not e5["injection_detected"] and e5["pii_redacted"] == 0
    labels = _labels(at)
    assert "Show detected text (none)" in labels and "Show redacted values (none)" in labels and "Show PII sent in clear (none)" in labels, labels

    # --- B2 empty document -> clear error, no exception -------------------------------------------------------
    at.text_area[0].set_value("")
    _btn(at, "1 ·").click().run()
    assert not at.exception and any("document is empty" in e.value for e in at.error)

    # --- B2 T = 100 on the short example -> selection fires --------------------------------------------------
    at.sidebar.number_input[0].set_value(100).run()
    at.text_input[0].set_value(q)
    at.text_area[0].set_value((ROOT / "demo" / "examples" / "harbor_pine.txt").read_text(encoding="utf-8"))
    _btn(at, "1 ·").click().run()
    _btn(at, "2 ·").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    e5 = at.session_state["runs"]["own"]["E5"]
    assert e5["T"] == 100 and "B" in e5["stages_run"] and "R1 fired" in " ".join(m.value for m in at.markdown)
    at.sidebar.number_input[0].set_value(2000).run()

    # --- B3 switch model to Llama, Scenario once (real calls) ---------------------------------------------------
    at.sidebar.selectbox[0].set_value("llama-3.3-70b").run()
    _btn(at, "▶ Run both").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    lkey = next(k for k in at.session_state["runs"] if k.startswith("scn:") and "llama" in k)
    lr = at.session_state["runs"][lkey]
    assert lr["E5"]["model"] == "meta-llama/llama-3.3-70b-instruct" and lr["E5"]["answer"] and not lr["E5"].get("error")
    at.sidebar.selectbox[0].set_value("gpt-4o-mini").run()

    # --- B4 network failure -> named-error fallback to the pre-recorded run, no traceback ----------------------
    monkeypatch.setattr(common, "LIVE_TIMEOUT_S", 8.0)
    monkeypatch.setenv("LLM_BASE_URL", "https://invalid.host.invalid/v1")
    _btn(at, "▶ Run both").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    warn = [w.value for w in at.warning if "CACHED response shown" in w.value]
    assert warn and any(("Error" in w or "error" in w) for w in warn), warn
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.setattr(common, "LIVE_TIMEOUT_S", 60.0)

    # --- B5 My runs: blocks, summary + TOTAL, download, clear ---------------------------------------------------
    log = at.session_state["my_runs"]
    assert [r["source"] for r in log][:2] == ["scenario", "own"] and len(log) >= 6
    blocks = [d.value for d in at.dataframe if "Tokens sent" in list(d.value.columns) and "Experiment" in list(d.value.columns)]
    assert len(blocks) >= len(log)
    summary = [d.value for d in at.dataframe if "reduction %" in list(d.value.columns) and "time" in list(d.value.columns)]
    assert summary and summary[0].iloc[-1]["time"] == "TOTAL" and len(summary[0]) == len(log) + 1
    assert sum(1 for el in at.main if el.type == "vega_lite_chart") >= 6
    _btn(at, "🗑 Clear").click().run()
    assert not at.exception and at.session_state["my_runs"] == [] and any("No live runs yet" in i.value for i in at.info)
