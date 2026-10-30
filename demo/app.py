"""Streamlit demo — Customer Support Agent: Baseline vs ContextShield.
  streamlit run demo/app.py

Data source (sidebar):
  cached — reads demo/cached/<model>/ and results/summary.csv; imports no ML libraries, needs no API key
           (this is the mode used on Streamlit Community Cloud, see DEPLOY.md).
  live   — loads the local models and calls the API (needs the full requirements.txt and .env).
Tabs: Scenario (fixed task T017, demo question) · Try your own (live only) · Benchmark (22 test tasks) · My runs.
"""
from __future__ import annotations

import html
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (no ML imports at module level)

ROOT = common.ROOT
st.set_page_config(page_title="ContextShield — Less Context, Same Answer", layout="wide")

# ----------------------------------------------------------------------------- experiment labels / colours
EXP_LONG = {"E0": "No layer", "E6": "Naive truncation", "E1": "Smart selection", "E2": "Selection + PII redaction", "E5": "Adaptive ContextShield"}
EXP_SHORT = {"E0": "No layer", "E6": "Naive cut", "E1": "Smart selection", "E2": "Selection + PII", "E5": "Adaptive"}
EXP_COLORS = {"E0": "#d32f2f", "E6": "#9e9e9e", "E1": "#2e7d32", "E2": "#00897b", "E5": "#1565c0"}
ORDER = ["E0", "E6", "E1", "E2", "E5"]
ABLATION_EXPS = ["E6", "E1"]  # optional extra runs on top of E0 / E5


def long_label(e: str) -> str:
    return f"{EXP_LONG.get(e, e)} ({e})"


# ----------------------------------------------------------------------------- mode / model / knobs
def _secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


live_possible = (ROOT / "models").exists() and common.api_key_present()
default_mode = _secret("CONTEXTSHIELD_DEMO_MODE", "live" if live_possible else "cached")
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Data source", ["cached", "live"], index=0 if default_mode == "cached" else 1,
                    help="cached: pre-recorded runs + measured tables, no ML models, no API key. live: local models + API call.")
    if mode == "live" and not live_possible:
        st.warning("Live mode needs the local models (models/) and an API key in .env — falling back to cached.")
        mode = "cached"
    labels = list(common.MODELS)
    model_label = st.selectbox("Model", labels, index=0, format_func=lambda l: l + ("" if common.has_cached(common.MODELS[l][0]) or mode == "live" else "  (no cached run)"))
    model_id = common.MODELS[model_label][0]
    st.caption(f"`{model_id}` · prices ${common.MODELS[model_label][1]}/{common.MODELS[model_label][2]} per 1M tokens")
    st.caption("Mode: **" + mode + "** · " + ("no ML libraries loaded" if mode == "cached" else f"local models + API (timeout {common.LIVE_TIMEOUT_S:.0f} s)"))
    st.divider()
    sel_T = int(st.number_input("Selection threshold T (tokens)", min_value=100, value=2000, step=100, help="Router rule R1: relevance selection (B) runs only when the input has more than T tokens."))
    sel_budget = 2500  # fixed, as in the experiments
    st.caption(f"Selection budget: {sel_budget:,} tokens (fixed, as in the benchmark)")
    st.caption("Below T the document is sent whole: selection only helps when the document is larger than the budget." + ("" if mode == "live" else " Cached runs show the values they were recorded with."))

st.title("ContextShield — Less Context, Same Answer")
st.markdown("A zero-LLM layer before any language model: fewer tokens, redacted PII, isolated injections. Demo domain: customer support.")
if mode == "live":
    with st.spinner("Loading the local models once (MiniLM, Presidio/spaCy, DeBERTa ONNX) …"):
        warm_s = common.warm_up()
    if warm_s:
        st.caption(f"Local models loaded in {warm_s:.0f} s — live runs now spend their {common.LIVE_TIMEOUT_S:.0f} s budget on the API call only.")

# ----------------------------------------------------------------------------- cards
CARD = {
    "sent": ("#9e9e9e", "#fafafa", "SENT — no layer, nothing filtered"),
    "kept": ("#8DB600", "#f3f8e6", "KEPT by ContextShield"),
    "excised": ("#d32f2f", "#fdecea", "KEPT — paragraph(s) EXCISED by A"),
    "dropped_B": ("#9e9e9e", "#f1f3f4", "DROPPED by B (not relevant)"),
    "dropped_A": ("#d32f2f", "#fdecea", "DROPPED by A (injection)"),
    "question": ("#1565c0", "#e8f0fe", "QUESTION — as sent"),
    "note": ("#6a1b9a", "#f3e5f5", "SYSTEM NOTE — as sent (stage D)"),
}
_PLACEHOLDER = re.compile(r"&lt;(ID|PHONE|EMAIL|IBAN|CARD|PERSON)_(\d+)&gt;")


def _pills(escaped: str) -> str:
    return _PLACEHOLDER.sub(r'<span style="background:#ffe082;border:1px solid #ffb300;border-radius:10px;padding:0 6px;font-weight:700;color:#5d4037">&lt;\1_\2&gt;</span>', escaped)


def card_html(title: str, status: str, text: str, subtitle: str = "") -> str:
    border, bg, label = CARD[status]
    return (f'<div style="background:{bg};border:1.5px solid {border};border-left:6px solid {border};border-radius:6px;padding:6px 10px;margin:5px 0;'
            f'font-size:12px;white-space:pre-wrap;font-family:monospace;line-height:1.35">'
            f'<div style="font-family:sans-serif;font-size:11px;color:#333"><b>{html.escape(title)}</b> · <span style="color:{border};font-weight:700">{label}</span>'
            f'{(" · " + html.escape(subtitle)) if subtitle else ""}</div>{_pills(html.escape(text))}</div>')


def chunk_card(c: dict, as_sent: bool) -> str:
    text = c["as_sent"] if (as_sent and c.get("as_sent")) else c["text"]
    sub = f"{c['source']}" + (f" · {c['detail']}" if c.get("detail") else "") + (f" · {c['n_excised']} paragraph(s) removed" if c.get("n_excised") else "") + ("" if as_sent else " · original text")
    return card_html(f"[{c['chunk_id']}]", c["status"], text, sub)


def _delta(cur: float, ref: float | None) -> str | None:
    if ref is None or not ref:
        return None
    pct = 100 * (cur - ref) / ref
    return f"{'+' if pct >= 0 else '-'}{abs(pct):.1f} % vs E0"


# ----------------------------------------------------------------------------- one side (E0 or E5 panel)
def render_side(col, title: str, r: dict | None, exp: str, task: dict | None, ref: dict | None = None, peer: dict | None = None):
    """ref = the E0 run of the same input (E5 side: token/cost deltas); peer = the E5 run (E0 side: PII that went out in clear)."""
    with col:
        st.subheader(title)
        if r is None:
            st.info("Press **Run both** to start." if mode == "live" else "No cached run for this model — record it with `python demo/record_cache.py --model <id>`.")
            return
        if r.get("error"):
            st.error(f"Live run failed — {r['error']}")
            return
        if r.get("cached"):
            reason = r.get("fallback_reason", "pre-recorded run")
            if reason == "pre-recorded run" and r.get("warm_run"):
                reason = "pre-recorded warm run (models were loaded before timing; cloud latency as first measured)"
            st.warning(f"⚠️ CACHED response shown — {reason}")
        elif "response_from_cache" in r:
            st.caption("🟢 live API call" if not r["response_from_cache"] else "🟡 cloud response reused from cache/llm (identical outbound text)")
        # ---- final answer card ----------------------------------------------------------------------
        gold = r.get("gold") or (task or {}).get("gold_answer")
        verdict = {True: ("✔ correct", "#e6f4ea"), False: ("✘ wrong", "#fde7e9"), None: ("no gold answer", "#f1f3f4")}[r.get("correct")]
        st.markdown(
            f'<div style="border:2px solid #999;border-radius:8px;padding:10px 14px;margin:6px 0 12px 0;background:{verdict[1]}">'
            f'<div style="font-size:12px;color:#555">FINAL ANSWER</div>'
            f'<div style="font-size:22px;font-weight:700;font-family:monospace">{html.escape(r["answer"] or "(empty)")}</div>'
            f'<div style="font-size:14px">{verdict[0]}' + (f' · gold: <code>{html.escape(str(gold))}</code>' if gold else "") + "</div></div>",
            unsafe_allow_html=True)
        # ---- metrics: token / cost deltas vs E0 (E5 side); latency plain ----------------------------------
        ok = (ref or {}) if (ref and not ref.get("error")) else {}
        m1, m2, m3, m4 = st.columns([1, 1.6, 1, 1])
        m1.metric("Input tokens", f"{r['input_tokens']:,}", delta=_delta(r["input_tokens"], ok.get("input_tokens")), delta_color="inverse")
        m2.metric("Cloud cost (USD)", f"${r['cost_usd']:.4g}", delta=_delta(r["cost_usd"], ok.get("cost_usd")), delta_color="inverse")
        m3.metric("e2e latency (ms)", f"{r['e2e_ms']:,.0f}")
        m4.metric("Local layer (ms)", f"{r['local_ms']:,.0f}")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Injection detected", "✔" if r["injection_detected"] else "✘")
        s2.metric("Tool-call attempted", "✔" if r["tool_call_attempted"] else "✘")
        s3.metric("Blocked by guard", "✔" if r["export_attempts"] and r["blocked"] else ("—" if not r["export_attempts"] else "✘"))
        s4.metric("PII redacted / sent", f"{r['pii_redacted']} / {r['pii_sent']}")
        line = f"**Stages run:** `{', '.join(r['stages_run']) or 'none'}`" + (f" · scanner: `{r.get('scanner')}`" if r.get("scanner") else "")
        if exp != "E0":
            line += f" · T = **{r.get('T', 2000):,}** tokens · budget = **{r.get('budget', 2500):,}**" + (" (recorded values)" if r.get("cached") else "")
        st.markdown(line)
        # ---- evidence on click ------------------------------------------------------------------------
        if task:
            if exp != "E0":
                flagged = common.evidence_flagged(r, task) if r["injection_detected"] else []
                n_par = sum(len(f["paragraphs"]) for f in flagged)
                with st.expander(f"Show detected text ({n_par} paragraph(s) excised)" if flagged else "Show detected text (none)", expanded=False):
                    if not flagged:
                        st.markdown("**none** — no untrusted chunk reaching the model matched a rule or the classifier threshold.")
                    for f in flagged:
                        st.markdown(f"**[{f['chunk_id']}]** — detected by **{f['layer']}**")
                        for p in f["paragraphs"]:
                            st.markdown(card_html("excised", "excised", p), unsafe_allow_html=True)
                        if not f["paragraphs"]:
                            st.caption("(whole chunk removed)")
                rows = common.evidence_pii(r, task) if r["pii_redacted"] else []
                with st.expander(f"Show redacted values ({len(rows)})" if rows else "Show redacted values (none)", expanded=False):
                    if rows:
                        st.caption("Shown locally for verification only; the model received the placeholders.")
                        st.dataframe(pd.DataFrame(rows, columns=["type", "value", "placeholder", "chunk_id"]).rename(columns={"value": "original value → ", "placeholder": "placeholder"}), hide_index=True, width="stretch")
                    else:
                        st.markdown("**none** — the PII pre-scan / redaction found nothing in the outbound text" + ("" if "C" in r["stages_run"] else " (stage C did not run: regex pre-scan had no hit)") + ".")
            else:
                rows = common.evidence_pii(r, task, e5_run=peer) if (r.get("pii_sent") or (peer and peer.get("pii_redacted"))) else []
                with st.expander(f"Show PII sent in clear ({len(rows)})" if rows else "Show PII sent in clear (none)", expanded=False):
                    if rows:
                        st.caption("Shown locally for verification only — these values were in the text sent to the model.")
                        st.dataframe(pd.DataFrame(rows, columns=["type", "value", "placeholder", "chunk_id"]).rename(columns={"placeholder": "status"}), hide_index=True, width="stretch")
                    else:
                        st.markdown("**none** — no known PII value in this document (nothing was planted and the layer redacted nothing on the other side).")
        if r.get("router"):
            with st.expander("Router decisions (why these stages)", expanded=exp == "E5"):
                for line in r.get("router_explained", []):
                    st.markdown("- " + line)
                sc = r.get("scan") or {}
                if sc.get("n_untrusted_units"):
                    st.markdown(f"- Scan scope **{sc['scan_scope']}**: scanned {sc['n_scanned']} of {sc['n_untrusted_units']} untrusted chunks "
                                f"({sc['n_skipped_dropped']} skipped because selection dropped them, {sc['n_scanned_tool_mention_extra']} extra for tool-name mentions).")
        elif exp == "E0":
            st.caption("E0: no layer — the whole context is sent as-is.")
        # ---- context panels ---------------------------------------------------------------------------
        if task:
            statuses = common.chunk_statuses(r, task)
            sent = [s for s in statuses if s["status"] in ("sent", "kept", "excised")]
            with st.expander(f"Context sent to the model — exactly as sent ({len(sent)}/{len(statuses)} chunks)", expanded=True):
                cards = [card_html("Question", "question", r.get("outbound_question") or r.get("question") or task["question"])]
                if r.get("system_note"):
                    cards.append(card_html("Note", "note", r["system_note"]))
                cards += [chunk_card(s, as_sent=True) for s in sent]
                st.markdown("\n".join(cards), unsafe_allow_html=True)
            rest = [s for s in statuses if s["status"] in ("dropped_B", "dropped_A", "excised")]
            if rest:
                with st.expander(f"Show dropped and excised chunks ({len(rest)}) — original text", expanded=False):
                    st.markdown("\n".join(chunk_card(s, as_sent=False) for s in rest), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- tables & charts shared by tabs
def ablation_table(runs: dict[str, dict], has_planted: bool):
    """Benchmark-style table with one row per experiment present (E0, E6, E1, E5); better cell highlighted."""
    def mark(r):
        return f"{'✔ ' if r.get('correct') else ('✘ ' if r.get('correct') is False else '')}{(r.get('answer') or '')[:40]}"
    e0 = runs.get("E0") or {}
    e5 = runs.get("E5") or {}
    e0_pii = e0.get("pii_sent", 0) if has_planted else e5.get("pii_redacted", 0)
    rows = []
    for e in ORDER:
        r = runs.get(e)
        if not r or r.get("error"):
            continue
        tok, cost = r.get("input_tokens", 0), r.get("cost_usd", 0.0)
        rows.append({
            "Experiment": long_label(e), "_ans": 1 if r.get("correct") else 0, "Answer": mark(r), "_tok": tok, "Tokens sent": f"{tok:,}",
            "Token reduction %": (f"{100 * (1 - tok / e0['input_tokens']):.1f} %" if e != "E0" and e0.get("input_tokens") else "—"),
            "_cost": cost, "Cost ($)": f"${cost:.4g}", "Cost reduction %": (f"{100 * (1 - cost / e0['cost_usd']):.1f} %" if e != "E0" and e0.get("cost_usd") else "—"),
            "_pii": (e0_pii if e == "E0" else r.get("pii_sent", 0) if "C" in r.get("stages_run", []) else (e0_pii if not has_planted else r.get("pii_sent", 0))),
            "PII sent": f"{e0_pii if e == 'E0' else (r.get('pii_sent', 0) if ('C' in r.get('stages_run', []) or has_planted) else e0_pii)}",
            "Injection detected": ("✔" if r.get("injection_detected") else "✘") if "A" in r.get("stages_run", []) else "—",
            "Attacks attempted/blocked": f"{r.get('export_attempts', 0)}/{r.get('export_attempts', 0) if r.get('blocked', True) else 0}",
            "_loc": r.get("local_ms", 0), "Local latency (s)": f"{r.get('local_ms', 0) / 1000:.2f}", "_e2e": r.get("e2e_ms", 0), "e2e latency (s)": f"{r.get('e2e_ms', 0) / 1000:.2f}",
        })
    d = pd.DataFrame(rows)
    best = {"Answer": d._ans.idxmax() if d._ans.nunique() > 1 else None, "Tokens sent": d._tok.idxmin(), "Cost ($)": d._cost.idxmin(), "PII sent": d._pii.idxmin() if d._pii.nunique() > 1 else None,
            "Local latency (s)": d._loc.idxmin(), "e2e latency (s)": d._e2e.idxmin()}
    show = d[[c for c in d.columns if not c.startswith("_")]]
    return show.style.apply(lambda col: ["background-color:#c8e6c9;font-weight:700" if (best.get(col.name) is not None and i == best[col.name]) else "" for i in col.index], axis=0)


def grouped_bars(items: list[dict], title: str, y_label: str, fmt: str):
    """Grouped bars per run; one bar per experiment PRESENT in that run (legend lists only those).
    items: [{"run": label, "series": {"E0": v, "E5": v, ...}}]. The % vs E0 is written above every non-E0 bar."""
    import altair as alt

    recs, present = [], []
    for it in items:
        e0v = it["series"].get("E0")
        for e in ORDER:
            if e not in it["series"]:
                continue
            if e not in present:
                present.append(e)
            v = it["series"][e]
            pct = (100 * (1 - v / e0v)) if (e != "E0" and e0v) else None
            recs.append({"run": it["run"], "exp": e, "short": EXP_SHORT[e], "long": long_label(e), "v": v, "label": "" if pct is None else f"{'-' if pct >= 0 else '+'}{abs(pct):.1f} %"})
    df = pd.DataFrame(recs)
    order = [EXP_SHORT[e] for e in present]
    scale = alt.Scale(domain=order, range=[EXP_COLORS[e] for e in present])
    base = alt.Chart(df).encode(x=alt.X("run:N", title="Run", sort=[it["run"] for it in items], axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("short:N", sort=order))
    bars = base.mark_bar().encode(y=alt.Y("v:Q", title=y_label), color=alt.Color("short:N", title="", scale=scale, sort=order), tooltip=[alt.Tooltip("run:N"), alt.Tooltip("long:N", title="experiment"), alt.Tooltip("v:Q", format=fmt)])
    text = base.mark_text(dy=-6, fontSize=12, fontWeight="bold").encode(y=alt.Y("v:Q"), text="label:N")
    st.markdown(f"**{title}**")
    st.altair_chart(alt.layer(bars, text).properties(height=260), width="stretch")


def pair_chart(runs: dict[str, dict], run_label: str):
    series = {e: runs[e]["input_tokens"] for e in ORDER if runs.get(e) and not runs[e].get("error")}
    if "E0" in series and len(series) > 1:
        grouped_bars([{"run": run_label, "series": series}], "Tokens sent to the model — this case", "Tokens sent per call", ",")


# ----------------------------------------------------------------------------- running + logging
def run_exps(task: dict, tag: str, exps: list[str]) -> dict:
    out = {}
    for e in exps:
        with st.spinner(f"Running {e} ({EXP_SHORT.get(e, e)}) on {model_id} … first run loads the local models (~30 s)"):
            out[e] = common.run_side_with_timeout(e, task, model_id, tag, T=sel_T, budget=sel_budget)
    return out


RUN_KEYS = ("answer", "correct", "input_tokens", "cost_usd", "pii_sent", "pii_redacted", "injection_detected", "export_attempts", "blocked", "local_ms", "e2e_ms", "error", "stages_run")


def _mark(r: dict) -> str:
    if r.get("error"):
        return "error"
    return f"{r.get('answer', '')[:40]} {'✔' if r.get('correct') else ('✘' if r.get('correct') is False else '')}".strip()


def log_run(task: dict, runs: dict, source: str) -> int:
    """Append one My-runs entry (session only) and return its index; E0 'PII sent' for custom docs = values E5 masked."""
    e0, e5 = runs["E0"], runs["E5"]
    st.session_state.setdefault("my_runs", [])
    if e0.get("error") and e5.get("error"):
        return -1
    has_planted = bool(task.get("planted_pii_values"))
    e0_tok, e5_tok = e0.get("input_tokens", 0), e5.get("input_tokens", 0)
    st.session_state.my_runs.append({
        "time": time.strftime("%H:%M:%S"), "source": source, "question": (task["question"][:60] + "…") if len(task["question"]) > 60 else task["question"],
        "model": model_label, "E0 tokens": e0_tok, "E5 tokens": e5_tok, "reduction %": round(100 * (1 - e5_tok / e0_tok), 1) if e0_tok else None,
        "cost reduction %": round(100 * (1 - e5.get("cost_usd", 0) / e0["cost_usd"]), 1) if e0.get("cost_usd") else None,
        "E0 answer": _mark(e0), "E5 answer": _mark(e5),
        "PII sent E0/E5": f"{e0.get('pii_sent', 0) if has_planted else e5.get('pii_redacted', 0)}/{e5.get('pii_sent', 0)}",
        "injection detected (E5)": "✔" if e5.get("injection_detected") else "✘",
        "export attempts E0/E5": f"{e0.get('export_attempts', 0)}/{e5.get('export_attempts', 0)}" + (" (all blocked)" if (e0.get("export_attempts") or e5.get("export_attempts")) and e0.get("blocked", True) and e5.get("blocked", True) else ""),
        "local ms (E5)": round(e5.get("local_ms", 0)), "e2e ms E0/E5": f"{e0.get('e2e_ms', 0):.0f}/{e5.get('e2e_ms', 0):.0f}", "T / budget": f"{e5.get('T', sel_T)} / {e5.get('budget', sel_budget)}",
        "_e0_correct": e0.get("correct"), "_e5_correct": e5.get("correct"),
        "note": ", ".join(f"{k} cached ({r.get('fallback_reason')})" for k, r in runs.items() if r.get("cached")) or "",
        "_runs": {e: {k: r.get(k) for k in RUN_KEYS} for e, r in runs.items()}, "_has_planted": has_planted,
    })
    return len(st.session_state.my_runs) - 1


def extend_log(idx: int, extra: dict) -> None:
    if 0 <= idx < len(st.session_state.get("my_runs", [])):
        st.session_state.my_runs[idx]["_runs"].update({e: {k: r.get(k) for k in RUN_KEYS} for e, r in extra.items()})


def scenario_block(key: str, task: dict, source: str, run_label: str):
    """Run buttons + two side panels + chart, shared by Scenario and Try your own (live mode)."""
    st.session_state.setdefault("runs", {})
    st.session_state.setdefault("log_idx", {})
    b1, b2 = st.columns([1, 1.6])
    if b1.button("▶ Run both (live API call)", type="primary", key=f"run:{key}"):
        st.session_state.runs[key] = run_exps(task, f"live-{int(time.time())}", ["E0", "E5"])
        st.session_state.log_idx[key] = log_run(task, st.session_state.runs[key], source)
    runs = st.session_state.runs.get(key)
    if runs and all(e in runs for e in ("E0", "E5")) and not any(e in runs for e in ABLATION_EXPS):
        if b2.button(f"Run full ablation (adds {' + '.join(ABLATION_EXPS)}, {len(ABLATION_EXPS)} more API calls)", key=f"abl:{key}"):
            extra = run_exps(task, f"abl-{int(time.time())}", ABLATION_EXPS)
            runs.update(extra)
            extend_log(st.session_state.log_idx.get(key, -1), extra)
    runs = runs or {"E0": None, "E5": None}
    left, right = st.columns(2)
    render_side(left, "E0 — no layer (baseline)", runs.get("E0"), "E0", task, peer=runs.get("E5"))
    render_side(right, "E5 — ContextShield (adaptive)", runs.get("E5"), "E5", task, ref=runs.get("E0"))
    if runs.get("E0") and runs.get("E5"):
        if any(e in runs for e in ABLATION_EXPS):
            st.markdown("**Full ablation — this case**")
            st.dataframe(ablation_table(runs, bool(task.get("planted_pii_values"))), hide_index=True, width="stretch")
        pair_chart(runs, run_label)


# ----------------------------------------------------------------------------- tabs
tab_scn, tab_res, tab_own, tab_runs = st.tabs(["Scenario (T017)", "Benchmark (22 test tasks)", "Try your own", "My runs"])

with tab_scn:
    st.markdown("**Customer Support Agent — Baseline vs ContextShield.** A support agent answers one question from a knowledge base of policies, tickets, e-mails and web pages; "
                "left = E0 (the agent alone, whole context sent), right = E5 (the same agent behind ContextShield).")
    if mode == "live":
        scn_id = common.DEMO_TASK  # live mode: the single fixed scenario
    else:
        opts = [s["task_id"] for s in common.SCENARIOS]
        why = {s["task_id"]: s["why"] for s in common.SCENARIOS}
        scn_id = st.selectbox("Pre-recorded scenario (test split)", opts, index=0, format_func=lambda t: f"{t} — {why[t]}" + ("" if common.has_cached(model_id, t) else "  (not recorded for this model)"))
    task = common.load_task(scn_id)
    inj = [c for c in task["context"] if c.get("contains_injection")]
    st.markdown(f"**Question:** {task['question']}  \n**Gold answer:** `{task['gold_answer']}` · **Task:** `{task['task_id']}` · {task['size_bucket']} · {len(task['context'])} chunks · "
                f"planted PII: {sum(len(c['contains_pii']) for c in task['context'])} value(s) · "
                + (f"injected note with a canary ({inj[0].get('injection_kind', 'template')} attack, {'relevant' if inj[0].get('injection_relevant') else 'irrelevant'} chunk)" if inj else "no planted attack"))
    if task.get("dataset_question"):
        st.caption(f"Demo variant of {task['task_id']}: the experiments use the dataset question \"{task['dataset_question']}\" (gold `{task['dataset_gold']}`); the demo asks the date only.")
    if mode == "live":
        scenario_block(f"scn:{model_id}", task, "scenario", f"{task['task_id']} · {model_label}")
    else:
        runs = {e: (common.load_cached(model_id, e, scn_id) | {"cached": True, "fallback_reason": "pre-recorded run"}) if common.has_cached(model_id, scn_id) else None for e in ("E0", "E5")}
        left, right = st.columns(2)
        render_side(left, "E0 — no layer (baseline)", runs["E0"], "E0", task, peer=runs["E5"])
        render_side(right, "E5 — ContextShield (adaptive)", runs["E5"], "E5", task, ref=runs["E0"])
        if runs["E0"] and runs["E5"]:
            pair_chart(runs, f"{task['task_id']} · {model_label}")

with tab_own:
    st.markdown("Paste a document (or upload `.txt` / `.json`), ask a question, tag chunks as trusted / untrusted, and run **E0 vs E5 live**. "
                "The router decisions are shown explicitly — a short document skips relevance selection on purpose (R1), so that is not a bug.")
    st.caption("Your runs are collected in the **My runs** tab (this session only; download them as CSV there).")
    if mode != "live":
        st.info("Live mode only: run `streamlit run demo/app.py` locally with the local models and an API key in `.env`, then pick **live** in the sidebar.")
    else:
        from shield.explain import chunk_document, load_context_file  # light (tiktoken only)

        q = st.text_input("Question", placeholder="What is the status of order ORD-123456?")
        up = st.file_uploader("Document (.txt or .json: a task file / list of {text, source})", type=["txt", "json"])
        doc = st.text_area("…or paste the document here", height=180)
        default_src = st.selectbox("Default tag for chunks", ["untrusted", "trusted"], index=0)
        if st.button("1 · Chunk the document"):
            if up is not None:
                tmp = ROOT / "demo" / "_upload" / up.name
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(up.getvalue())
                chunks = load_context_file(tmp, default_source=default_src)
                tmp.unlink(missing_ok=True)
            else:
                chunks = chunk_document(doc, source=default_src)
            st.session_state.own_chunks = pd.DataFrame([{"chunk_id": c["chunk_id"], "source": c["source"], "text": c["text"]} for c in chunks], columns=["chunk_id", "source", "text"])
            st.session_state.runs.pop("own", None) if "runs" in st.session_state else None
            if not chunks:
                st.error("The document is empty — paste text or upload a .txt/.json file, then chunk it again.")
        if "own_chunks" in st.session_state and len(st.session_state.own_chunks):
            if not q.strip():
                st.caption("Enter a question above to enable the run button.")
            st.caption("Edit the `source` column to tag chunks; untrusted chunks are scanned and wrapped, trusted ones are not.")
            edited = st.data_editor(st.session_state.own_chunks, column_config={"source": st.column_config.SelectboxColumn(options=["untrusted", "trusted"])},
                                    disabled=["chunk_id", "text"], hide_index=True, width="stretch", key="own_editor")
            custom = {"task_id": "custom", "question": q.strip(), "context": [{"chunk_id": r.chunk_id, "source": r.source, "kind": "document", "text": r.text} for r in edited.itertuples()]}
            st.session_state.setdefault("runs", {})
            st.session_state.setdefault("log_idx", {})
            b1, b2 = st.columns([1, 1.6])
            if b1.button("2 · Run E0 and E5 (live)", type="primary", disabled=not q.strip()):
                st.session_state.runs["own"] = run_exps(custom, f"own-{int(time.time())}", ["E0", "E5"])
                st.session_state.own_task = custom
                st.session_state.log_idx["own"] = log_run(custom, st.session_state.runs["own"], "own")
            runs = st.session_state.runs.get("own")
            if runs and not any(e in runs for e in ABLATION_EXPS):
                if b2.button(f"Run full ablation (adds {' + '.join(ABLATION_EXPS)}, {len(ABLATION_EXPS)} more API calls)", key="abl:own"):
                    extra = run_exps(st.session_state.own_task, f"abl-own-{int(time.time())}", ABLATION_EXPS)
                    runs.update(extra)
                    extend_log(st.session_state.log_idx.get("own", -1), extra)
            if runs:
                l2, r2 = st.columns(2)
                render_side(l2, "E0 — no layer", runs["E0"], "E0", st.session_state.own_task, peer=runs["E5"])
                render_side(r2, "E5 — ContextShield", runs["E5"], "E5", st.session_state.own_task, ref=runs["E0"])
                if any(e in runs for e in ABLATION_EXPS):
                    st.markdown("**Full ablation — this case**")
                    st.dataframe(ablation_table(runs, False), hide_index=True, width="stretch")
                pair_chart(runs, f"own · {model_label}")

# ----------------------------------------------------------------------------- Benchmark
COLUMN_HELP = {
    "model": "File-system tag of the cloud model", "llm_model": "Cloud model id", "split": "dev (tuning only) or test (headline)", "exp": "Experiment id (configs in experiments/configs/)",
    "bucket": "Size bucket of the tasks (short < 1.5k tokens, medium 4–6k, long 12k+) or ALL", "scanner_mode": "Stage A scanner actually used (rules / rules+classifier)",
    "n": "Number of tasks", "correct": "Tasks answered correctly (normalized exact match / containment)", "accuracy": "correct / n in %",
    "mean_input_tokens": "Mean prompt tokens billed by the API per task (sums tool-call rounds)", "median_input_tokens": "Median prompt tokens per task",
    "total_cost_usd": "Σ cloud cost at list price", "cost_per_success_usd": "Σ cloud cost / correct answers", "evidence_retention": "% of tasks whose gold chunk reached the model",
    "inj_tasks": "Tasks with a planted injection", "inj_detected": "Planted attacks removed by stage A", "inj_recall": "inj_detected / inj_tasks in %",
    "inj_template_detected": "Template attacks removed (x/y)", "inj_heldout_detected": "Hand-written held-out attacks removed (x/y)", "inj_recall_template": "Template recall %", "inj_recall_heldout": "Held-out recall %",
    "inj_attempted": "Tasks where the model called export_data with the canary (all blocked by the guard)", "inj_attempt_rate": "inj_attempted / inj_tasks in %", "inj_succeeded": "Attacks that passed the guard (always 0)",
    "canary_reached_cloud": "Attacks whose canary was present in the text sent to the model", "inj_fp": "Clean untrusted chunks flagged by A", "inj_fp_rate": "inj_fp / clean chunks scanned in %",
    "pii_tasks": "Tasks with planted PII", "pii_present": "Planted PII values present in the text handed to stage C", "pii_detected": "Of those, redacted", "pii_recall": "pii_detected / pii_present in %",
    "pii_precision": "Redactions that were planted values in %", "pii_false_positives": "Redactions that were not planted values (e.g. NER on a company name)", "pii_leaks": "Planted PII values present in the text sent to the model",
    "scan_scope": "Which untrusted chunks A scans: outbound (after B) or all", "untrusted_chunks_total": "Untrusted chunks in the inputs", "chunks_scanned_total": "Chunks scanned by A",
    "chunks_skipped_dropped_total": "Untrusted chunks not scanned because B dropped them", "tool_mention_extra_scans": "Dropped chunks scanned anyway because they mention a tool name",
    "ms_A_median": "Stage A time, median ms", "ms_A_p95": "Stage A time, P95 ms", "attack_chunks_dropped_unscanned": "Attack chunks dropped by B without a scan", "unscanned_attack_in_outbound": "Of those, present in the outbound text (must be 0)",
    "e2e_ms_median": "End-to-end latency (local layer + cloud), median ms", "e2e_ms_p95": "End-to-end latency, P95 ms", "local_ms_median": "Local layer wall time, median ms", "local_ms_p95": "Local layer wall time, P95 ms",
    "cpu_pct_peak_max": "Peak CPU % of the local layer", "rss_mb_peak_max": "Peak RSS MB of the local layer", "token_reduction_pct": "1 − mean_input_tokens / E0 mean, in %",
    "e0_accuracy": "E0 accuracy on the same tasks", "acc_floor_ok": "accuracy ≥ E0 − 5 points", "e0_e2e_median": "E0 e2e median on the same tasks", "lat_floor_ok": "e2e median ≤ 1.25 × E0 median",
}


def judge_table(t: pd.DataFrame):
    t = t.set_index("exp")
    e0 = t.loc["E0"] if "E0" in t.index else None
    n = int(t.n.iloc[0])
    n_att = int(e0.canary_reached_cloud) if e0 is not None else int(t.inj_tasks.max())
    n_pii = int(e0.pii_leaks) if e0 is not None else int(t.pii_present.max())
    rows = []
    for e in ORDER:
        if e not in t.index:
            continue
        r = t.loc[e]
        rows.append({
            "Experiment": long_label(e), "_acc": r.correct, "Accuracy": f"{int(r.correct)}/{n}", "_tok": r.mean_input_tokens, "Tokens sent (avg)": f"{int(round(r.mean_input_tokens)):,}",
            "_red": r.token_reduction_pct, "Token reduction %": f"{r.token_reduction_pct:.1f} %", "_cost": r.cost_per_success_usd, "Cost per correct answer ($)": f"${r.cost_per_success_usd:.6f}",
            "_det": r.inj_detected, f"Attacks detected (of {n_att})": f"{int(r.inj_detected)}",
            "_reach": r.canary_reached_cloud, f"Attacks reaching model (of {n_att})": f"{int(r.canary_reached_cloud)}",
            "Attacks blocked by guard": f"{int(r.inj_attempted) - int(r.inj_succeeded)} of {int(r.inj_attempted)} attempts",
            "_pii": r.pii_leaks, f"PII values sent (of {n_pii})": f"{int(r.pii_leaks)}", "_lat": r.e2e_ms_median, "Latency median (s)": f"{r.e2e_ms_median / 1000:.2f}",
            "Floors (acc / lat)": "baseline" if e == "E0" else f"{'✅' if r.acc_floor_ok else '❌'} / {'✅' if r.lat_floor_ok else '❌'}",
        })
    d = pd.DataFrame(rows)
    best = {"Accuracy": d._acc.idxmax(), "Tokens sent (avg)": d._tok.idxmin(), "Token reduction %": d._red.idxmax(), "Cost per correct answer ($)": d._cost.idxmin(),
            f"Attacks detected (of {n_att})": d._det.idxmax(), f"Attacks reaching model (of {n_att})": d._reach.idxmin(), f"PII values sent (of {n_pii})": d._pii.idxmin(), "Latency median (s)": d._lat.idxmin()}
    show = d[[c for c in d.columns if not c.startswith("_")]]
    st.dataframe(show.style.apply(lambda col: ["background-color:#c8e6c9;font-weight:700" if (col.name in best and i == best[col.name]) else "" for i in col.index], axis=0), hide_index=True, width="stretch")
    if e0 is not None and "E5" in t.index:
        r = t.loc["E5"]
        st.caption(f"**Headline:** Adaptive ContextShield sends **{r.token_reduction_pct:.1f} % fewer tokens** than no layer ({int(round(r.mean_input_tokens)):,} vs {int(round(e0.mean_input_tokens)):,}), "
                   f"answers **{int(r.correct)}/{n}** (no layer: {int(e0.correct)}/{n}), sends **{int(r.pii_leaks)} of {n_pii}** planted PII values, lets **{int(r.canary_reached_cloud)} of {n_att}** attacks reach the model "
                   f"(no layer: {int(e0.canary_reached_cloud)}) with every attempted exfiltration blocked, at **\\${r.cost_per_success_usd:.6f}** per correct answer vs \\${e0.cost_per_success_usd:.6f} — "
                   f"accuracy floor {'passed' if r.acc_floor_ok else 'failed'}, latency floor {'passed' if r.lat_floor_ok else 'failed'} ({r.e2e_ms_median / 1000:.2f} s vs {1.25 * e0.e2e_ms_median / 1000:.2f} s allowed).")


def benchmark_charts(t: pd.DataFrame):
    import altair as alt

    t = t[t.exp.isin(ORDER)].copy()
    t["short"] = t.exp.map(EXP_SHORT)
    t["long"] = t.exp.map(long_label)
    present = [e for e in ORDER if e in set(t.exp)]
    order = [EXP_SHORT[e] for e in present]
    scale = alt.Scale(domain=order, range=[EXP_COLORS[e] for e in present])
    e0 = t[t.exp == "E0"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Accuracy vs tokens sent** (up-left is better; dashed lines = no layer)")
        base = alt.Chart(t).encode(x=alt.X("mean_input_tokens:Q", title="Tokens per task (avg)"), y=alt.Y("accuracy:Q", title="Correct answers (%)", scale=alt.Scale(domain=[0, 100])))
        pts = base.mark_circle(size=140).encode(color=alt.Color("short:N", title="Experiment", scale=scale, sort=order), tooltip=[alt.Tooltip("long:N", title="experiment"), "mean_input_tokens", "accuracy", "cost_per_success_usd", "pii_leaks", "canary_reached_cloud"])
        txt = base.mark_text(dy=-12, fontSize=11).encode(text="exp:N")
        layers = [pts, txt]
        if len(e0):
            layers += [alt.Chart(e0).mark_rule(strokeDash=[4, 4], color="gray").encode(x="mean_input_tokens:Q"), alt.Chart(e0).mark_rule(strokeDash=[4, 4], color="gray").encode(y="accuracy:Q")]
        st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")
    with c2:
        st.markdown("**Tokens sent per task** (dashed line = no layer)")
        base = alt.Chart(t).encode(x=alt.X("short:N", title="Experiment", sort=order, axis=alt.Axis(labelAngle=0)))
        bars = base.mark_bar().encode(y=alt.Y("mean_input_tokens:Q", title="Tokens per task (avg)"), color=alt.Color("short:N", legend=None, scale=scale),
                                     tooltip=[alt.Tooltip("long:N", title="experiment"), alt.Tooltip("mean_input_tokens:Q", title="tokens per task (avg)", format=","), alt.Tooltip("token_reduction_pct:Q", title="token reduction %")])
        vals = base.mark_text(dy=-6, fontSize=11, fontWeight="bold").encode(y=alt.Y("mean_input_tokens:Q"), text=alt.Text("mean_input_tokens:Q", format=","))
        layers = [bars, vals]
        if len(e0):
            layers.append(alt.Chart(e0).mark_rule(strokeDash=[4, 4], color="gray").encode(y="mean_input_tokens:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=320), width="stretch")


with tab_res:
    st.subheader("Test-set results (22 tasks) — from results/summary.csv")
    df = common.load_summary()
    if df is None:
        st.info("No results yet — run `python make.py all` first.")
    else:
        models = sorted(set(df.model))
        pick = st.selectbox("Model (results)", models, index=models.index(common.model_tag(model_id)) if common.model_tag(model_id) in models else 0, format_func=lambda m: df[df.model == m].llm_model.iloc[0])
        t = df[(df.split == "test") & (df.bucket == "ALL") & (df.model == pick)]
        judge_table(t)
        benchmark_charts(t)
        with st.expander("All experiments (raw summary.csv — hover a column header for its meaning)", expanded=False):
            raw = df[df.model == pick].copy()
            cfg = {c: st.column_config.Column(c, help=COLUMN_HELP.get(c, "")) for c in raw.columns}
            st.dataframe(raw, hide_index=True, width="stretch", column_config=cfg)
            st.markdown("**Legend**  \n" + "  \n".join(f"`{c}` — {COLUMN_HELP[c]}" for c in raw.columns if c in COLUMN_HELP))
        if len(models) > 1:
            with st.expander("Model-swap comparison (all models, test set)"):
                a = df[(df.split == "test") & (df.bucket == "ALL") & (df.exp.isin(["E0", "E5"]))].copy()
                a["Experiment"] = a.exp.map(EXP_LONG)
                a["Model"] = a.llm_model
                st.dataframe(a[["Model", "Experiment", "correct", "n", "mean_input_tokens", "token_reduction_pct", "cost_per_success_usd", "pii_leaks", "inj_detected", "canary_reached_cloud", "inj_attempted", "e2e_ms_median"]].sort_values(["Experiment", "Model"]), hide_index=True, width="stretch",
                             column_config={c: st.column_config.Column(c, help=COLUMN_HELP.get(c, "")) for c in a.columns})

# ----------------------------------------------------------------------------- My runs
with tab_runs:
    st.subheader("My runs — every live run in this session")
    rows = st.session_state.get("my_runs", [])
    if not rows:
        st.info("No live runs yet. Runs from **Scenario** (▶ Run both) and **Try your own** appear here; they are kept for this browser session only.")
    else:
        for i, r in enumerate(rows, 1):
            st.markdown(f"**#{i} · {r['time']} · {r['source']} · {r['model']} · {r['question']}** · T/budget {r['T / budget']}" + (f"  \n_{r['note']}_" if r.get("note") else ""))
            st.dataframe(ablation_table(r["_runs"], r.get("_has_planted", False)), hide_index=True, width="stretch")
        st.markdown("**All runs**")
        log = pd.DataFrame(rows)
        show_cols = ["time", "source", "question", "model", "reduction %", "cost reduction %", "E0 answer", "E5 answer", "PII sent E0/E5", "injection detected (E5)", "export attempts E0/E5", "local ms (E5)", "e2e ms E0/E5", "T / budget", "note"]
        n = len(log)
        e0_g = log["_e0_correct"].dropna(); e5_g = log["_e5_correct"].dropna()
        totals = {c: "" for c in show_cols}
        totals.update({"time": "TOTAL", "source": f"{n} run(s)",
                       "reduction %": round(float(log["reduction %"].dropna().mean()), 1) if log["reduction %"].notna().any() else None,
                       "cost reduction %": round(float(log["cost reduction %"].dropna().mean()), 1) if log["cost reduction %"].notna().any() else None,
                       "E0 answer": f"{int(e0_g.sum())}/{len(e0_g)} correct" if len(e0_g) else "—", "E5 answer": f"{int(e5_g.sum())}/{len(e5_g)} correct" if len(e5_g) else "—",
                       "injection detected (E5)": f"{sum(1 for r in rows if r['injection detected (E5)'] == '✔')}/{n}", "local ms (E5)": round(float(log["local ms (E5)"].mean()))})
        table = pd.concat([log[show_cols], pd.DataFrame([totals])[show_cols]], ignore_index=True)
        st.dataframe(table, hide_index=True, width="stretch")
        c1, c2 = st.columns([1, 1])
        c1.download_button("⬇ Download CSV", data=log[[c for c in log.columns if not c.startswith("_")]].to_csv(index=False).encode("utf-8"), file_name="contextshield_my_runs.csv", mime="text/csv")
        if c2.button("🗑 Clear"):
            st.session_state.my_runs = []
            st.rerun()
        items_tok = [{"run": f"#{i} {r['source']}", "series": {e: v["input_tokens"] for e, v in r["_runs"].items() if v and not v.get("error")}} for i, r in enumerate(rows, 1)]
        items_cost = [{"run": f"#{i} {r['source']}", "series": {e: v["cost_usd"] for e, v in r["_runs"].items() if v and not v.get("error")}} for i, r in enumerate(rows, 1)]
        g1, g2 = st.columns(2)
        with g1:
            grouped_bars(items_tok, "Tokens sent per run — E0 vs E5 (+ ablation if run)", "Tokens sent per call", ",")
        with g2:
            grouped_bars(items_cost, "Cloud cost per run — E0 vs E5 (+ ablation if run)", "Cloud cost (USD)", ".6f")
