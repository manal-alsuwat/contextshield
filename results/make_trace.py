"""Build results/trace_<task>.md: one fully worked trace from results/<exp>/<task>.json.

  python results/make_trace.py --task T017 --exp E5
Shows: chunks in, A verdicts, B scores/selection, merge, redaction counts,
text actually sent, tokens, answer, correctness. Planted PII values are never
printed (the stored outbound text is already redacted for C-experiments).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T017")
    ap.add_argument("--exp", default="E5")
    ap.add_argument("--model", default=None, help="model tag under results/runs/ (default: LLM_MODEL from .env)")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from agent.llm_client import model_tag

    tag = args.model or model_tag()
    run_dir = ROOT / "results" / "runs" / tag
    task = None
    for s in ("dev", "test"):
        f = ROOT / "data" / s / f"{args.task}.json"
        if f.exists():
            task = json.loads(f.read_text(encoding="utf-8"))
    r = json.loads((run_dir / args.exp / f"{args.task}.json").read_text(encoding="utf-8"))
    e0f = run_dir / "E0" / f"{args.task}.json"
    e0 = json.loads(e0f.read_text(encoding="utf-8")) if e0f.exists() else None
    assert task, "task not found"
    if "C" not in r["stages_run"]:
        raise SystemExit("trace must come from an experiment where C ran (outbound text is redacted)")

    L = [f"# Trace — {args.task} under {args.exp} (model `{r.get('llm_model', tag)}`, scanner: {r.get('scanner_mode')})", ""]
    L += [f"**Question:** {task['question']}", f"**Gold answer:** `{task['gold_answer']}`  ·  **Bucket:** {task['size_bucket']}  ·  **Chunks in:** {len(task['context'])}", ""]
    L += ["_The live demo (`demo/app.py`) runs this same task with the narrower demo question \"What is the completion date of refund request RF-88213?\" (gold `2026-03-14`); the experiment numbers below use the dataset question above. The layer's decisions (router, selection, scan, redaction) do not depend on the question wording except through the must-keep entity `RF-88213`, which both share._", ""]
    L += ["## 1. Input", "", "| chunk | source | kind | tokens | gold | planted PII types | injection |", "|---|---|---|---|---|---|---|"]
    tok = {x["chunk_id"]: x["tokens"] for x in r["relevance_log"]}
    for c in task["context"]:
        L.append(f"| {c['chunk_id']} | {c['source']} | {c['kind']} | {tok.get(c['chunk_id'], '—')} | {'✔' if c['chunk_id'] in task['gold_chunk_ids'] else ''} | {', '.join(c['contains_pii']) or ''} | {('✔ ' + ('relevant' if c.get('injection_relevant') else 'irrelevant') + ', ' + c.get('injection_position', '')) if c['contains_injection'] else ''} |")
    L += ["", "## 2. Router", "", f"`{json.dumps(r['router_decisions'])}`" if r.get("router_decisions") else "_(fixed pipeline)_", "", f"Stages run: **{', '.join(r['stages_run'])}**", ""]
    sc = r.get("scan_scope")
    L += ["## 3. Stage A — injection scan (untrusted chunks)", ""]
    if sc:
        L += [f"Scanner scope **`{sc}`**: untrusted chunks {r.get('n_untrusted_units')} → scanned {r.get('n_scanned')} ({r.get('n_scanned_selected')} selected/must-kept by B + {r.get('n_scanned_tool_mention_extra')} tool-name extras), skipped because B dropped them {r.get('n_skipped_dropped')}; classifier sequences {r.get('clf_sequences')}; A time {r['local_ms_A']} ms.", ""]
    L += ["| chunk | rule hits | classifier score | decision | ms |", "|---|---|---|---|---|"]
    for v in r["injection_log"]:
        L.append(f"| {v['chunk_id']} | {', '.join(v['rule_hit']) or '—'} | {v['clf_score'] if v['clf_score'] is not None else '—'} | {v['decision']} | {v['latency_ms']} |")
    L += ["", f"Removals: `{json.dumps(r['injection_removed'])}`", ""]
    L += ["## 4. Stage B — relevance selection", "", f"Entities from question: `{r['relevance_info'].get('entities')}`  ·  budget {r['relevance_info'].get('budget_tokens')} tokens  ·  selected {r['relevance_info'].get('n_selected')}/{r['relevance_info'].get('n_units')} chunks, {r['relevance_info'].get('selected_tokens')} tokens", "", "| chunk | cosine | tokens | selected | must-keep reason |", "|---|---|---|---|---|"]
    for x in sorted(r["relevance_log"], key=lambda x: -x["score"]):
        L.append(f"| {x['chunk_id']} | {x['score']} | {x['tokens']} | {'✔' if x['selected'] else ''} | {x['must_keep_reason'] or ''} |")
    L += ["", "## 5. Merge → Stage C — PII redaction", "", f"Redactions by type: `{json.dumps(r['pii_redactions'])}`  ·  planted values present in kept text: {r['pii_present_pre_C']}  ·  detected: {r['pii_detected']}  ·  false positives: {r['pii_false_positive']}  ·  **sent to cloud: {r['pii_sent_to_cloud']}**", ""]
    L += ["## 6. Text actually sent to the cloud model", "", "```text", r["system_note"], "", f"Question: {r['outbound_question']}", "", "Context:", r["outbound_context"], "```", ""]
    L += ["## 7. Outcome", "", "| metric | value |", "|---|---|"]
    rows = [("input tokens (API usage)", r["input_tokens"]), ("output tokens", r["output_tokens"]), ("local tokens in → out", f"{r['input_tokens_local']} → {r['outbound_tokens_local']}"),
            ("cloud cost (USD)", r["cloud_cost_usd"]), ("answer", f"`{r['answer']}`"), ("correct", r["answer_correct"]), ("evidence retained", r["evidence_retained"]),
            ("injection detected", r["injection_detected"]), ("tool call attempted", r["tool_call_attempted"]), ("attack succeeded", r["attack_succeeded"]), ("canary reached cloud", r["canary_in_outbound"]),
            ("local ms A / B / C / wall", f"{r['local_ms_A']} / {r['local_ms_B']} / {r['local_ms_C']} / {r['local_ms_total_wall']}"), ("cloud ms", r["cloud_ms"]), ("e2e ms", r["e2e_ms"]),
            ("CPU peak % / RSS peak MB", f"{r['cpu_pct_peak']} / {r['rss_mb_peak']}")]
    if e0:
        rows += [("— E0 input tokens", e0["input_tokens"]), ("— E0 answer / correct", f"`{e0['answer']}` / {e0['answer_correct']}"), ("— E0 tool call attempted", e0["tool_call_attempted"]), ("— E0 PII values sent to cloud", e0["pii_sent_to_cloud"]), ("— E0 e2e ms", e0["e2e_ms"])]
    L += [f"| {k} | {v} |" for k, v in rows]
    out = ROOT / "results" / f"trace_{args.task}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
