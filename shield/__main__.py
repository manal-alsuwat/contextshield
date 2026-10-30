"""CLI:  python -m shield run --question "..." --context file.txt [--source trusted|untrusted] [--call] [--T 2000] [--budget 2500]

Runs the official E5 configuration (router, B → A on outbound, C, D) on your own document,
prints the router decisions in plain words, what was kept / excised / redacted, and the text
that would be sent. With --call it also queries the configured model through E0 (no layer)
and E5 and prints both answers. --json prints the machine-readable result instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .explain import chunk_document, explain_router, load_context_file


def _official_e5(T: int, budget: int):
    import yaml

    from .pipeline import ContextShield, ShieldConfig

    cfg_file = Path(__file__).resolve().parent.parent / "experiments" / "configs" / "E5.yaml"
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
    sc = ShieldConfig(name="E5", mode="router", T=T, budget=budget, injection_mode=cfg.get("injection_mode", "auto"),
                      injection_threshold=cfg.get("injection_threshold", 0.9), injection_model=cfg.get("injection_model", "small"),
                      rule_action=cfg.get("rule_action", "excise"), scan_scope=cfg.get("scan_scope", "outbound"))
    return ContextShield(sc)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m shield", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the layer on a question + document")
    r.add_argument("--question", required=True)
    r.add_argument("--context", required=True, help=".txt (chunked automatically) or .json (task file / list of {text, source})")
    r.add_argument("--source", default="untrusted", choices=["trusted", "untrusted"], help="tag for chunks without one (default untrusted)")
    r.add_argument("--T", type=int, default=2000)
    r.add_argument("--budget", type=int, default=2500)
    r.add_argument("--call", action="store_true", help="also call the configured model via E0 and E5 and print both answers")
    r.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    chunks = load_context_file(args.context, default_source=args.source)
    shield = _official_e5(args.T, args.budget)
    res = shield.process(args.question, chunks)
    out = {
        "question": args.question, "n_chunks": len(chunks), "input_tokens": res.input_tokens_local, "outbound_tokens": res.outbound_tokens_local,
        "router": res.router, "router_explained": explain_router(res.router, args.T), "stages_run": res.stages_run,
        "scanner": shield.scanner_info, "scan": res.scan, "kept_chunks": [c.get("parent_id", c["chunk_id"]) for c in res.outbound_chunks],
        "injection_log": res.injection_log, "injection_removed": res.injection_removed, "relevance": res.relevance_log,
        "pii_redactions": res.pii_counts, "timings_ms": res.timings, "outbound_question": res.outbound_question,
        "outbound_context": res.outbound_context, "system_note": res.system_note,
    }
    if args.call:
        from agent.agent import run_agent
        from agent.llm_client import LLMClient
        from .wrap import render_context

        client = LLMClient()
        if not client.configured():
            print("LLM not configured (.env) — skipping --call", file=sys.stderr)
        else:
            e0 = run_agent(client, args.question, render_context(chunks, wrap_untrusted=False), "", {"cli": "E0"}, "CLI")
            e5 = run_agent(client, res.outbound_question, res.outbound_context, res.system_note, {"cli": "E5"}, "CLI")
            out["answers"] = {
                "E0": {"answer": e0.answer, "input_tokens": e0.usage.get("prompt_tokens"), "export_attempts": len(e0.tool_attempts), "blocked": all(a["blocked"] for a in e0.tool_attempts)},
                "E5": {"answer": e5.answer, "input_tokens": e5.usage.get("prompt_tokens"), "export_attempts": len(e5.tool_attempts), "blocked": all(a["blocked"] for a in e5.tool_attempts)},
            }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    print(f"Question: {args.question}")
    delta = 100 * (1 - res.outbound_tokens_local / max(1, res.input_tokens_local))
    print(f"Document: {len(chunks)} chunk(s), {res.input_tokens_local:,} tokens → outbound {res.outbound_tokens_local:,} tokens "
          + (f"({delta:.0f}% less)" if delta >= 0 else f"(+{-delta:.0f}%: on a short document the wrapper tags and note add more tokens than selection removes)"))
    print("\nRouter decisions:")
    for line in out["router_explained"]:
        print("  •", line)
    print(f"Stages run: {', '.join(res.stages_run) or 'none'} | scanner: {shield.scanner_info['mode_used']}" + (f" (fallback: {shield.scanner_info['fallback_reason']})" if shield.scanner_info.get("fallback_reason") else ""))
    if res.scan.get("n_untrusted_units"):
        s = res.scan
        print(f"Scan scope {s['scan_scope']}: scanned {s['n_scanned']} of {s['n_untrusted_units']} untrusted chunks ({s['n_skipped_dropped']} skipped because selection dropped them, {s['n_scanned_tool_mention_extra']} extra for tool-name mentions)")
    kept = set(out["kept_chunks"])
    removed = {x["chunk_id"].split(".")[0]: x for x in res.injection_removed}
    print("\nChunks:")
    for c in chunks:
        cid = c["chunk_id"]
        status = "KEPT" if cid in kept else ("DROPPED (injection)" if cid in removed and removed[cid]["action"].startswith("dropped") else "DROPPED (not relevant)")
        if cid in removed and cid in kept:
            status = f"KEPT, {removed[cid]['n_paragraphs']} paragraph(s) excised ({removed[cid]['decision']})"
        print(f"  [{cid}] {c['source']:9} {status}")
    if sum(res.pii_counts.values()):
        print("PII redacted:", ", ".join(f"{k}={v}" for k, v in res.pii_counts.items() if v))
    print(f"Local time: {res.timings['local_ms_total_wall']:.0f} ms (A {res.timings['local_ms_A']:.0f}, B {res.timings['local_ms_B']:.0f}, C {res.timings['local_ms_C']:.0f})")
    print("\n--- text that would be sent ---")
    if res.system_note:
        print(res.system_note, "\n")
    print(f"Question: {res.outbound_question}\n\nContext:\n{res.outbound_context}")
    if "answers" in out:
        print("\n--- answers ---")
        for k, v in out["answers"].items():
            print(f"{k}: {v['answer']!r} | input tokens {v['input_tokens']} | export_data attempts {v['export_attempts']}" + (" (blocked)" if v["export_attempts"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
