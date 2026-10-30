"""Record the pre-recorded scenarios (demo.common.SCENARIOS × models) into demo/cached/<model_tag>/<task>/{E0,E5}.json.
  python demo/record_cache.py [--model openai/gpt-4o-mini] [--all] [--tasks T017,T019]

Each side is run twice: the first pass loads the local models, the second — warm — pass is recorded, so cached
local/e2e latency reflects a warm run. Cloud responses come from the content-keyed cache when the outbound text
is identical to an experiment run (it is, for the dataset question); only the T017 demo question needed live calls.
Originals (excised text, redaction map, raw E0 context) are never written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, SCENARIOS, load_task, run_side, save_cached  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELS["gpt-4o-mini"][0])
    ap.add_argument("--all", action="store_true", help="record every model in demo.common.MODELS")
    ap.add_argument("--tasks", default=",".join(s["task_id"] for s in SCENARIOS))
    a = ap.parse_args()
    for mid in ([m[0] for m in MODELS.values()] if a.all else [a.model]):
        for tid in [t.strip() for t in a.tasks.split(",") if t.strip()]:
            task = load_task(tid)
            for exp in ("E0", "E5"):
                run_side(exp, task, mid, live_tag=None)  # warm-up (loads models; response cached)
                r = run_side(exp, task, mid, live_tag=None)  # warm pass: recorded
                r["warm_run"] = True
                save_cached(mid, exp, r, task_id=tid)
                print(f"{mid} {tid} {exp}: answer={r['answer']!r} correct={r['correct']} tokens={r['input_tokens']} attempted={r['tool_call_attempted']} "
                      f"pii_sent={r['pii_sent']} inj_detected={r['injection_detected']} local={r['local_ms']:.0f}ms e2e={r['e2e_ms']:.0f}ms {'(cloud response reused)' if r.get('response_from_cache') else '(live call)'}")
    print("Saved under demo/cached/<model_tag>/<task>/")
