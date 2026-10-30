"""Leak check: grep logs/, results/, demo/cached/ and captured stdout for planted PII values.

Scope: result files of experiments where stage C is part of the pipeline
(E2, E4, E5*) must never contain a planted value — the outbound text stored
there is what was sent to the cloud. E0/E1/E6/E3 send raw context by design,
and their `pii_sent_to_cloud` metric counts those values; they are reported,
not treated as a leak of the layer. logs/ and demo/cached/ are always in scope.
The dataset under data/ is excluded (it legitimately holds the planted values).

Exit code 1 on any hit.  python results/leak_check.py [--stdout file]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_EXPS = ("E2", "E4", "E5")


def planted_values() -> dict[str, list[str]]:
    vals = {}
    for s in ("dev", "test"):
        for f in (ROOT / "data" / s).glob("T*.json"):
            t = json.loads(f.read_text(encoding="utf-8"))
            vals[t["task_id"]] = t["planted_pii_values"]
    return vals


def variants(v: str) -> set[str]:
    return {v, v.replace(" ", ""), v.replace(" ", "-")}


def scan_file(path: Path, values: list[str]) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return [v for v in values if any(x in text for x in variants(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", help="captured stdout file to scan")
    args = ap.parse_args()
    vals = planted_values()
    all_vals = sorted({v for vs in vals.values() for v in vs})
    hits = []
    files: list[Path] = []
    for d in ("logs", "demo/cached"):
        p = ROOT / d
        if p.exists():
            files += [f for f in p.rglob("*") if f.is_file()]
    for base in (ROOT / "results" / "runs", ROOT / "results" / "_dry"):
        if not base.exists():
            continue
        for mdir in base.iterdir():  # one directory per LLM model tag
            if not mdir.is_dir():
                continue
            for d in mdir.iterdir():
                if d.is_dir() and d.name.startswith(C_EXPS):
                    files += list(d.glob("*.json"))
    for name in ("summary.csv", "REPORT.md", "per_task.csv", "trace_T017.md"):
        if (ROOT / "results" / name).exists():
            files.append(ROOT / "results" / name)
    if (ROOT / "README.md").exists():
        files.append(ROOT / "README.md")
    if args.stdout:
        files.append(Path(args.stdout))
    for f in files:
        found = scan_file(f, all_vals)
        if found:
            hits.append((f, len(found)))
    print(f"Leak check: {len(files)} files scanned against {len(all_vals)} planted values.")
    if hits:
        for f, n in hits:
            print(f"  LEAK: {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f} ({n} planted value(s))")
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS — no planted PII value found in logs/, results/ (C-experiments), demo/cached/, README.")


if __name__ == "__main__":
    main()
