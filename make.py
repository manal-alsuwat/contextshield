"""Cross-platform `make` (Windows has no make).  python make.py <target> [<target> ...]

Targets:
  setup     pip install the pinned dependencies (torch CPU wheels, then requirements.txt)
  check     unit tests (no API) + leak check
  dataset   generate data/dev + data/test (seed=42)
  bench     model benchmarks -> bench/BENCH.md
  E0 E1 E6 E4 E5 E5-all E2 E3 E5T   run one experiment on dev then test (E5T = E5-T1000 + E5-T4000)
  core      E0 E1 E6 E4 E5 E5-all
  optional  E2 E3 E5T
  summary   results/summary.csv + REPORT.md + chart
  trace     results/trace_T017.md
  leak      leak check
  test      unit tests
  all       dataset bench core summary trace leak
  dry       dataset + every experiment with --dry-run (no API) + leak  (smoke test)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}  # Windows console is cp1252 by default


def run(*args: str) -> None:
    print(f"\n>>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT, env=ENV)
    if r.returncode != 0:
        print(f"!!! step failed with exit code {r.returncode}")
        sys.exit(r.returncode)


def exp(name: str, extra: list[str] | None = None) -> None:
    extra = extra or []
    run("experiments/run_experiment.py", "--exp", name, "--split", "dev", *extra)
    run("experiments/run_experiment.py", "--exp", name, "--split", "test", *extra)


def setup() -> None:
    """Install the pinned dependencies into the current interpreter (torch CPU wheels first)."""
    run("-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.15.0")
    run("-m", "pip", "install", "-r", "requirements.txt")


def check() -> None:
    """Fast verification without API calls: unit tests (live demo test skipped) + leak check."""
    run("-m", "pytest", "-q", "tests", "--ignore=tests/test_demo_live.py")
    run("results/leak_check.py")


TARGETS = {
    "setup": setup,
    "check": check,
    "dataset": lambda: run("data/gen_dataset.py"),
    "bench": lambda: run("bench/bench_models.py"),
    "E0": lambda: exp("E0"), "E1": lambda: exp("E1"), "E6": lambda: exp("E6"), "E4": lambda: exp("E4"), "E5": lambda: exp("E5"),
    "E2": lambda: exp("E2"), "E3": lambda: exp("E3"),
    "E5T": lambda: (exp("E5-T1000"), exp("E5-T4000")),
    "summary": lambda: run("results/summary.py"),
    "trace": lambda: run("results/make_trace.py", "--task", "T017", "--exp", "E5"),
    "leak": lambda: run("results/leak_check.py"),
    "test": lambda: run("-m", "pytest", "-q", "tests"),
    "demo-cache": lambda: run("demo/record_cache.py"),
}
TARGETS["E5-all"] = lambda: exp("E5-all")
TARGETS["core"] = lambda: [TARGETS[e]() for e in ("E0", "E1", "E6", "E4", "E5", "E5-all")]
TARGETS["optional"] = lambda: [TARGETS[e]() for e in ("E2", "E3", "E5T")]
TARGETS["all"] = lambda: [TARGETS[t]() for t in ("dataset", "bench", "core", "summary", "trace", "leak")]
TARGETS["dry"] = lambda: (run("data/gen_dataset.py"), [exp(e, ["--dry-run", "--force"]) for e in ("E0", "E1", "E6", "E2", "E3", "E4", "E5")], run("results/leak_check.py"))

if __name__ == "__main__":
    targets = sys.argv[1:] or ["all"]
    for t in targets:
        if t not in TARGETS:
            print(__doc__)
            sys.exit(1)
        TARGETS[t]()
    print("\nDONE:", " ".join(targets))
