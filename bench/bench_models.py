"""§8.1 Benchmarks: classifier & embedder load time, per-chunk inference, full-document
scans at 10/50/100 chunks, parallel A+B wall time, CPU% and RSS. Cold and warm.

  python bench/bench_models.py   -> bench/BENCH.md + bench/bench.json
Decision gate (§3[A]): classifier load > 10 s or median per-chunk > 300 ms -> rules only.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shield.injection import InjectionClassifier, InjectionScanner, rule_scan  # noqa: E402
from shield.relevance import Embedder  # noqa: E402
from shield.resources import ResourceMonitor  # noqa: E402


def load_chunks(n: int) -> list[str]:
    chunks = []
    for s in ("dev", "test"):
        for f in sorted((ROOT / "data" / s).glob("T*.json")):
            t = json.loads(f.read_text(encoding="utf-8"))
            chunks += [c["text"] for c in t["context"]]
            if len(chunks) >= n:
                return chunks[:n]
    return (chunks * (n // max(1, len(chunks)) + 1))[:n]


def med_p95(xs):
    xs = sorted(xs)
    return round(st.median(xs), 1), round(xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))], 1)


def main():
    out = {}
    chunks = load_chunks(100)
    # --- classifier -----------------------------------------------------------
    t0 = time.perf_counter()
    with ResourceMonitor() as m:
        clf = InjectionClassifier()
    out["clf_load_s_cold"] = round(time.perf_counter() - t0, 2)
    out["clf_load_rss_mb"] = round(m.rss_peak_mb, 0)
    t0 = time.perf_counter()
    InjectionClassifier()
    out["clf_load_s_warm"] = round(time.perf_counter() - t0, 2)
    lat_cold = []
    for c in chunks[:10]:
        t0 = time.perf_counter(); clf.score(c); lat_cold.append((time.perf_counter() - t0) * 1000)
    lat = []
    for c in chunks:
        t0 = time.perf_counter(); clf.score(c); lat.append((time.perf_counter() - t0) * 1000)
    out["clf_chunk_ms_cold_med_p95"] = med_p95(lat_cold)
    out["clf_chunk_ms_warm_med_p95"] = med_p95(lat)
    for n in (10, 50, 100):
        with ResourceMonitor() as m:
            t0 = time.perf_counter()
            for c in chunks[:n]:
                rule_scan(c) or clf.score(c)
            out[f"clf_scan_{n}_chunks_ms"] = round((time.perf_counter() - t0) * 1000, 0)
        out[f"clf_scan_{n}_cpu_pct"] = round(m.cpu_peak, 0)
    rl = []
    for c in chunks:
        t0 = time.perf_counter(); rule_scan(c); rl.append((time.perf_counter() - t0) * 1000)
    out["rules_chunk_ms_med_p95"] = med_p95(rl)

    # --- embedder -------------------------------------------------------------
    Embedder._instance = None
    t0 = time.perf_counter()
    with ResourceMonitor() as m:
        emb = Embedder.get()
    out["emb_load_s_cold"] = round(time.perf_counter() - t0, 2)
    out["emb_load_rss_mb"] = round(m.rss_peak_mb, 0)
    el = []
    for c in chunks[:30]:
        t0 = time.perf_counter(); emb.model.encode([c], normalize_embeddings=True); el.append((time.perf_counter() - t0) * 1000)
    out["emb_chunk_ms_med_p95"] = med_p95(el)
    for n in (10, 50, 100):
        t0 = time.perf_counter()
        emb.model.encode(chunks[:n], batch_size=32, normalize_embeddings=True)
        out[f"emb_batch_{n}_chunks_ms"] = round((time.perf_counter() - t0) * 1000, 0)

    # --- parallel A + B (100 chunks) ---------------------------------------------
    def A():
        for c in chunks:
            rule_scan(c) or clf.score(c)

    def B():
        emb.model.encode(chunks, batch_size=32, normalize_embeddings=True)

    t0 = time.perf_counter(); A(); B(); out["A_then_B_100_ms"] = round((time.perf_counter() - t0) * 1000, 0)
    with ResourceMonitor() as m:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(2) as ex:
            fa, fb = ex.submit(A), ex.submit(B)
            fa.result(); fb.result()
        out["A_parallel_B_100_ms"] = round((time.perf_counter() - t0) * 1000, 0)
    out["A_parallel_B_cpu_pct"] = round(m.cpu_peak, 0)
    out["A_parallel_B_rss_mb"] = round(m.rss_peak_mb, 0)

    # --- int8 sanity results per model (written by prepare_onnx_model) ---------------------------------
    from shield.injection import MODELS, onnx_dir

    out["quant_check"] = {}
    for name, mid in MODELS.items():
        f = onnx_dir(mid) / "quant_check.json"
        if f.exists():
            out["quant_check"][name] = json.loads(f.read_text(encoding="utf-8"))

    # --- Attack detection offline (no cloud calls): rules-only vs rules+classifier for every available model,
    #     split by attack kind (template / held-out) and by split (dev / test). Classifier forced on (no gate)
    #     for measurement only; thresholds 0.5 / 0.9 / 0.99 (threshold choice uses the DEV rows only).
    del clf
    tasks = []
    for sp in ("dev", "test"):
        tasks += [json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "data" / sp).glob("T*.json"))]
    configs: dict[str, InjectionScanner | None] = {"rules": None}
    gates = {}
    for name in ("base", "small"):
        try:
            sc_auto = InjectionScanner(mode="auto", model=name)
            gates[name] = {"gate": sc_auto.gate, "mode_used": sc_auto.mode_used, "fallback_reason": sc_auto.fallback_reason}
            del sc_auto
            configs[f"rules+{name}"] = InjectionScanner(mode="classifier", model=name)
        except Exception as e:
            gates[name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    out["gates"] = gates
    det: dict[str, dict] = {}
    missed: dict[str, list[str]] = {}
    for cname, sc in configs.items():
        ths = [None] if sc is None else [0.5, 0.9, 0.99]
        for th in ths:
            key = cname if th is None else f"{cname}@{th}"
            d = det[key] = {"dev_template": [0, 0], "dev_heldout": [0, 0], "test_template": [0, 0], "test_heldout": [0, 0], "fp_dev": 0, "fp_test": 0, "clean_dev": 0, "clean_test": 0, "ms_per_chunk": 0.0}
            missed[key] = []
            n_ms = 0
            for t in tasks:
                for c in t["context"]:
                    if c["source"] != "untrusted":
                        continue
                    is_att = bool(c.get("contains_injection"))
                    t0 = time.perf_counter()
                    hit = bool(rule_scan(c["text"]))
                    if not hit and sc is not None:
                        s, _ = sc._clf_hits(c["text"])
                        hit = s >= th
                    d["ms_per_chunk"] += (time.perf_counter() - t0) * 1000
                    n_ms += 1
                    if is_att:
                        k = f"{t['split']}_{c.get('injection_kind', 'template')}"
                        d[k][1] += 1
                        d[k][0] += hit
                        if not hit:
                            missed[key].append(f"{t['task_id']}/{c['chunk_id']}({c.get('injection_kind', 'template')})")
                    else:
                        d[f"clean_{t['split']}"] += 1
                        d[f"fp_{t['split']}"] += hit
            d["ms_per_chunk"] = round(d["ms_per_chunk"] / max(1, n_ms), 1)
            for k in ("dev_template", "dev_heldout", "test_template", "test_heldout"):
                d[k] = f"{d[k][0]}/{d[k][1]}"
    out["attack_detection"] = det
    out["attack_missed"] = missed

    (ROOT / "bench" / "bench.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = ["# Model benchmarks (§8.1) — CPU only", "", "| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in out.items()]
    (ROOT / "bench" / "BENCH.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
