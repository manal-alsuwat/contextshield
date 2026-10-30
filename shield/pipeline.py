"""ContextShield pipeline: router -> B -> A -> C -> D  (production, scan_scope="outbound")
                         router -> [A || B] -> C -> D (measurement, scan_scope="all")

Scanner scopes (FIXED names, logged per task):
  all      : stage A scans every untrusted chunk, including chunks B already dropped.
             Kept only to measure detection recall (bench/BENCH.md, REPORT.md "E5-all").
  outbound : B runs first; A scans only (a) untrusted chunks B selected / must-kept, plus
             (b) any untrusted chunk whose text mentions an agent tool name (cheap substring
             check) regardless of B. Chunks outside this set are never sent, so they are
             not scanned — no risk, no compute.

Usage:
    shield = ContextShield(ShieldConfig(mode="router", T=2000, budget=2500))
    res = shield.process(question, chunks)
    res.outbound_question, res.outbound_context  -> what goes to the cloud model
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .injection import InjectionScanner
from .pii import Redactor, regex_prescan
from .relevance import naive_truncate, select_relevant, split_chunk
from .resources import ResourceMonitor
from .router import route
from .tokens import count_tokens
from .wrap import UNTRUSTED_NOTE, render_context

SCAN_SCOPES = ("all", "outbound")


@dataclass
class ShieldConfig:
    name: str = "E5"
    mode: str = "router"  # "router" | "fixed" | "none" | "truncate"
    stages: tuple[str, ...] = ()  # for mode="fixed": subset of ("A","B","C","D")
    T: int = 2000  # router R1 threshold
    budget: int = 2500  # stage B token budget (also E6 truncation length)
    min_chunks: int = 2
    injection_mode: str = "auto"  # auto | rules | classifier
    injection_threshold: float = 0.5
    injection_model: str = "base"  # base | small | <hf id>  (see shield.injection.MODELS)
    rule_action: str = "excise"  # excise | drop
    scan_scope: str = "outbound"  # "outbound" (production) | "all" (measurement only)
    tool_names: tuple[str, ...] = ("export_data",)  # scope=outbound rule (b): always scan chunks mentioning a tool
    clf_batch_size: int = 8  # classifier sequences per forward pass (length-sorted; 8 measured best on this CPU)
    clf_max_length: int = 256  # tokens per classifier sequence
    clf_threads: int = 0  # ONNX Runtime intra-op threads (0 = ORT default)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShieldResult:
    outbound_question: str
    outbound_chunks: list[dict]
    outbound_context: str
    system_note: str
    stages_run: list[str]
    router: dict | None
    injection_log: list[dict]
    injection_removed: list[dict]
    relevance_log: list[dict]
    relevance_info: dict
    scan: dict
    pii_counts: dict
    pii_spans: list[dict]
    timings: dict
    resources: dict
    input_tokens_local: int
    outbound_tokens_local: int
    _redactor: Redactor | None = field(default=None, repr=False)
    _pre_c_text: str = field(default="", repr=False)  # text as it stood before C (after A/B) — local metrics only
    # LOCAL ONLY (never persisted): what A excised and what C replaced — for on-screen verification
    _flagged: list[dict] = field(default_factory=list, repr=False)  # {chunk_id, layer, paragraphs:[text]}
    _pii_map: list[dict] = field(default_factory=list, repr=False)  # {type, value, placeholder, chunk_id}

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return d


class ContextShield:
    def __init__(self, config: ShieldConfig):
        if config.scan_scope not in SCAN_SCOPES:
            raise ValueError(f"scan_scope must be one of {SCAN_SCOPES}")
        self.cfg = config
        self.scanner: InjectionScanner | None = None
        needs_A = config.mode == "router" or (config.mode == "fixed" and "A" in config.stages)
        if needs_A:
            self.scanner = InjectionScanner(mode=config.injection_mode, threshold=config.injection_threshold, rule_action=config.rule_action, model=config.injection_model,
                                            max_length=config.clf_max_length, threads=config.clf_threads)
        self.scanner_info = {
            "mode_used": self.scanner.mode_used if self.scanner else None,
            "fallback_reason": self.scanner.fallback_reason if self.scanner else None,
            "gate": self.scanner.gate if self.scanner else None,
            "scan_scope": config.scan_scope,
        }

    # ------------------------------------------------------------------
    def _decide(self, question: str, chunks: list[dict]) -> tuple[set[str], dict | None]:
        cfg = self.cfg
        if cfg.mode == "router":
            d = route(question, chunks, T=cfg.T)
            stages = {s for s, on in (("A", d.run_A), ("B", d.run_B), ("C", d.run_C), ("D", d.run_D)) if on}
            return stages, d.as_dict()
        if cfg.mode == "fixed":
            return set(cfg.stages), None
        if cfg.mode == "truncate":
            return {"TRUNC"}, None
        return set(), None  # none -> E0

    def _mentions_tool(self, text: str) -> bool:
        low = text.lower()
        return any(t.lower() in low for t in self.cfg.tool_names)

    # ------------------------------------------------------------------
    def process(self, question: str, chunks: list[dict]) -> ShieldResult:
        cfg = self.cfg
        t_start = time.perf_counter()
        in_tokens = count_tokens(question) + sum(count_tokens(c["text"]) for c in chunks)
        stages, router_dec = self._decide(question, chunks)
        timings = {"local_ms_A": 0.0, "local_ms_B": 0.0, "local_ms_C": 0.0, "local_ms_router": 0.0}
        if router_dec is not None:
            timings["local_ms_router"] = round((time.perf_counter() - t_start) * 1000, 2)

        units: list[dict] = []
        for c in chunks:
            units.extend(split_chunk(c))
        n_untrusted = sum(1 for u in units if u.get("source") == "untrusted")

        inj_log, inj_removed, rel_log, rel_info = [], [], [], {}
        scan = {"scan_scope": cfg.scan_scope, "n_untrusted_units": n_untrusted, "n_scanned": 0, "n_scanned_selected": 0,
                "n_scanned_tool_mention_extra": 0, "n_skipped_dropped": 0, "clf_sequences": 0}
        run_A = "A" in stages and self.scanner is not None
        with ResourceMonitor() as mon:
            def run_B():
                t0 = time.perf_counter()
                if "B" in stages:
                    sel, lg, info = select_relevant(question, units, budget_tokens=cfg.budget, min_chunks=cfg.min_chunks)
                elif "TRUNC" in stages:
                    sel, lg, info = naive_truncate(units, cfg.budget), [], {"truncated_to": cfg.budget}
                else:
                    sel, lg, info = list(units), [], {}
                return sel, lg, info, (time.perf_counter() - t0) * 1000

            def run_A_on(targets: list[dict]):
                t0 = time.perf_counter()
                v = self.scanner.scan(targets, batch_size=cfg.clf_batch_size) if targets else []
                return v, (time.perf_counter() - t0) * 1000

            if run_A and cfg.scan_scope == "all":
                # measurement configuration: A on every untrusted chunk, in parallel with B
                with ThreadPoolExecutor(max_workers=2) as ex:
                    fa = ex.submit(run_A_on, units)
                    fb = ex.submit(run_B)
                    verdicts, timings["local_ms_A"] = fa.result()
                    selected, rel_log, rel_info, timings["local_ms_B"] = fb.result()
                scan["n_scanned"] = scan["n_scanned_selected"] = len(verdicts)
                scan["n_skipped_dropped"] = 0
            else:
                selected, rel_log, rel_info, timings["local_ms_B"] = run_B()
                verdicts = []
                if run_A:
                    # production configuration: B -> A on outbound untrusted chunks (+ tool-name mentions)
                    sel_ids = {u["chunk_id"] for u in selected}
                    targets, extra = [], 0
                    for u in units:
                        if u.get("source") != "untrusted":
                            continue
                        if u["chunk_id"] in sel_ids:
                            targets.append(u)
                        elif self._mentions_tool(u["text"]):
                            targets.append(u)
                            extra += 1
                    verdicts, timings["local_ms_A"] = run_A_on(targets)
                    scan["n_scanned"] = len(targets)
                    scan["n_scanned_selected"] = len(targets) - extra
                    scan["n_scanned_tool_mention_extra"] = extra
                    scan["n_skipped_dropped"] = n_untrusted - len(targets)
            flagged: list[dict] = []
            if run_A:
                inj_log = [v.as_dict() for v in verdicts]
                selected, inj_removed = self.scanner.apply(selected, verdicts)
                scan["clf_sequences"] = getattr(self.scanner, "last_scan_sequences", 0) if verdicts else 0
                for v in verdicts:
                    if v.decision != "clean":
                        layer = ("rules: " + ", ".join(v.rule_hit)) if v.rule_hit else f"classifier score {v.clf_score}"
                        flagged.append({"chunk_id": v.chunk_id, "layer": layer, "paragraphs": list(v.excised)})

            # ---- C -------------------------------------------------------
            redactor = Redactor()
            pre_c_text = question + "\n" + "\n".join(c["text"] for c in selected)  # LOCAL ONLY (metrics), never persisted
            out_q = question
            if "C" in stages:
                out_q = redactor.redact(question, "question")
                selected = [{**c, "text": redactor.redact(c["text"], c["chunk_id"])} for c in selected]
                timings["local_ms_C"] = redactor.ms
            else:
                # cheap pre-scan only (pass-through) — recorded, nothing changed
                regex_prescan(question)

            # ---- D -------------------------------------------------------
            wrap = "D" in stages
            context_text = render_context(selected, wrap_untrusted=wrap)
            note = UNTRUSTED_NOTE if wrap else ""

        timings["local_ms_total_wall"] = round((time.perf_counter() - t_start) * 1000, 2)
        for k in ("local_ms_A", "local_ms_B", "local_ms_C"):
            timings[k] = round(timings[k], 2)
        out_tokens = count_tokens(out_q) + count_tokens(context_text) + count_tokens(note)
        return ShieldResult(
            outbound_question=out_q,
            outbound_chunks=selected,
            outbound_context=context_text,
            system_note=note,
            stages_run=sorted(stages),
            router=router_dec,
            injection_log=inj_log,
            injection_removed=inj_removed,
            relevance_log=rel_log,
            relevance_info=rel_info,
            scan=scan,
            pii_counts=redactor.counts,
            pii_spans=redactor.spans,
            timings=timings,
            resources={"cpu_pct_peak": round(mon.cpu_peak, 1), "rss_mb_peak": round(mon.rss_peak_mb, 1)},
            input_tokens_local=in_tokens,
            outbound_tokens_local=out_tokens,
            _redactor=redactor,
            _pre_c_text=pre_c_text,
            _flagged=flagged,
            _pii_map=[{"type": s["type"], "value": v, "placeholder": redactor._map.get(f"{s['type']}:{v}", ""), "chunk_id": s["where"]}
                      for s, v in zip(redactor.spans, redactor._span_values)],
        )
