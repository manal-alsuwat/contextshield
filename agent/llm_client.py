"""OpenAI-compatible chat client with disk cache, retry/backoff and a concurrency limiter.

Cache key = sha256(experiment_config + task_id + exact outbound messages + tool schema).
A cache hit makes no network call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

import httpx
from dotenv import load_dotenv

log = logging.getLogger("agent.llm")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ROOT / "cache" / "llm"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def model_tag(model: str | None = None) -> str:
    """File-system-safe tag of the configured model, e.g. 'openai_gpt-oss-120b'. Results are stored per tag."""
    m = model or os.environ.get("LLM_MODEL", "") or "unknown-model"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", m).strip("_")


class RequestTooLarge(RuntimeError):
    """The provider rejects this single request outright (e.g. exceeds its per-minute token cap)."""


class DailyQuotaExhausted(RuntimeError):
    """The provider's per-day quota is used up: retrying is pointless; the run stops cleanly and resumes later."""


class TPMPacer:
    """Optional tokens-per-minute pacing (LLM_TPM_LIMIT) so free tiers don't 429-storm."""

    def __init__(self, limit: int):
        self.limit = limit
        self._sent: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def wait_for(self, tokens: int):
        if self.limit <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                while self._sent and now - self._sent[0][0] > 60:
                    self._sent.popleft()
                used = sum(t for _, t in self._sent)
                if used + tokens <= self.limit or not self._sent:
                    self._sent.append((now, tokens))
                    return
                wait = 60 - (now - self._sent[0][0]) + 0.5
            time.sleep(max(0.5, min(wait, 60)))


class ConcurrencyLimiter:
    """At most MAX_CONCURRENCY in flight; halves itself if >3 429s in a minute."""

    def __init__(self, limit: int):
        self.limit = max(1, limit)
        self._cond = threading.Condition()
        self._inflight = 0
        self._429s: deque[float] = deque()

    def __enter__(self):
        with self._cond:
            while self._inflight >= self.limit:
                self._cond.wait()
            self._inflight += 1
        return self

    def __exit__(self, *exc):
        with self._cond:
            self._inflight -= 1
            self._cond.notify_all()

    def note_429(self):
        now = time.time()
        with self._cond:
            self._429s.append(now)
            while self._429s and now - self._429s[0] > 60:
                self._429s.popleft()
            if len(self._429s) > 3 and self.limit > 1:
                self.limit = max(1, self.limit // 2)
                self._429s.clear()
                log.warning("Too many 429s — concurrency halved to %d", self.limit)


class LLMClient:
    def __init__(self):
        self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "256"))
        self.seed = int(os.environ.get("LLM_SEED", "42"))
        self.price_in = float(os.environ.get("LLM_PRICE_IN_PER_M", "0"))
        self.price_out = float(os.environ.get("LLM_PRICE_OUT_PER_M", "0"))
        self.timeout = float(os.environ.get("LLM_TIMEOUT_S", "60"))
        self.limiter = ConcurrencyLimiter(int(os.environ.get("MAX_CONCURRENCY", "4")))
        self.pacer = TPMPacer(int(os.environ.get("LLM_TPM_LIMIT", "0")))  # 0 = no pacing
        self.calls = 0
        self.cache_hits = 0
        self.retries = 0
        self.consecutive_failures = 0
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=self.timeout)

    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    # ------------------------------------------------------------------
    def cache_key(self, exp_config: dict, task_id: str, messages: list[dict], tools: list[dict], model: str) -> str:
        """Legacy key (experiment config included) — kept so earlier cached responses stay readable."""
        gen = {"max_tokens": self.max_tokens, "seed": self.seed, "temperature": 0}
        blob = json.dumps({"cfg": exp_config, "task": task_id, "msgs": messages, "tools": tools, "model": model, "gen": gen}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def content_key(self, task_id: str, messages: list[dict], tools: list[dict], model: str) -> str:
        """Content key: identical outbound messages for a task (any experiment) share one cached response."""
        gen = {"max_tokens": self.max_tokens, "seed": self.seed, "temperature": 0}
        blob = json.dumps({"task": task_id, "msgs": messages, "tools": tools, "model": model, "gen": gen}, sort_keys=True, ensure_ascii=False)
        return "c_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def cost_usd(self, usage: dict) -> float:
        return (usage.get("prompt_tokens", 0) * self.price_in + usage.get("completion_tokens", 0) * self.price_out) / 1_000_000

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict], exp_config: dict, task_id: str, bypass_cache: bool = False) -> tuple[dict, dict]:
        """Return (response_json, meta). meta: {cached, cloud_ms, retries}.
        bypass_cache=True forces a real network call (the response is still stored)."""
        key = self.cache_key(exp_config, task_id, messages, tools, self.model)
        ckey = self.content_key(task_id, messages, tools, self.model)
        f = CACHE_DIR / f"{key}.json"
        cf = CACHE_DIR / f"{ckey}.json"
        for hit in (() if bypass_cache else (cf, f)):  # content key first (shared across experiments), then the legacy key
            if hit.exists():
                with self._lock:
                    self.cache_hits += 1
                data = json.loads(hit.read_text(encoding="utf-8"))
                if not cf.exists():
                    cf.write_text(hit.read_text(encoding="utf-8"), encoding="utf-8")  # migrate legacy entry to the content key
                return data["response"], {"cached": True, "cloud_ms": data.get("cloud_ms", 0.0), "retries": 0}

        if not self.configured():
            raise RuntimeError("LLM not configured: set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env")

        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if os.environ.get("LLM_SEND_SEED", "1") == "1":
            body["seed"] = self.seed
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        est_tokens = sum(len((m.get("content") or "")) for m in messages) // 4 + self.max_tokens
        if self.pacer.limit and est_tokens > self.pacer.limit:
            raise RequestTooLarge(f"estimated {est_tokens} tokens > provider cap {self.pacer.limit} TPM")
        delay, retries, last_err = 2.0, 0, None
        for attempt in range(5):
            with self.limiter:
                self.pacer.wait_for(est_tokens)
                t0 = time.perf_counter()
                try:
                    r = self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                    cloud_ms = (time.perf_counter() - t0) * 1000
                    if r.status_code == 413 or (r.status_code in (400, 429) and "too large" in r.text.lower()):
                        raise RequestTooLarge(f"HTTP {r.status_code}: {r.text[:200]}")
                    if r.status_code == 200:
                        resp = r.json()
                        with self._lock:
                            self.calls += 1
                            self.consecutive_failures = 0
                        payload = json.dumps({"response": resp, "cloud_ms": cloud_ms, "ts": time.time()}, ensure_ascii=False)
                        f.write_text(payload, encoding="utf-8")
                        cf.write_text(payload, encoding="utf-8")
                        return resp, {"cached": False, "cloud_ms": cloud_ms, "retries": retries}
                    if r.status_code == 429 and ("per day" in r.text.lower() or "tpd" in r.text.lower() or "rpd" in r.text.lower()):
                        raise DailyQuotaExhausted(f"HTTP 429 daily quota: {r.text[:200]}")
                    if r.status_code == 429 or r.status_code >= 500:
                        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                        if r.status_code == 429:
                            self.limiter.note_429()
                        ra = r.headers.get("Retry-After")
                        wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                    else:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                    wait = delay
            retries += 1
            with self._lock:
                self.retries += 1
            log.warning("Retry %d/4 for %s after %s (sleep %.0fs)", retries, task_id, last_err, wait)
            time.sleep(min(wait, 32.0))
            delay = min(delay * 2, 32.0)
        with self._lock:
            self.consecutive_failures += 1
        raise RuntimeError(f"LLM call failed after 5 attempts: {last_err}")
