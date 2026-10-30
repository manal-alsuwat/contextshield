"""Peak CPU% / RSS sampler for the local layer (psutil)."""
from __future__ import annotations

import threading
import time

import psutil


class ResourceMonitor:
    """Samples the current process every `interval` seconds while active."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.proc = psutil.Process()
        self.cpu_peak = 0.0
        self.rss_peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        self.proc.cpu_percent(None)  # prime
        while not self._stop.is_set():
            try:
                cpu = self.proc.cpu_percent(None)
                rss = self.proc.memory_info().rss / (1024 * 1024)
                self.cpu_peak = max(self.cpu_peak, cpu)
                self.rss_peak_mb = max(self.rss_peak_mb, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self.rss_peak_mb = max(self.rss_peak_mb, self.proc.memory_info().rss / (1024 * 1024))
        except Exception:
            pass
        return False
