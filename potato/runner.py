"""Benchmark runner for Phase 1.

Executes llama-bench as a subprocess while sampling VRAM from nvidia-smi in a
background thread, so the peak is measured *during* the run rather than guessed.

Stdlib only (subprocess, threading, shutil, time) - no new dependencies
(docs/rules.md: no dependency without approval).

VRAM query verified against this machine's driver (616.56, GTX 1660 Ti):
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
returns a single integer in MiB (e.g. "644"). We poll that at a fixed interval
and keep the maximum observed. If nvidia-smi is absent the peaks stay None -
an honest "unknown", never a fabricated number.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_SAMPLE_INTERVAL_S = 0.05
DEFAULT_TIMEOUT_S = 1800  # 30 min; first-time backend builds/loads can be slow.


@dataclass
class VramSampler:
    """Polls nvidia-smi memory.used and records the peak (max) observed."""

    interval_s: float = DEFAULT_SAMPLE_INTERVAL_S
    samples: List[int] = field(default_factory=list)
    baseline_mb: Optional[int] = None  # idle VRAM before the process launched
    peak_mb: Optional[int] = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)

    def _query_used_mb(self) -> Optional[int]:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return None
        try:
            proc = subprocess.run(
                [nvidia_smi,
                 "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        first = proc.stdout.strip().splitlines()[0].split(",")[0].strip()
        try:
            return int(first)
        except ValueError:
            return None

    def sample_once(self) -> Optional[int]:
        used = self._query_used_mb()
        if used is not None:
            self.samples.append(used)
            if self.peak_mb is None or used > self.peak_mb:
                self.peak_mb = used
        return used

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.baseline_mb = self._query_used_mb()  # idle VRAM before launch
        if shutil.which("nvidia-smi") is None:
            return  # no sampler thread; peaks remain None (honest)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def delta_mb(self) -> Optional[int]:
        if self.peak_mb is None or self.baseline_mb is None:
            return None
        return self.peak_mb - self.baseline_mb


@dataclass
class BenchRun:
    """Result of one llama-bench invocation, including measured VRAM."""

    cmd: List[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    vram_baseline_mb: Optional[int]
    vram_peak_mb: Optional[int]
    vram_delta_mb: Optional[int]
    vram_samples: int
    timed_out: bool = False

    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def run_benchmark(
    cmd: List[str],
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
) -> BenchRun:
    """Run *cmd* while sampling VRAM; return a BenchRun with the measured peak.

    A timeout is captured as timed_out=True (exit_code=-1) rather than raising,
    so a slow first run never loses the partial output it produced.
    """
    sampler = VramSampler(interval_s=sample_interval_s)
    sampler.start()
    timed_out = False
    stdout = stderr = ""
    code = -1
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        out = exc.stdout or ""
        err = exc.stderr or ""
        stdout = out if isinstance(out, str) else out.decode("utf-8", "replace")
        stderr = err if isinstance(err, str) else err.decode("utf-8", "replace")
    except FileNotFoundError:
        sampler.stop()
        raise
    finally:
        duration = time.monotonic() - start
        sampler.stop()

    return BenchRun(
        cmd=list(cmd),
        exit_code=code,
        stdout=stdout,
        stderr=stderr,
        duration_s=round(duration, 3),
        vram_baseline_mb=sampler.baseline_mb,
        vram_peak_mb=sampler.peak_mb,
        vram_delta_mb=sampler.delta_mb,
        vram_samples=len(sampler.samples),
        timed_out=timed_out,
    )
