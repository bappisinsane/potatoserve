"""``potato bench`` baseline: run llama-bench and store a section-8 result record.

Phase 0/1 scope (docs/phases.md): a *baseline*. We measure the two throughputs
llama-bench can produce — prefill (pp) and decode (tg) — with at least 3
repeats, plus the spread and per-repeat samples. Since Phase 1 the process is
launched through ``potato.runner``, which samples ``nvidia-smi`` VRAM *during*
the run, so ``vram_peak_mb`` is a real measurement rather than a guess; it stays
``null`` when nvidia-smi is unavailable (honest "unknown", never fabricated).
The remaining section-8 metrics (kl_mean, top1_agree, ppl, agent_success) are
produced by later phases (perplexity, agent-eval); they stay ``null`` here.

Records are persisted through ``potato.store``, which is APPEND-ONLY (ADR-5):
re-running with an existing run_id refuses to clobber prior measured data.

llama-bench output is parsed as JSON (``-o json``). The field mapping below was
verified against the pinned third_party/llama.cpp @ v0.6.0 source and its
README, not from memory:
    avg_ts        -> tokens/sec for that row (decode row when n_gen>0,
                     prefill row when n_gen==0)
    stddev_ts     -> spread
    samples_ts    -> per-repeat throughputs (len == --repetitions)
    build_commit  -> llama_cpp_commit
    gpu_info      -> gpu
    test_time     -> date
    n_prompt,n_gen,n_gpu_layers,flash_attn,type_k,type_v -> config

Standard library only (docs/rules.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from potato import runner
from potato import store as store_mod
from potato.runner import DEFAULT_TIMEOUT_S
from potato.schema import ResultRecord, new_record, validate

DEFAULT_REPEATS = 3
DEFAULT_PP = 512
DEFAULT_TG = 128


def _find_llama_bench(explicit: Optional[str]) -> str:
    """Resolve the llama-bench binary: explicit path, else common build dirs."""
    if explicit:
        return explicit
    repo = Path(__file__).resolve().parents[1]
    candidates = [
        repo / "third_party/llama.cpp/build-cuda/bin/llama-bench",
        repo / "third_party/llama.cpp/build-vulkan/bin/llama-bench",
        repo / "third_party/llama.cpp/build/bin/llama-bench",
    ]
    # On Windows the executable carries a .exe suffix.
    if sys.platform.startswith("win"):
        candidates = [Path(str(c) + ".exe") for c in candidates] + candidates
    for cand in candidates:
        if cand.exists():
            return str(cand)
    # Fall back to PATH.
    return "llama-bench"


def build_bench_command(
    binary: str,
    model: str,
    repeats: int = DEFAULT_REPEATS,
    pp: int = DEFAULT_PP,
    tg: int = DEFAULT_TG,
    ngl: Optional[int] = None,
    flash_attn: Optional[str] = None,
    cache_type_k: Optional[str] = None,
    cache_type_v: Optional[str] = None,
) -> List[str]:
    """Construct the llama-bench argv using only flags verified in v0.6.0.

    Flags verified against the pinned source:
      -r/--repetitions, -p/--n-prompt, -n/--n-gen, -o/--output,
      -ngl/--n-gpu-layers, -fa/--flash-attn, -ctk/--cache-type-k,
      -ctv/--cache-type-v.
    """
    cmd = [
        binary,
        "-m", model,
        "-r", str(repeats),
        "-p", str(pp),
        "-n", str(tg),
        "-o", "json",
    ]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    if flash_attn is not None:
        cmd += ["-fa", str(flash_attn)]
    if cache_type_k is not None:
        cmd += ["-ctk", str(cache_type_k)]
    if cache_type_v is not None:
        cmd += ["-ctv", str(cache_type_v)]
    return cmd



def _classify(row: Dict[str, Any]) -> str:
    """Label a llama-bench row as 'prefill' or 'decode' (spec metric source).

    llama-bench emits a prefill row (n_gen==0) and a decode row (n_prompt==0)
    for each requested pp/tg pair.
    """
    return "decode" if int(row.get("n_gen", 0)) > 0 else "prefill"


def parse_llama_bench_json(stdout: str) -> Dict[str, Any]:
    """Parse ``llama-bench -o json`` stdout into extracted metrics.

    Returns a dict with prefill/decode avg_ts, their stddev (spread), per-repeat
    samples, and the raw rows for provenance. Missing kinds are left out rather
    than defaulted, so the caller records null.
    """
    data = json.loads(stdout)
    rows: List[Dict[str, Any]] = data if isinstance(data, list) else [data]

    out: Dict[str, Any] = {"rows": rows, "samples": {}, "avg": {}, "stddev": {}}
    for row in rows:
        kind = _classify(row)
        out["avg"][kind] = row.get("avg_ts")
        out["stddev"][kind] = row.get("stddev_ts")
        out["samples"][kind] = row.get("samples_ts")
        # Record the prompt/gen sizes actually used for each kind, so the
        # result record's config reflects both the prefill and decode sizes
        # (a single shared row would only carry one of them).
        out.setdefault("sizes", {})[kind] = {
            "n_prompt": row.get("n_prompt"),
            "n_gen": row.get("n_gen"),
        }
    if rows:
        meta = rows[0]
        out["meta"] = {
            "build_commit": meta.get("build_commit"),
            "gpu_info": meta.get("gpu_info"),
            "backends": meta.get("backends"),
            "test_time": meta.get("test_time"),
            "model_filename": meta.get("model_filename"),
            "model_type": meta.get("model_type"),
            "n_prompt": meta.get("n_prompt"),
            "n_gen": meta.get("n_gen"),
            "n_gpu_layers": meta.get("n_gpu_layers"),
            "flash_attn": meta.get("flash_attn"),
            "type_k": meta.get("type_k"),
            "type_v": meta.get("type_v"),
        }
    return out


def _hardware_fingerprint() -> Optional[Dict[str, Any]]:
    """Best-effort live hardware fingerprint (driver/OS) for result enrichment.

    Uses potato.probe (stdlib only). Returns None if detection fails, so a
    bench run still succeeds and simply records driver/os as null rather than
    inventing values.
    """
    try:
        from potato import probe

        return probe.probe()
    except Exception:
        return None


def _model_sha256(model_path: str, chunk: int = 1 << 20) -> Optional[str]:
    """Compute the SHA-256 of the model file, or None if it can't be read.

    Real value computed on the user's machine (never fabricated). Returns None
    if the file is missing or unreadable so the caller records null.
    """
    import hashlib

    p = Path(model_path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def result_from_bench(
    parsed: Dict[str, Any],
    run_id: str,
    driver: Optional[str] = None,
    os_name: Optional[str] = None,
    model_sha256: Optional[str] = None,
    vram_peak_mb: Optional[int] = None,
    vram_baseline_mb: Optional[int] = None,
    vram_samples: Optional[int] = None,
    bench_duration_s: Optional[float] = None,
) -> ResultRecord:
    """Build a section-8 ResultRecord from parsed llama-bench output.

    Prefill/decode throughput come from llama-bench; ``vram_peak_mb`` comes from
    the live nvidia-smi sampling in ``potato.runner`` (None if it could not be
    measured). The remaining section-8 metrics stay null (no invention).
    """
    meta = parsed.get("meta", {})
    avg = parsed.get("avg", {})
    stddev = parsed.get("stddev", {})
    samples = parsed.get("samples", {})

    decode_samples = samples.get("decode") or []
    prefill_samples = samples.get("prefill") or []

    record = new_record(
        run_id,
        llama_cpp_commit=meta.get("build_commit"),
        build_flags={
            "tool": "llama-bench",
            "backends": meta.get("backends"),
            "repeats_default": DEFAULT_REPEATS,
        },
        driver=driver,  # from hardware fingerprint (real detection), else None
        os=os_name,
        gpu=meta.get("gpu_info"),
        model_id=meta.get("model_filename"),
        model_sha256=model_sha256,  # real SHA-256 of the model file, else None
        config={
            # Prefill size from the prefill row, decode size from the decode row.
            "n_prompt": parsed.get("sizes", {}).get("prefill", {}).get("n_prompt",
                        meta.get("n_prompt")),
            "n_gen": parsed.get("sizes", {}).get("decode", {}).get("n_gen",
                        meta.get("n_gen")),
            "n_gpu_layers": meta.get("n_gpu_layers"),
            "flash_attn": meta.get("flash_attn"),
            "cache_type_k": meta.get("type_k"),
            "cache_type_v": meta.get("type_v"),
        },
        metrics={
            "prefill_tps": avg.get("prefill"),
            "decode_tps": avg.get("decode"),
            # Measured live by potato.runner (nvidia-smi sampling during the
            # run). None means "not measured", which is not the same as zero.
            "vram_peak_mb": vram_peak_mb,
        },
        repeats=len(decode_samples) or len(prefill_samples) or None,
        spread={
            "decode_stddev_ts": stddev.get("decode"),
            "prefill_stddev_ts": stddev.get("prefill"),
            "decode_samples_ts": decode_samples,
            "prefill_samples_ts": prefill_samples,
        },
    )
    # Provenance for the VRAM number so the report can explain how it was
    # obtained (and that it was measured, not assumed).
    if record.config is not None:
        record.config.update(
            {
                "vram_baseline_mb": vram_baseline_mb,
                "vram_samples": vram_samples,
                "bench_duration_s": bench_duration_s,
            }
        )
    # Preserve llama-bench's own timestamp if present.
    if meta.get("test_time"):
        record.date = meta["test_time"]
    return record


def write_record(record: ResultRecord, results_dir: Path) -> Path:
    """Validate and persist a result record via the append-only store (ADR-5).

    Raises FileExistsError if ``<results_dir>/<run_id>.json`` already exists, so
    a previously measured run can never be silently overwritten.
    """
    problems = validate(record)
    if problems:
        raise ValueError(f"result record invalid: {problems}")
    path = store_mod.save_record(record.to_dict(), results_dir=str(results_dir))
    return Path(path)


def run_baseline(
    model: str,
    results_dir: Path,
    repeats: int = DEFAULT_REPEATS,
    pp: int = DEFAULT_PP,
    tg: int = DEFAULT_TG,
    binary: Optional[str] = None,
    ngl: Optional[int] = None,
    flash_attn: Optional[str] = None,
    run_id: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    report: bool = True,
) -> Path:
    """Run llama-bench once and persist a section-8 baseline record.

    The process is launched through ``potato.runner`` so peak VRAM is sampled
    live from nvidia-smi during the run (null if unavailable). Returns the path
    written. Raises if llama-bench fails or the record already exists (append-only
    store, ADR-5) - no fabricated data.
    """
    import datetime as _dt

    from potato import report as report_mod

    bin_path = _find_llama_bench(binary)
    cmd = build_bench_command(
        bin_path, model, repeats=repeats, pp=pp, tg=tg, ngl=ngl, flash_attn=flash_attn
    )
    print(">> " + " ".join(cmd))
    try:
        run = runner.run_benchmark(cmd, timeout_s=timeout_s)
    except FileNotFoundError as exc:
        # llama-bench not found (not built, or not on PATH). Fail cleanly with
        # guidance — never fabricate a result (docs/rules.md).
        raise RuntimeError(
            f"llama-bench executable not found at '{bin_path}'. Build it first: "
            "scripts/build_cuda.ps1 (needs the CUDA Toolkit) or "
            "scripts/build_vulkan.ps1, or pass --binary <path>. "
            "See docs/memory.md for the environment fingerprint."
        ) from exc
    if run.timed_out:
        sys.stderr.write(run.stdout or "")
        sys.stderr.write(run.stderr or "")
        raise RuntimeError(
            f"llama-bench timed out after {timeout_s:g}s (partial output above). "
            "A first-time backend build/load can be slow; raise --timeout."
        )
    if run.exit_code != 0:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        raise RuntimeError(f"llama-bench failed with exit code {run.exit_code}")

    parsed = parse_llama_bench_json(run.stdout)
    rid = run_id or _dt.datetime.now(_dt.timezone.utc).strftime("baseline-%Y%m%dT%H%M%SZ")
    # Enrich with real detected values (never invented): driver + OS from the
    # live hardware fingerprint, and the model's actual SHA-256.
    fingerprint = _hardware_fingerprint()
    driver = None
    os_name = None
    if fingerprint:
        nvidia = fingerprint.get("gpu", {}).get("nvidia", {})
        driver = nvidia.get("driver")
        os_name = fingerprint.get("os", {}).get("platform")
    record = result_from_bench(
        parsed,
        rid,
        driver=driver,
        os_name=os_name,
        model_sha256=_model_sha256(model),
        vram_peak_mb=run.vram_peak_mb,
        vram_baseline_mb=run.vram_baseline_mb,
        vram_samples=run.vram_samples,
        bench_duration_s=run.duration_s,
    )
    try:
        path = write_record(record, results_dir)
    except FileExistsError as exc:
        raise RuntimeError(
            f"{exc} Re-run with --run-id <fresh-id> to supersede rather than "
            "overwrite (ADR-5: results are append-only)."
        ) from exc
    print(f"Wrote {path}")
    if run.vram_peak_mb is None:
        print(
            "note: peak VRAM not measured (nvidia-smi unavailable); "
            "vram_peak_mb recorded as null rather than guessed."
        )
    else:
        print(
            f"VRAM: peak {run.vram_peak_mb} MiB "
            f"(baseline {run.vram_baseline_mb} MiB, "
            f"delta {run.vram_delta_mb} MiB, {run.vram_samples} samples)"
        )
    if report:
        # Phase 1 exit criterion: the bench emits a median+spread report.
        print()
        print(report_mod.render_record_report(record.to_dict()))
    return path
