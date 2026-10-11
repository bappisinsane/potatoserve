"""``potato bench`` baseline: run llama-bench and store a section-8 result record.

Phase 0 scope (docs/phases.md): a *baseline* only. We measure the two
throughputs llama-bench can produce — prefill (pp) and decode (tg) — with at
least 3 repeats, plus the spread and per-repeat samples. The other section-8
metrics (vram_peak_mb, kl_mean, top1_agree, ppl, agent_success) are produced by
later phases (llama-server probe, perplexity, agent-eval); they stay ``null``
here rather than being invented.

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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def result_from_bench(parsed: Dict[str, Any], run_id: str) -> ResultRecord:
    """Build a section-8 ResultRecord from parsed llama-bench output.

    Only prefill/decode throughput are populated; the remaining section-8
    metrics stay null (Phase 0 baseline — no invention).
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
        driver=None,  # populated by caller who has hardware.json, if available
        os=None,
        gpu=meta.get("gpu_info"),
        model_id=meta.get("model_filename"),
        model_sha256=None,  # optional; add a hashing step if required by spec
        config={
            "n_prompt": meta.get("n_prompt"),
            "n_gen": meta.get("n_gen"),
            "n_gpu_layers": meta.get("n_gpu_layers"),
            "flash_attn": meta.get("flash_attn"),
            "cache_type_k": meta.get("type_k"),
            "cache_type_v": meta.get("type_v"),
        },
        metrics={
            "prefill_tps": avg.get("prefill"),
            "decode_tps": avg.get("decode"),
        },
        repeats=len(decode_samples) or len(prefill_samples) or None,
        spread={
            "decode_stddev_ts": stddev.get("decode"),
            "prefill_stddev_ts": stddev.get("prefill"),
            "decode_samples_ts": decode_samples,
            "prefill_samples_ts": prefill_samples,
        },
    )
    # Preserve llama-bench's own timestamp if present.
    if meta.get("test_time"):
        record.date = meta["test_time"]
    return record


def write_record(record: ResultRecord, results_dir: Path) -> Path:
    """Validate and write a result record as JSON under results_dir."""
    problems = validate(record)
    if problems:
        raise ValueError(f"result record invalid: {problems}")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record.run_id}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


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
) -> Path:
    """Run llama-bench once and persist a section-8 baseline record.

    Returns the path written. Raises if llama-bench fails (no fabricated data).
    """
    import datetime as _dt

    bin_path = _find_llama_bench(binary)
    cmd = build_bench_command(
        bin_path, model, repeats=repeats, pp=pp, tg=tg, ngl=ngl, flash_attn=flash_attn
    )
    print(">> " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"llama-bench failed with exit code {proc.returncode}")

    parsed = parse_llama_bench_json(proc.stdout)
    rid = run_id or _dt.datetime.now(_dt.timezone.utc).strftime("baseline-%Y%m%dT%H%M%SZ")
    record = result_from_bench(parsed, rid)
    path = write_record(record, results_dir)
    print(f"Wrote {path}")
    return path
