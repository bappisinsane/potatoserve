"""Quantization pipeline (Phase 2, FR-2): llama-quantize + llama-imatrix.

Produces GGUF variants of a *source* (high-precision) model with optional
importance-matrix weighting, records size + SHA-256 for every variant in
``quant_manifest.json`` (project-spec section 7), and feeds the quality-vs-size
curve measured by ``potato bench --corpus/--kl-base``.

Flags verified against the pinned v0.6.0 source (never from memory):
  * tools/quantize/quantize.cpp usage():
        llama-quantize [--imatrix file] model-f32.gguf [model-quant.gguf]
                       type [nthreads]
    (options precede the positionals; no overwrite prompt exists, so the run is
    automation-safe; ``--dry-run`` inside the tool computes sizes only.)
  * tools/imatrix/README.md:
        llama-imatrix -m model.gguf -f text.txt [-o out.gguf]
                      [--chunks N] [-ngl N]

Stdlib only - no new dependencies (docs/rules.md).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from potato import runner

# Weight-quant ladder from docs/project-spec.md section 3 (+"-imatrix" variants).
DEFAULT_TYPES: List[str] = ["Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K"]

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "quant_manifest.json"

MANIFEST_HELP = (
    "potato quant quantizes an existing GGUF (highest-precision source, e.g. "
    "F16). Converting a raw Hugging Face repo needs convert_hf_to_gguf.py plus "
    "its Python requirements - not installed (docs/rules.md: dependencies "
    "require owner approval). Download an F16 GGUF of the model instead."
)


def sha256_file(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """SHA-256 of a file, or None if it cannot be read (never fabricated)."""
    p = Path(path)
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


def build_quantize_cmd(
    binary: str,
    source: str,
    dest: str,
    qtype: str,
    imatrix: Optional[str] = None,
    threads: Optional[int] = None,
) -> List[str]:
    """llama-quantize argv (options before positionals, per the pinned usage)."""
    cmd = [binary]
    if imatrix is not None:
        cmd += ["--imatrix", imatrix]
    cmd += [source, dest, qtype]
    if threads is not None:
        cmd.append(str(threads))
    return cmd


def build_imatrix_cmd(
    binary: str,
    model: str,
    corpus: str,
    out: str,
    ngl: Optional[int] = None,
    chunks: Optional[int] = None,
) -> List[str]:
    """llama-imatrix argv (flags per tools/imatrix/README.md in the pin)."""
    cmd = [binary, "-m", model, "-f", corpus, "-o", out]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    return cmd


def variant_output_path(source: str, qtype: str, out_dir: Optional[str] = None) -> Path:
    """``<out_dir>/<source-stem>-<qtype>.gguf`` (deterministic, collision-free)."""
    src = Path(source)
    stem = src.stem
    directory = Path(out_dir) if out_dir is not None else src.parent
    return directory / f"{stem}-{qtype}.gguf"


# --------------------------------------------------------------------------
# quant_manifest.json (project-spec section 7): append-then-merge, ADR-5-safe.
# --------------------------------------------------------------------------

def new_manifest(
    source: str,
    source_sha256: Optional[str] = None,
    imatrix: Optional[Dict[str, Any]] = None,
    llama_cpp_commit: Optional[str] = None,
    build_flags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A fresh manifest describing one quantization source run."""
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    src = Path(source)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created": now,
        "updated": now,
        "llama_cpp_commit": llama_cpp_commit,
        "build_flags": build_flags or {"tool": "llama-quantize"},
        "source": {
            "path": str(src),
            "size_bytes": src.stat().st_size if src.is_file() else None,
            "sha256": source_sha256,
        },
        "imatrix": imatrix,
        "variants": [],
    }


def load_manifest(path: str) -> Optional[Dict[str, Any]]:
    """Read an existing manifest; None when missing or unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def merge_variants(
    existing: Sequence[Dict[str, Any]], new_entries: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge *new_entries* into *existing*, deduped by output path.

    A re-run of the same output path supersedes its old entry (the file on disk
    was replaced), while every other previously measured variant is kept - the
    manifest is never wholesale-overwritten (ADR-5 spirit: provenance of past
    measurements survives).
    """
    by_path: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for entry in list(existing) + list(new_entries):
        key = str(entry.get("path") or entry.get("type") or id(entry))
        if key not in by_path:
            order.append(key)
        by_path[key] = entry  # later entries win
    return [by_path[k] for k in order]


def save_manifest(path: str, manifest: Dict[str, Any]) -> str:
    """Write the manifest atomically (tmp file + replace) and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    tmp.replace(p)
    return str(p)


def variant_entry(
    qtype: str,
    dest: Path,
    imatrix: Optional[str],
    duration_s: float,
    exit_code: int,
    source_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Provenance record for one produced variant (size + hash are measured)."""
    return {
        "type": qtype,
        "path": str(dest),
        "size_bytes": dest.stat().st_size if dest.is_file() else None,
        "sha256": sha256_file(str(dest)),
        "imatrix": imatrix,
        "source_sha256": source_sha256,
        "duration_s": duration_s,
        "exit_code": exit_code,
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# Execution (everything runs through potato.runner: timeout + honest errors).
# --------------------------------------------------------------------------

def run_imatrix(
    model: str,
    corpus: str,
    out: str,
    ngl: Optional[int] = None,
    chunks: Optional[int] = None,
    binary: Optional[str] = None,
    timeout_s: float = runner.DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """Compute an importance matrix; returns its provenance dict.

    Reuses an existing non-empty file at *out* instead of recomputing, so a
    repeated ``potato quant`` does not redo the calibration pass.
    """
    out_path = Path(out)
    if out_path.is_file() and out_path.stat().st_size > 0:
        print(f"imatrix already present: {out_path} (reusing)")
        return {
            "path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256_file(str(out_path)),
            "reused": True,
        }

    bin_path = runner.find_llama_binary("llama-imatrix", binary)
    cmd = build_imatrix_cmd(bin_path, model, corpus, out, ngl=ngl, chunks=chunks)
    print(">> " + " ".join(cmd))
    try:
        run = runner.run_benchmark(cmd, timeout_s=timeout_s)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"llama-imatrix executable not found at '{bin_path}'. Build it with "
            "scripts/build_cuda.ps1 / scripts/build_vulkan.ps1 (both target "
            "llama-imatrix), or pass --imatrix-binary <path>. If you already "
            "have an imatrix file, pass --imatrix <file> to skip this step."
        ) from exc
    if run.timed_out:
        raise RuntimeError(
            f"llama-imatrix timed out after {timeout_s:g}s; use --chunks to "
            "limit the calibration corpus or raise --timeout."
        )
    if run.exit_code != 0:
        raise RuntimeError(
            f"llama-imatrix failed with exit code {run.exit_code}\n"
            + (run.stderr or run.stdout)
        )
    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise RuntimeError(
            f"llama-imatrix finished but produced no non-empty file at "
            f"{out_path}.\nOutput:\n" + (run.stderr or run.stdout)
        )
    return {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256_file(str(out_path)),
        "duration_s": run.duration_s,
        "reused": False,
    }



def run_quant(
    source: str,
    types: Optional[Sequence[str]] = None,
    out_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    imatrix: Optional[str] = None,
    imatrix_from: Optional[str] = None,
    imatrix_out: Optional[str] = None,
    imatrix_out_dir: Optional[str] = None,
    ngl: Optional[int] = None,
    chunks: Optional[int] = None,
    threads: Optional[int] = None,
    quant_binary: Optional[str] = None,
    imatrix_binary: Optional[str] = None,
    timeout_s: float = runner.DEFAULT_TIMEOUT_S,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Produce the requested quant variants of *source* and update the manifest.

    With ``dry_run=True`` nothing is executed or written: the returned dict
    holds the exact commands (project-spec section 10: every benchmark script
    has a ``--dry-run``).

    Returns {"commands", "variants", "imatrix", "manifest", "manifest_path"}.
    Raises RuntimeError with guidance when the source or a tool is missing.
    """
    src = Path(source)
    if not src.is_file():
        raise RuntimeError(f"source GGUF not found: {src}. " + MANIFEST_HELP)
    if types is None:
        types = DEFAULT_TYPES
    types = list(types)
    if not types:
        raise ValueError("no quant types requested")
    out_directory = Path(out_dir) if out_dir else src.parent
    manifest_path = manifest_path or str(out_directory / DEFAULT_MANIFEST_NAME)

    # Resolve the imatrix plan: use a given file, or generate one from corpus.
    imatrix_info: Optional[Dict[str, Any]] = None
    imatrix_path: Optional[str] = None
    commands: List[List[str]] = []
    if imatrix:
        imatrix_path = imatrix
        p = Path(imatrix)
        imatrix_info = {
            "path": str(p),
            "size_bytes": p.stat().st_size if p.is_file() else None,
            "sha256": sha256_file(imatrix),
            "reused": True,
        }
    elif imatrix_from:
        imatrix_path = imatrix_out or str(
            Path(imatrix_out_dir or out_directory) / f"{src.stem}.imatrix.gguf"
        )
        imatrix_bin = runner.find_llama_binary("llama-imatrix", imatrix_binary)
        commands.append(
            build_imatrix_cmd(imatrix_bin, str(src), imatrix_from,
                              imatrix_path, ngl=ngl, chunks=chunks)
        )

    # Build one quantize command per requested type (options first: pinned usage).
    quant_bin = runner.find_llama_binary("llama-quantize", quant_binary)
    plan: List[Dict[str, Any]] = []
    for qtype in types:
        dest = variant_output_path(str(src), qtype, str(out_directory))
        cmd = build_quantize_cmd(
            quant_bin, str(src), str(dest), qtype,
            imatrix=imatrix_path, threads=threads,
        )
        commands.append(cmd)
        plan.append({"type": qtype, "dest": dest, "cmd": cmd})

    if dry_run:
        return {
            "commands": commands,
            "variants": [],
            "imatrix": imatrix_info,
            "manifest": None,
            "manifest_path": manifest_path,
        }

    out_directory.mkdir(parents=True, exist_ok=True)

    # 1) imatrix (only when requested and not already on disk)
    if imatrix_path and imatrix_info is None:
        imatrix_info = run_imatrix(
            str(src), imatrix_from or "", imatrix_path,
            ngl=ngl, chunks=chunks, binary=imatrix_binary, timeout_s=timeout_s,
        )
        imatrix_info = dict(imatrix_info)
        imatrix_info.setdefault("path", imatrix_path)

    source_hash = sha256_file(str(src))

    # 2) quantize each type
    entries: List[Dict[str, Any]] = []
    for item in plan:
        dest: Path = item["dest"]
        print(">> " + " ".join(item["cmd"]))
        if dest.exists():
            # llama-quantize replaces the file silently; the manifest supersedes
            # the old entry by output path (ADR-5: provenance is kept there).
            print(f"note: replacing existing {dest} (manifest supersedes it)")
        try:
            run = runner.run_benchmark(item["cmd"], timeout_s=timeout_s)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"llama-quantize executable not found at '{quant_bin}'. Build "
                "it with scripts/build_cuda.ps1 / scripts/build_vulkan.ps1 "
                "(both target llama-quantize), or pass --quant-binary <path>."
            ) from exc
        if run.timed_out:
            raise RuntimeError(
                f"llama-quantize timed out after {timeout_s:g}s for "
                f"{item['type']}; raise --timeout."
            )
        if run.exit_code != 0:
            raise RuntimeError(
                f"llama-quantize failed for {item['type']} with exit code "
                f"{run.exit_code}\n" + (run.stderr or run.stdout)
            )
        if not dest.is_file():
            raise RuntimeError(
                f"llama-quantize reported success but {dest} does not exist.\n"
                + (run.stderr or run.stdout)
            )
        entries.append(
            variant_entry(
                item["type"], dest, imatrix_path,
                duration_s=run.duration_s, exit_code=run.exit_code,
                source_sha256=source_hash,
            )
        )
        size_mib = (entries[-1]["size_bytes"] or 0) / (1024 * 1024)
        print(f"  {item['type']:9s} -> {dest.name}  ({size_mib:.1f} MiB)")

    # 3) manifest: merge, never wholesale-overwrite (ADR-5 spirit)
    manifest = load_manifest(manifest_path)
    if manifest is None:
        manifest = new_manifest(
            str(src), source_sha256=source_hash, imatrix=imatrix_info,
            llama_cpp_commit=runner.pinned_commit(),
        )
    else:
        if imatrix_info:
            manifest["imatrix"] = imatrix_info
        manifest["source"] = {
            "path": str(src),
            "size_bytes": src.stat().st_size,
            "sha256": source_hash,
        }
    manifest["variants"] = merge_variants(
        manifest.get("variants") or [], entries
    )
    save_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path} ({len(manifest['variants'])} variant(s) tracked)")

    return {
        "commands": commands,
        "variants": entries,
        "imatrix": imatrix_info,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }

