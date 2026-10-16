"""Quality gate (Phase 2): KL divergence + perplexity via llama-perplexity.

Implements the measurement half of docs/project-spec.md section 5:

1. Reference: ``llama-perplexity -m <high-precision model> -f <corpus>
   --kl-divergence-base ref.kld`` writes the base logits file.
2. Candidate: same command plus ``--kl-divergence`` -> mean KL, top-1
   agreement ("Same top p"), perplexity vs that base.
3. Plain mode: only perplexity; KL stays null - unknown, never zero.

Output format verified against pinned v0.6.0 (third_party/llama.cpp/tools/
perplexity/perplexity.cpp lines 647 and 1892-1982): "Final estimate: PPL = ...",
"Mean PPL(Q) ...", "Mean    KLD: ... +- ...", "Same top p: ... %".

Stdlib only (re, statistics, subprocess via potato.runner) - no dependencies.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from potato import runner

DEFAULT_CTX = 512  # llama-perplexity's own default n_ctx (perplexity.cpp:1993)
DEFAULT_REPEATS = 3  # docs/rules.md: >= 3 runs per point

# ANSI color/erase sequences llama.cpp may emit when attached to a console.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Floats as printed by the pinned format strings (%10.6lf etc.).
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

# The uncertainty separator as emitted: UTF-8 bytes for U+00B1 (plus-minus),
# the decoded char, or the "+/-" spelling some tools use.
_PM = r"(?:±|\xc2\xb1|\+/-)"

# Patterns written from the exact format strings in the pinned perplexity.cpp.
# Lines are anchored, so per-chunk table rows (which start with the chunk
# index) can never match.
_PLAIN_PPL_RE = re.compile(rf"^Final estimate:\s*PPL\s*=\s*({_NUM})\s*{_PM}\s*({_NUM})")
_PPL_Q_RE = re.compile(rf"^Mean PPL\(Q\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
_PPL_BASE_RE = re.compile(rf"^Mean PPL\(base\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
_PPL_RATIO_RE = re.compile(rf"^Mean PPL\(Q\)/PPL\(base\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
_PPL_DIFF_RE = re.compile(rf"^Mean PPL\(Q\)-PPL\(base\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
_LOG_PPL_RATIO_RE = re.compile(rf"^Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
_COR_RE = re.compile(rf"^Cor\(ln\(PPL\(Q\)\), ln\(PPL\(base\)\)\):\s*({_NUM})%")
_KL_MEAN_RE = re.compile(rf"^Mean\s+KLD\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
_KL_EXTREME_RE = re.compile(rf"^(Maximum|Median|Minimum)\s+KLD\s*:\s*({_NUM})")
_KL_PERCENTILE_RE = re.compile(rf"^({_NUM})%\s+KLD\s*:\s*({_NUM})")
# "Same top p: 99.739 +- 0.013 %" (percent values; stored as 0..1 fraction).
_TOP1_RE = re.compile(rf"^Same top p\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
# Delta-p lines ("Mean    Δp:" / "RMS Δp") - Δ may arrive as '?' after
# errors="replace", so the prefix before 'p' is matched loosely.
_MEAN_DP_RE = re.compile(rf"^Mean\s+(\S*?)p\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")
_RMS_DP_RE = re.compile(rf"^RMS\s+(\S*?)p\s*:\s*({_NUM})\s*{_PM}\s*({_NUM})")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so console-colored logs parse cleanly."""
    return _ANSI_RE.sub("", text)


def parse_perplexity_output(text: str) -> Dict[str, Any]:
    """Parse llama-perplexity stdout+stderr into a flat metrics dict.

    Every value is Optional: the tool legitimately omits whole sections (the
    KL summary is skipped when fewer than 100 tokens were compared, and a
    plain run has no KL at all). Missing -> None, never a fabricated number.

    Returns: ppl, ppl_unc, ppl_base(+_unc), ppl_ratio(+_unc), ppl_diff(+_unc),
    log_ppl_ratio(+_unc), correlation_pct, kl_mean(+_unc), kl_median, kl_max,
    kl_min, kl_percentiles (dict), top1_agree (0..1 fraction, +_unc),
    delta_p_mean_pct, delta_p_rms_pct.
    """
    res: Dict[str, Any] = {
        "ppl": None, "ppl_unc": None, "ppl_base": None, "ppl_base_unc": None,
        "ppl_ratio": None, "ppl_ratio_unc": None, "ppl_diff": None,
        "ppl_diff_unc": None, "log_ppl_ratio": None, "log_ppl_ratio_unc": None,
        "correlation_pct": None,
        "kl_mean": None, "kl_mean_unc": None, "kl_median": None,
        "kl_max": None, "kl_min": None, "kl_percentiles": {},
        "top1_agree": None, "top1_agree_unc": None,
        "delta_p_mean_pct": None, "delta_p_rms_pct": None,
    }
    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _PLAIN_PPL_RE.match(line) or _PPL_Q_RE.match(line)
        if m:
            res["ppl"], res["ppl_unc"] = float(m.group(1)), float(m.group(2))
            continue
        m = _PPL_BASE_RE.match(line)
        if m:
            res["ppl_base"] = float(m.group(1))
            res["ppl_base_unc"] = float(m.group(2))
            continue
        m = _PPL_RATIO_RE.match(line)
        if m:
            res["ppl_ratio"] = float(m.group(1))
            res["ppl_ratio_unc"] = float(m.group(2))
            continue
        m = _PPL_DIFF_RE.match(line)
        if m:
            res["ppl_diff"] = float(m.group(1))
            res["ppl_diff_unc"] = float(m.group(2))
            continue
        m = _LOG_PPL_RATIO_RE.match(line)
        if m:
            res["log_ppl_ratio"] = float(m.group(1))
            res["log_ppl_ratio_unc"] = float(m.group(2))
            continue
        m = _COR_RE.match(line)
        if m:
            res["correlation_pct"] = float(m.group(1))
            continue
        m = _KL_MEAN_RE.match(line)
        if m:
            res["kl_mean"] = float(m.group(1))
            res["kl_mean_unc"] = float(m.group(2))
            continue
        m = _KL_EXTREME_RE.match(line)
        if m:
            suffix = {"maximum": "max", "median": "median", "minimum": "min"}
            res[f"kl_{suffix[m.group(1).lower()]}"] = float(m.group(2))
            continue
        m = _KL_PERCENTILE_RE.match(line)
        if m:
            res["kl_percentiles"][m.group(1)] = float(m.group(2))
            continue
        m = _TOP1_RE.match(line)
        if m:
            res["top1_agree"] = float(m.group(1)) / 100.0
            res["top1_agree_unc"] = float(m.group(2)) / 100.0
            continue
        m = _MEAN_DP_RE.match(line)
        if m and "PPL" not in line:
            vals = _line_floats(line)
            if vals:
                res["delta_p_mean_pct"] = vals[0]
            continue
        m = _RMS_DP_RE.match(line)
        if m:
            vals = _line_floats(line)
            if vals:
                res["delta_p_rms_pct"] = vals[0]
            continue
    return res


def _line_floats(line: str) -> List[float]:
    """All floats on *line* (loose extraction for the delta-p lines)."""
    out: List[float] = []
    for tok in line.replace(":", " ").replace("±", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def build_perplexity_command(
    binary: str,
    model: str,
    corpus: str,
    kl_base: Optional[str] = None,
    kl_divergence: bool = False,
    chunks: Optional[int] = None,
    ngl: Optional[int] = None,
    ctx: int = DEFAULT_CTX,
    stride: Optional[int] = None,
) -> List[str]:
    """Build a llama-perplexity argv using only flags verified in pinned v0.6.0.

    Verified (third_party/llama.cpp/common/arg.cpp:1751,2508,2515,2522 and
    examples/model-conversion/scripts/utils/perplexity-run.sh): -m, -f, -c,
    --chunks, --ppl-stride, -ngl, --kl-divergence-base FNAME, --kl-divergence.

    Modes:
      * plain     : no kl flags -> only "Final estimate: PPL" is printed.
      * reference : kl_base set, kl_divergence=False -> *writes* base logits.
      * candidate : kl_base set + kl_divergence=True -> computes KL vs base.

    Reference and candidates MUST share corpus/ctx/stride/chunks or the base
    file will not line up with candidate tokens (the tool then fails loudly).
    """
    cmd = [binary, "-m", model, "-f", corpus, "-c", str(ctx)]
    if kl_base is not None:
        cmd += ["--kl-divergence-base", kl_base]
    if kl_divergence:
        cmd += ["--kl-divergence"]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]
    if stride is not None:
        cmd += ["--ppl-stride", str(stride)]
    if ngl is not None:
        cmd += ["-ngl", str(ngl)]
    return cmd


@dataclass
class QualityResult:
    """Aggregated quality measurement over >= 1 repeats (median + spread)."""

    mode: str  # "plain" | "reference" | "kl"
    cmd: List[str]
    metrics: Dict[str, Optional[float]]  # kl_mean / top1_agree / ppl (medians)
    spread: Dict[str, Any]  # per-repeat samples + full parse detail
    repeats: int
    duration_s: float = 0.0
    vram_peak_mb: Optional[int] = None
    vram_baseline_mb: Optional[int] = None


def aggregate_quality(parsed_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-repeat parses into median metrics + sample arrays in spread.

    Follows the project rule "report median + spread": the stored metric is the
    MEDIAN across repeats, and every per-repeat value is kept in ``spread`` so
    the report shows the spread (and flags n<3).
    """
    def samples(key: str) -> List[float]:
        return [float(p[key]) for p in parsed_runs if p.get(key) is not None]

    kl = samples("kl_mean")
    top1 = samples("top1_agree")
    ppl = samples("ppl")
    metrics: Dict[str, Optional[float]] = {
        "kl_mean": statistics.median(kl) if kl else None,
        "top1_agree": statistics.median(top1) if top1 else None,
        "ppl": statistics.median(ppl) if ppl else None,
    }
    spread: Dict[str, Any] = {
        "kl_samples": kl,
        "top1_samples": top1,
        "ppl_samples": ppl,
    }
    if parsed_runs:
        # Full parse of the first repeat: carries kld percentiles, uncertainties
        # and the ppl(base)/ratio context that don't fit the flat metrics.
        spread["quality_detail"] = parsed_runs[0]
    return {"metrics": metrics, "spread": spread}


def has_measurement(parsed: Dict[str, Any]) -> bool:
    """True if *parsed* contains at least one measured quality value."""
    return any(
        parsed.get(k) is not None for k in ("ppl", "kl_mean", "top1_agree")
    )


def run_quality(
    model: str,
    corpus: str,
    mode: str = "plain",
    kl_base: Optional[str] = None,
    repeats: int = DEFAULT_REPEATS,
    chunks: Optional[int] = None,
    ngl: Optional[int] = None,
    ctx: int = DEFAULT_CTX,
    stride: Optional[int] = None,
    binary: Optional[str] = None,
    timeout_s: float = runner.DEFAULT_TIMEOUT_S,
) -> QualityResult:
    """Run llama-perplexity *repeats* times and aggregate median + spread.

    mode:
      * ``"plain"``      - perplexity only.
      * ``"reference"``  - write base logits to *kl_base* (the file's
        existence is the success criterion).
      * ``"kl"``         - compute KL/top1/ppl against the existing *kl_base*.

    Raises RuntimeError (never fabricates) when the run fails, times out, the
    tool is missing, the reference file was not written, or the output carries
    no measurable summary.
    """
    from pathlib import Path

    if mode not in ("plain", "reference", "kl"):
        raise ValueError(f"unknown quality mode: {mode!r}")
    if mode in ("reference", "kl") and not kl_base:
        raise ValueError(f"mode {mode!r} requires kl_base")
    if mode == "kl" and not Path(kl_base).exists():
        raise RuntimeError(
            f"KL base file not found: {kl_base}. Generate it first with "
            "'potato bench --save-kl-base --corpus <txt> --quality-only' using "
            "the reference (highest-precision) model."
        )

    bin_path = runner.find_llama_binary("llama-perplexity", binary)
    cmd = build_perplexity_command(
        bin_path, model, corpus,
        kl_base=kl_base if mode in ("reference", "kl") else None,
        kl_divergence=(mode == "kl"),
        chunks=chunks, ngl=ngl, ctx=ctx, stride=stride,
    )
    print(">> " + " ".join(cmd))

    parsed_runs: List[Dict[str, Any]] = []
    duration = 0.0
    last_run = None
    for i in range(max(1, repeats)):
        try:
            last_run = runner.run_benchmark(cmd, timeout_s=timeout_s)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"llama-perplexity executable not found at '{bin_path}'. Build "
                "it first: scripts/build_cuda.ps1 or scripts/build_vulkan.ps1 "
                "(both build llama-perplexity), or pass a --binary path."
            ) from exc
        duration += last_run.duration_s
        if last_run.timed_out:
            raise RuntimeError(
                f"llama-perplexity timed out after {timeout_s:g}s on repeat "
                f"{i + 1}/{repeats}. Use --chunks to limit the corpus or raise "
                "--timeout."
            )
        if last_run.exit_code != 0:
            raise RuntimeError(
                f"llama-perplexity failed with exit code {last_run.exit_code}\n"
                + (last_run.stderr or last_run.stdout)
            )
        parsed_runs.append(
            parse_perplexity_output(last_run.stdout + "\n" + last_run.stderr)
        )

    agg = aggregate_quality(parsed_runs)
    detail = agg["spread"].get("quality_detail") or {}
    output_tail = (last_run.stderr or last_run.stdout) if last_run else ""

    if mode == "reference":
        # Success criterion is the base file itself; ppl is a bonus.
        if not Path(kl_base).exists() or Path(kl_base).stat().st_size <= 0:
            raise RuntimeError(
                f"reference run finished but produced no non-empty logits file "
                f"at {kl_base}.\nOutput:\n{output_tail}"
            )
    elif mode == "kl" and detail.get("kl_mean") is None:
        raise RuntimeError(
            "llama-perplexity ran but printed no KL summary. The pinned tool "
            "skips it below 100 compared tokens - raise --chunks, and keep "
            f"corpus/ctx/chunks identical to the run that wrote {kl_base}.\n"
            f"Output:\n{output_tail}"
        )
    elif mode == "plain" and detail.get("ppl") is None:
        raise RuntimeError(
            "llama-perplexity ran but printed no perplexity estimate.\n"
            f"Output:\n{output_tail}"
        )

    return QualityResult(
        mode=mode,
        cmd=cmd,
        metrics=agg["metrics"],
        spread=agg["spread"],
        repeats=max(1, repeats),
        duration_s=round(duration, 3),
        vram_peak_mb=last_run.vram_peak_mb if last_run else None,
        vram_baseline_mb=last_run.vram_baseline_mb if last_run else None,
    )

