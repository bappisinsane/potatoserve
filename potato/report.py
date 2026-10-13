"""Median + spread reporting for Phase 1.

Implements the reporting half of the Phase 1 exit criterion: a report that
shows, for each measured metric, the MEDIAN across repeats plus the SPREAD,
and that is explicit about what was NOT measured (null stays null - we never
invent a number).

Core rule from docs/rules.md enforced here: never claim a difference that
falls inside the noise band. ``render_record_report`` therefore always prints
the spread next to the median, and ``compare`` labels two values "within
noise" rather than declaring a winner when their spread bands overlap.

Stdlib only (statistics, argparse) - no new dependency.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

from potato import store
from potato.schema import METRIC_FIELDS, validate

# Human labels for the section-8 metrics (order matches METRIC_FIELDS).
METRIC_LABELS: Dict[str, str] = {
    "prefill_tps": "Prefill throughput (tok/s)",
    "decode_tps": "Decode throughput (tok/s)",
    "vram_peak_mb": "Peak VRAM (MiB)",
    "kl_mean": "KL divergence (mean)",
    "top1_agree": "Top-1 agreement",
    "ppl": "Perplexity",
    "agent_success": "Agent task success rate",
}

# Where a metric's per-repeat samples live inside a record's ``spread`` dict.
# ``potato.bench`` (llama-bench) stores them as ``prefill_samples_ts`` /
# ``decode_samples_ts``. A different writer may instead store a ready-made
# ``{median, stdev, min, max, n}`` dict keyed by the metric name itself; both
# layouts are honoured, in that order, by ``_metric_summary_from_record``.
SAMPLE_KEYS: Dict[str, str] = {
    "prefill_tps": "prefill_samples_ts",
    "decode_tps": "decode_samples_ts",
}

# Metrics where a SMALLER number is the better outcome (drives verdict wording).
LOWER_IS_BETTER = frozenset({"kl_mean", "ppl", "vram_peak_mb"})


def summarize_samples(values: List[float]) -> Dict[str, Optional[float]]:
    """Median + spread for a list of numeric samples (population statistics).

    Returns median, stdev (population), min, max, and n. For a single sample,
    stdev is 0.0. Empty input yields all-None so callers can distinguish
    "not measured" from "measured zero".

    Non-numeric entries (including bools, which are ints in Python) are ignored
    rather than coerced, so a malformed spread cannot silently skew the median.
    """
    nums = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    n = len(nums)
    if n == 0:
        return {"median": None, "stdev": None, "min": None, "max": None, "n": 0}
    return {
        "median": statistics.median(nums),
        "stdev": statistics.pstdev(nums) if n > 1 else 0.0,
        "min": min(nums),
        "max": max(nums),
        "n": n,
    }


def _summary(median: Any, stdev: Any, lo: Any, hi: Any, n: int) -> Dict[str, Any]:
    """Build the canonical summary dict used everywhere in this module."""
    return {"median": median, "stdev": stdev, "min": lo, "max": hi, "n": n}


def _metric_summary_from_record(record: Dict[str, Any], metric: str) -> Dict[str, Any]:
    """Summarize one metric from a record, preferring real per-repeat samples.

    Resolution order (first hit wins):
      1. a ready-made ``spread[<metric>]`` dict with ``n >= 1`` (a writer that
         already computed median/stdev);
      2. the per-repeat sample list ``spread[<sample_key>]`` -> recomputed via
         ``summarize_samples`` (this is what ``potato.bench`` writes, and it is
         the case that actually carries the >=3 repeats the spec demands);
      3. a single scalar ``metrics[<metric>]`` (n=1, stdev 0).

    Falling through to (3) with an n=1/0.0 stdev would hide the real spread, so
    step (2) must come before it. All-None means "not measured".
    """
    spread = record.get("spread") or {}

    # 1. Pre-computed summary keyed by the metric name.
    entry = spread.get(metric)
    if isinstance(entry, dict) and entry.get("n", 0) >= 1:
        return _summary(
            entry.get("median"), entry.get("stdev"),
            entry.get("min"), entry.get("max"), entry.get("n"),
        )

    # 2. Per-repeat samples (e.g. spread["decode_samples_ts"]) -> real spread.
    sample_key = SAMPLE_KEYS.get(metric)
    if sample_key:
        raw = spread.get(sample_key)
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            s = summarize_samples(list(raw))
            if s["n"] >= 1:
                return _summary(s["median"], s["stdev"], s["min"], s["max"], s["n"])

    # 3. Single stored scalar value.
    value = (record.get("metrics") or {}).get(metric)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _summary(float(value), 0.0, float(value), float(value), 1)

    return _summary(None, None, None, None, 0)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "not measured"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def render_record_report(record: Dict[str, Any]) -> str:
    """Render a single stored record as a human-readable median+spread report."""
    lines: List[str] = []
    lines.append("PotatoServe benchmark report")
    lines.append("=" * 40)
    lines.append(f"run_id          : {record.get('run_id', '(unknown)')}")
    lines.append(f"date            : {record.get('date', '(unknown)')}")
    lines.append(f"llama_cpp_commit: {record.get('llama_cpp_commit', '(unknown)')}")
    lines.append(f"model_id        : {record.get('model_id', '(unknown)')}")
    lines.append(f"gpu             : {record.get('gpu', '(unknown)')}")
    lines.append(f"repeats         : {record.get('repeats', '(unknown)')}")
    lines.append("")

    problems = validate(record)
    if problems:
        lines.append("SCHEMA WARNINGS:")
        for p in problems:
            lines.append(f"  - {p}")
        lines.append("")

    lines.append(
        f"{'metric':32s} {'median':>14s} {'stdev':>12s} "
        f"{'min':>11s} {'max':>11s} {'n':>4s}"
    )
    lines.append("-" * 88)
    for metric in METRIC_FIELDS:
        s = _metric_summary_from_record(record, metric)
        label = METRIC_LABELS.get(metric, metric)
        if s["n"] == 0:
            lines.append(
                f"{label:32s} {'not measured':>14s} {'':>12s} "
                f"{'':>11s} {'':>11s} {0:>4d}"
            )
        else:
            lines.append(
                f"{label:32s} {_fmt(s['median']):>14s} "
                f"{_fmt(s['stdev']):>12s} {_fmt(s['min']):>11s} "
                f"{_fmt(s['max']):>11s} {s['n']:>4d}"
            )
    lines.append("")
    lines.append(
        "Note: metrics marked 'not measured' were not captured this phase; "
        "they are intentionally null, not zero."
    )

    # Flag a record that claims >= 3 repeats but only stored one sample, and
    # vice versa: the spec (project-spec section 8) requires >= 3 per point.
    for metric in METRIC_FIELDS:
        s = _metric_summary_from_record(record, metric)
        if 0 < s["n"] < 3:
            lines.append(
                f"WARNING: {metric} has only {s['n']} sample(s); the spec "
                "requires >= 3 repeats, so its spread is not trustworthy."
            )
    return "\n".join(lines)


def render_comparison(
    base: Dict[str, Any], cand: Dict[str, Any], metric: str
) -> str:
    """Render ``compare`` output as an aligned text block."""
    result = compare(base, cand, metric)
    label = METRIC_LABELS.get(metric, metric)
    a, b = result["base"], result["candidate"]
    lines = [
        f"Comparison: {label} ({metric})",
        "-" * 60,
        f"base      ({base.get('run_id', '?')}): "
        f"median={_fmt(a['median'])} stdev={_fmt(a['stdev'])} n={a['n']}",
        f"candidate ({cand.get('run_id', '?')}): "
        f"median={_fmt(b['median'])} stdev={_fmt(b['stdev'])} n={b['n']}",
        f"verdict   : {result['verdict']}",
    ]
    return "\n".join(lines)


def bands_overlap(
    a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]
) -> bool:
    """True if two median±stdev bands overlap (difference is inside noise)."""
    am, asd = a.get("median"), a.get("stdev")
    bm, bsd = b.get("median"), b.get("stdev")
    if am is None or bm is None:
        return False  # can't compare; not "within noise", just unknown
    asd = asd or 0.0
    bsd = bsd or 0.0
    return (am + asd) >= (bm - bsd) and (bm + bsd) >= (am - asd)


def compare(base: Dict[str, Any], cand: Dict[str, Any], metric: str) -> Dict[str, Any]:
    """Compare one metric between two records, honoring the no-noise rule.

    Verdicts, in order of precedence:
      * "incomparable (missing measurement)" - either side is null (unknown,
        which is NOT the same as "no difference");
      * "within noise (no claim)" - the median+/-stdev bands overlap, so per
        docs/rules.md we refuse to name a winner;
      * "<base|candidate> is better" - bands are disjoint, so the direction of
        the difference is stated (lower is better for LOWER_IS_BETTER metrics).
    """
    a = _metric_summary_from_record(base, metric)
    b = _metric_summary_from_record(cand, metric)
    result: Dict[str, Any] = {"metric": metric, "base": a, "candidate": b}
    if a["median"] is None or b["median"] is None:
        result["verdict"] = "incomparable (missing measurement)"
    elif bands_overlap(a, b):
        result["verdict"] = "within noise (no claim)"
    else:
        if metric in LOWER_IS_BETTER:
            better = "candidate" if b["median"] < a["median"] else "base"
        else:
            better = "candidate" if b["median"] > a["median"] else "base"
        result["verdict"] = f"{better} is better (bands do not overlap)"
    return result


def compare_all(
    base: Dict[str, Any], cand: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compare every section-8 metric between two records, in schema order."""
    return [compare(base, cand, metric) for metric in METRIC_FIELDS]


def render_compare_all(base: Dict[str, Any], cand: Dict[str, Any]) -> str:
    """Render every metric comparison as one aligned table (spec section 8)."""
    lines = [
        f"PotatoServe comparison: {cand.get('run_id', '?')} vs "
        f"{base.get('run_id', '?')} (candidate vs base)",
        "=" * 88,
        f"{'metric':32s} {'base':>16s} {'candidate':>16s}  verdict",
        "-" * 88,
    ]
    for metric in METRIC_FIELDS:
        r = compare(base, cand, metric)
        a, b = r["base"], r["candidate"]
        base_cell = (
            f"{_fmt(a['median'])}+-{_fmt(a['stdev'])}" if a["n"] else "not measured"
        )
        cand_cell = (
            f"{_fmt(b['median'])}+-{_fmt(b['stdev'])}" if b["n"] else "not measured"
        )
        label = METRIC_LABELS.get(metric, metric)
        lines.append(
            f"{label:32s} {base_cell:>16s} {cand_cell:>16s}  {r['verdict']}"
        )
    lines.append("")
    lines.append(
        "Note: 'within noise' means the median bands overlap, so no winner is "
        "claimed (docs/rules.md). 'incomparable' means the metric was never "
        "measured on one side - that is unknown, not equal."
    )
    return "\n".join(lines)



def _load(results_dir: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Load a stored record for the CLI, or None (the caller reports the miss)."""
    return store.load_record(run_id, results_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: ``python -m potato.report --run-id ID [--results-dir DIR]``.

    Modes:
      * ``--run-id ID``               -> median+spread report for one record
      * ``--compare BASE CAND``       -> all-metric comparison table
      * ``--compare BASE CAND --metric M`` -> single-metric comparison
      * ``--list``                    -> list stored run_ids
      * (no flags)                    -> report every stored record
    """
    parser = argparse.ArgumentParser(
        prog="potato report",
        description="Render stored benchmark records as median+spread reports.",
    )
    parser.add_argument(
        "--results-dir",
        default=store.DEFAULT_RESULTS_DIR,
        help="Directory holding the append-only JSON records.",
    )
    parser.add_argument("--run-id", default=None, help="Report a single record.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE", "CAND"),
        help="Compare two stored records (base first, candidate second).",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="With --compare, restrict the table to one section-8 metric.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List stored run_ids and exit."
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))

    if args.metric is not None and args.metric not in METRIC_FIELDS:
        parser.error(f"unknown metric '{args.metric}'; expected one of {METRIC_FIELDS}")

    if args.list:
        ids = store.list_run_ids(args.results_dir)
        if not ids:
            print(f"no stored records in {args.results_dir}")
            return 1
        print(f"{len(ids)} stored record(s) in {args.results_dir}:")
        for rid in ids:
            print(f"  {rid}")
        return 0

    if args.compare:
        base = _load(args.results_dir, args.compare[0])
        cand = _load(args.results_dir, args.compare[1])
        missing = [rid for rid, rec in zip(args.compare, (base, cand)) if rec is None]
        if missing:
            print(
                f"potato report: no stored record for {', '.join(missing)} "
                f"in {args.results_dir}",
                file=sys.stderr,
            )
            return 1
        assert base is not None and cand is not None  # narrowed by `missing` above
        if args.metric:
            print(render_comparison(base, cand, args.metric))
        else:
            print(render_compare_all(base, cand))
        return 0

    if args.run_id:
        record = _load(args.results_dir, args.run_id)
        if record is None:
            print(
                f"potato report: no stored record '{args.run_id}' in "
                f"{args.results_dir}",
                file=sys.stderr,
            )
            return 1
        print(render_record_report(record))
        return 0

    # Default: report every stored record (deterministic run_id order).
    records = store.load_all(args.results_dir)
    if not records:
        print(f"no stored records in {args.results_dir}")
        return 1
    for i, record in enumerate(records):
        if i:
            print()
        print(render_record_report(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
