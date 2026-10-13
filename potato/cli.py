"""PotatoServe command-line interface (Phase 0).

Implements the subset of the project-spec section 7 CLI contract that exists in
Phase 1. ``potato probe`` (hardware fingerprint), ``potato bench`` (baseline
llama-bench run with live VRAM sampling -> section-8 record) and ``potato
report`` (median+spread reports and A/B comparison of stored records) are wired
up; the remaining commands (``quant``, ``agent-eval``, ``tune``, ``serve``) are
stubbed with clear "not yet implemented" messages so the surface is visible.

Dependency policy (docs/rules.md): the project spec anticipates ``typer`` for
the CLI, but no dependency may be added without owner approval. This dispatcher
therefore uses the standard-library ``argparse``. If/when ``typer`` is approved,
this file is the single place to swap the framework.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

# Commands from project-spec section 7, in contract order.
COMMANDS = ["probe", "quant", "bench", "agent-eval", "tune", "serve", "report"]


def _not_implemented(name: str) -> int:
    print(f"potato {name}: not implemented yet. See docs/phases.md for the roadmap.")
    return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="potato",
        description="Auto-tune llama.cpp for low-VRAM GPUs, with honest measurement.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_probe = sub.add_parser(
        "probe", help="Detect hardware and write hardware.json (spec section 7)."
    )
    p_probe.add_argument(
        "--out",
        default="hardware.json",
        help="Output path for the hardware fingerprint (default: hardware.json).",
    )

    p_bench = sub.add_parser(
        "bench", help="Run a baseline llama-bench and store a section-8 record."
    )
    p_bench.add_argument(
        "-m", "--model", required=True, help="Path to a GGUF model."
    )
    p_bench.add_argument(
        "-r", "--repeats", type=int, default=3, help="llama-bench repeats (min 3 per spec)."
    )
    p_bench.add_argument("-p", "--pp", type=int, default=512, help="Prefill tokens (-p).")
    p_bench.add_argument("-n", "--tg", type=int, default=128, help="Decode tokens (-n).")
    p_bench.add_argument("-ngl", "--ngl", type=int, default=None, help="GPU layers (-ngl).")
    p_bench.add_argument(
        "-fa", "--flash-attn", default=None, help="Flash attention on|off|auto (-fa)."
    )
    p_bench.add_argument("--binary", default=None, help="Path to llama-bench binary.")
    p_bench.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write the result record JSON (default: results).",
    )
    p_bench.add_argument("--run-id", default=None, help="Override the generated run id.")
    p_bench.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Kill llama-bench after this many seconds (default: 900).",
    )
    p_bench.add_argument(
        "--no-report",
        action="store_true",
        help="Store the record without printing the median+spread report.",
    )

    p_report = sub.add_parser(
        "report",
        help="Render stored records as median+spread reports / A/B comparisons.",
    )
    p_report.add_argument(
        "--results-dir",
        default="results",
        help="Directory holding the result record JSON files (default: results).",
    )
    p_report.add_argument("--run-id", default=None, help="Report a single stored record.")
    p_report.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE", "CAND"),
        help="Compare two stored records (base first, candidate second).",
    )
    p_report.add_argument(
        "--metric",
        default=None,
        help="With --compare, restrict output to one metric (e.g. decode_tps).",
    )
    p_report.add_argument(
        "--list", action="store_true", help="List stored run_ids and exit."
    )

    for name in COMMANDS:
        if name in ("probe", "bench", "report"):
            continue
        sub.add_parser(name, help=f"(not yet implemented) potato {name}")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "probe":
        from potato import probe as probe_mod

        return probe_mod.main(["--out", args.out])
    if args.command == "bench":
        from pathlib import Path

        from potato import bench as bench_mod

        if args.repeats < 3:
            print(
                "warning: spec (section 7) requires >= 3 repeats; continuing anyway.",
                file=sys.stderr,
            )
        try:
            bench_mod.run_baseline(
                model=args.model,
                results_dir=Path(args.results_dir),
                repeats=args.repeats,
                pp=args.pp,
                tg=args.tg,
                binary=args.binary,
                ngl=args.ngl,
                flash_attn=args.flash_attn,
                run_id=args.run_id,
                timeout_s=args.timeout,
                report=not args.no_report,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"potato bench: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "report":
        from potato import report as report_mod

        return report_mod.main(
            [
                "--results-dir",
                args.results_dir,
                *(["--run-id", args.run_id] if args.run_id else []),
                *(["--compare", *args.compare] if args.compare else []),
                *(["--metric", args.metric] if args.metric else []),
                *(["--list"] if args.list else []),
            ]
        )

    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
