"""PotatoServe command-line interface.

Implements the project-spec section 7 CLI contract as phases land: ``probe``
(hardware fingerprint), ``bench`` (speed + VRAM + Phase 2 quality: KL/top-1/
perplexity), ``quant`` (GGUF variants + imatrix -> quant_manifest.json) and
``report`` (median+spread reports, A/B comparisons, quality-vs-size curve) are
wired up; ``agent-eval``, ``tune``, ``serve`` remain stubbed with clear "not
yet implemented" messages so the surface stays visible.

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

# Subcommands that have real implementations.
IMPLEMENTED = ("probe", "quant", "bench", "report")


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
    # Phase 2 quality gate (FR-3): llama-perplexity runs alongside llama-bench.
    p_bench.add_argument(
        "--corpus",
        default=None,
        help="Text corpus (e.g. wiki.test.raw): measure perplexity, and with "
             "--kl-base also KL divergence + top-1 agreement.",
    )
    p_bench.add_argument(
        "--kl-base",
        default=None,
        help="Reference logits file (.kld). Read with --corpus to compute KL, "
             "or written when combined with --save-kl-base.",
    )
    p_bench.add_argument(
        "--save-kl-base",
        action="store_true",
        help="Generate the reference logits file at --kl-base (run this with "
             "the highest-precision model; candidates reuse it later).",
    )
    p_bench.add_argument(
        "--chunks",
        type=int,
        default=None,
        help="Max corpus chunks for llama-perplexity/--imatrix (limits .kld "
             "size: ~n_vocab x 2 bytes per token; keep reference and "
             "candidates identical).",
    )
    p_bench.add_argument(
        "--quality-repeats",
        type=int,
        default=3,
        help="Perplexity/KL repeats (default 3: median + spread, rules.md).",
    )
    p_bench.add_argument(
        "--quality-only",
        action="store_true",
        help="Skip llama-bench; only run the quality pass (reference phase).",
    )
    p_bench.add_argument(
        "--quality-ctx",
        type=int,
        default=512,
        help="Context for llama-perplexity (-c; default 512; must match "
             "between reference and candidates).",
    )
    p_bench.add_argument(
        "--quality-binary",
        default=None,
        help="Path to the llama-perplexity binary (default: auto-detect).",
    )
    p_bench.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact commands and exit without running anything.",
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
    p_report.add_argument(
        "--curve",
        action="store_true",
        help="Render the quality-vs-size curve (Phase 2 exit artifact) from "
             "all stored records that carry a model size.",
    )

    p_quant = sub.add_parser(
        "quant",
        help="Quantize a source GGUF into variants (+ optional imatrix) and "
             "record them in quant_manifest.json (spec section 7).",
    )
    p_quant.add_argument(
        "source",
        help="Source GGUF (highest precision you have, e.g. F16). "
             "Raw Hugging Face repos need convert_hf_to_gguf.py + its Python "
             "requirements, which require owner approval (docs/rules.md).",
    )
    p_quant.add_argument(
        "--types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="Quant types to produce (default: Q3_K_M IQ4_XS Q4_K_M Q5_K_M Q6_K, "
             "the project-spec section 3 ladder).",
    )
    p_quant.add_argument(
        "--out-dir",
        default=None,
        help="Directory for variant GGUFs + quant_manifest.json (default: "
             "the source's directory).",
    )
    p_quant.add_argument(
        "--manifest",
        default=None,
        help="Explicit manifest path (default: <out-dir>/quant_manifest.json).",
    )
    p_quant.add_argument(
        "--imatrix",
        default=None,
        help="Use an existing importance-matrix file (.gguf/.dat).",
    )
    p_quant.add_argument(
        "--imatrix-from",
        default=None,
        metavar="CORPUS",
        help="Compute an importance matrix from this text corpus if --imatrix "
             "is not given (wiki.train.raw by convention).",
    )
    p_quant.add_argument(
        "--imatrix-out",
        default=None,
        help="Where to write a generated imatrix (default: <out-dir>/<src>.imatrix.gguf).",
    )
    p_quant.add_argument("-ngl", "--ngl", type=int, default=None,
                         help="GPU layers for llama-imatrix (-ngl 99 recommended).")
    p_quant.add_argument("--chunks", type=int, default=None,
                         help="Max corpus chunks for llama-imatrix.")
    p_quant.add_argument("--threads", type=int, default=None,
                         help="Threads for llama-quantize (default: tool default).")
    p_quant.add_argument("--timeout", type=float, default=1800.0,
                         help="Per-tool timeout in seconds (default: 1800).")
    p_quant.add_argument("--quant-binary", default=None,
                         help="Path to llama-quantize (default: auto-detect).")
    p_quant.add_argument("--imatrix-binary", default=None,
                         help="Path to llama-imatrix (default: auto-detect).")
    p_quant.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact commands and exit without writing anything.",
    )

    for name in COMMANDS:
        if name in IMPLEMENTED:
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
                corpus=args.corpus,
                kl_base=args.kl_base,
                save_kl_base=args.save_kl_base,
                chunks=args.chunks,
                quality_repeats=args.quality_repeats,
                quality_only=args.quality_only,
                quality_ctx=args.quality_ctx,
                quality_binary=args.quality_binary,
                dry_run=args.dry_run,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"potato bench: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "quant":
        from potato import quant as quant_mod

        try:
            result = quant_mod.run_quant(
                source=args.source,
                types=args.types,
                out_dir=args.out_dir,
                manifest_path=args.manifest,
                imatrix=args.imatrix,
                imatrix_from=args.imatrix_from,
                imatrix_out=args.imatrix_out,
                ngl=args.ngl,
                chunks=args.chunks,
                threads=args.threads,
                quant_binary=args.quant_binary,
                imatrix_binary=args.imatrix_binary,
                timeout_s=args.timeout,
                dry_run=args.dry_run,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"potato quant: {exc}", file=sys.stderr)
            return 1
        if args.dry_run:
            print("dry run - commands that would execute:")
            for cmd in result["commands"]:
                print("  " + " ".join(cmd))
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
                *(["--curve"] if args.curve else []),
            ]
        )

    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
