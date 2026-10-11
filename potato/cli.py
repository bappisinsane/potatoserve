"""PotatoServe command-line interface (Phase 0).

Implements the subset of the project-spec section 7 CLI contract that exists in
Phase 0. Only ``potato probe`` is wired up now; the remaining commands
(``quant``, ``bench``, ``agent-eval``, ``tune``, ``serve``, ``report``) are
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
    print(f"potato {name}: not implemented yet (Phase 0). See docs/phases.md.")
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

    for name in COMMANDS:
        if name == "probe":
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

    return _not_implemented(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
