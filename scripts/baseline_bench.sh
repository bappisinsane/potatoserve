#!/usr/bin/env bash
# Baseline benchmark runner (Phase 0 step 4): run llama-bench with 3 repeats and
# store a project-spec section-8 result record under results/.
#
# Thin wrapper around `python -m potato.cli bench`. Uses verified v0.6.0 flags,
# defaults to 3 repeats, and invents no numbers (fails cleanly if llama-bench or
# the model is missing).
#
# Usage:
#   ./baseline_bench.sh <model.gguf> [repeats] [pp] [tg] [ngl] [flash_attn]
#   MODEL=path REPEATS=3 ./baseline_bench.sh
set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [ -z "$MODEL" ]; then
  echo "ERROR: model path required. Usage: $0 <model.gguf> [repeats] [pp] [tg] [ngl] [fa]" >&2
  exit 2
fi

REPEATS="${2:-${REPEATS:-3}}"
PP="${3:-${PP:-512}}"
TG="${4:-${TG:-128}}"
NGL="${5:-${NGL:-99}}"
FA="${6:-${FA:-on}}"
RESULTS_DIR="${RESULTS_DIR:-results}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ">> python -m potato.cli bench --model '$MODEL' --repeats $REPEATS --pp $PP --tg $TG --ngl $NGL --flash-attn $FA --results-dir $RESULTS_DIR"
python -m potato.cli bench --model "$MODEL" --repeats "$REPEATS" --pp "$PP" \
  --tg "$TG" --ngl "$NGL" --flash-attn "$FA" --results-dir "$RESULTS_DIR"

echo "Baseline complete. Records in: $RESULTS_DIR"
