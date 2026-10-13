"""Append-only results store for Phase 1.

Implements ADR-5 from architecture.md: results are APPEND-ONLY. We never
overwrite or mutate a previously written run; every save writes a NEW file
named ``<run_id>.json`` under the results directory. If a file with that name
already exists we refuse by default (the caller must choose a fresh run_id),
which prevents an accidental clobber of measured data.

Stdlib only - JSON on disk, no database, no new dependency (docs/rules.md).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

DEFAULT_RESULTS_DIR = os.path.join("results")


def _path_for(results_dir: str, run_id: str) -> str:
    return os.path.join(results_dir, f"{run_id}.json")


def save_record(
    record: Dict[str, Any],
    results_dir: str = DEFAULT_RESULTS_DIR,
    overwrite: bool = False,
) -> str:
    """Persist *record* as ``<results_dir>/<run_id>.json``; return the path.

    Append-only (ADR-5): if the target exists and ``overwrite`` is False this
    raises FileExistsError rather than destroying prior measured data.
    """
    run_id = record.get("run_id")
    if not run_id:
        raise ValueError("record has no run_id; cannot store append-only")

    os.makedirs(results_dir, exist_ok=True)
    path = _path_for(results_dir, run_id)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing result {path} (append-only store; "
            f"choose a unique run_id or pass overwrite=True)"
        )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def load_record(
    run_id: str, results_dir: str = DEFAULT_RESULTS_DIR
) -> Optional[Dict[str, Any]]:
    """Load a stored record by run_id, or None if it does not exist."""
    path = _path_for(results_dir, run_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_run_ids(results_dir: str = DEFAULT_RESULTS_DIR) -> List[str]:
    """Return stored run_ids (filenames without .json), sorted for determinism."""
    if not os.path.isdir(results_dir):
        return []
    ids = [f[:-5] for f in os.listdir(results_dir) if f.endswith(".json")]
    return sorted(ids)


def load_all(results_dir: str = DEFAULT_RESULTS_DIR) -> List[Dict[str, Any]]:
    """Load every stored record, in run_id order."""
    records: List[Dict[str, Any]] = []
    for run_id in list_run_ids(results_dir):
        rec = load_record(run_id, results_dir)
        if rec is not None:
            records.append(rec)
    return records
