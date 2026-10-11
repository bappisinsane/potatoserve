"""Result-record schema for PotatoServe.

Implements the JSON result record defined in docs/project-spec.md section 8:

    {run_id, date, llama_cpp_commit, build_flags, driver, os, gpu,
     model_id, model_sha256, config,
     metrics:{prefill_tps, decode_tps, vram_peak_mb, kl_mean,
              top1_agree, ppl, agent_success},
     repeats, spread}

This module is dependency-free (standard library only) so it works in CI and
before any optional dependency is approved (see docs/rules.md).

It provides:
  * the canonical field lists (TOP_LEVEL_FIELDS, METRIC_FIELDS),
  * a `ResultRecord` dataclass + `to_dict()` that always emits every §8 key
    (missing metrics are recorded as ``null`` rather than omitted, so a record
    is self-describing about what was *not* measured),
  * light validation (`validate`) used by tests and by ``potato bench``.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

# --- Canonical §8 fields (do not rename without updating the spec) ---
TOP_LEVEL_FIELDS: List[str] = [
    "run_id",
    "date",
    "llama_cpp_commit",
    "build_flags",
    "driver",
    "os",
    "gpu",
    "model_id",
    "model_sha256",
    "config",
    "metrics",
    "repeats",
    "spread",
]

# Numeric metrics from §8. Any may be None if not measured for a given run.
METRIC_FIELDS: List[str] = [
    "prefill_tps",
    "decode_tps",
    "vram_peak_mb",
    "kl_mean",
    "top1_agree",
    "ppl",
    "agent_success",
]

SCHEMA_VERSION = 1


def _blank_metrics() -> Dict[str, Optional[float]]:
    """Every §8 metric present but unset (null) until actually measured."""
    return {name: None for name in METRIC_FIELDS}


@dataclass
class ResultRecord:
    """A single benchmark result, matching project-spec section 8.

    Only ``run_id`` is strictly required to construct; every other field has a
    safe default so partial records (e.g. a probe with no bench yet) validate.
    """

    run_id: str
    date: Optional[str] = None  # ISO 8601 UTC; filled by new_record() if omitted
    llama_cpp_commit: Optional[str] = None
    build_flags: Optional[Dict[str, Any]] = None
    driver: Optional[str] = None
    os: Optional[str] = None
    gpu: Optional[str] = None
    model_id: Optional[str] = None
    model_sha256: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Optional[float]] = field(default_factory=_blank_metrics)
    repeats: Optional[int] = None
    spread: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Guarantee every §8 metric key is present.

        Callers may pass a partial ``metrics`` dict (e.g. a baseline that only
        measured prefill/decode throughput). We merge it over the blank set so
        the record always exposes all §8 metrics, with unmeasured ones as null.
        """
        merged = _blank_metrics()
        merged.update(self.metrics or {})
        self.metrics = merged

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict that contains exactly the §8 keys, in order."""
        out: Dict[str, Any] = {}
        out["schema_version"] = SCHEMA_VERSION
        for name in TOP_LEVEL_FIELDS:
            value = getattr(self, name)
            if name == "date" and value is None:
                value = _dt.datetime.now(_dt.timezone.utc).isoformat()
            out[name] = value
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultRecord":
        """Build a ResultRecord from a dict, ignoring unknown keys.

        Metrics are merged over the blank set so a partial metrics dict still
        yields all §8 metric keys (unmeasured ones stay null).
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        metrics = _blank_metrics()
        metrics.update(data.get("metrics") or {})
        kwargs["metrics"] = metrics
        return cls(**kwargs)


def new_record(run_id: str, **kwargs: Any) -> ResultRecord:
    """Convenience constructor that stamps an ISO-8601 UTC date if not given."""
    if "date" not in kwargs or kwargs["date"] is None:
        kwargs["date"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return ResultRecord(run_id=run_id, **kwargs)


def validate(record: ResultRecord | Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems; empty list means valid.

    Structural validation only (we do not invent or check numeric ranges here
    — measurement happens on the user's hardware, per the project rules).
    """
    problems: List[str] = []
    data = record.to_dict() if isinstance(record, ResultRecord) else dict(record)

    for key in TOP_LEVEL_FIELDS:
        if key not in data:
            problems.append(f"missing top-level field: {key}")

    if not data.get("run_id"):
        problems.append("run_id must be a non-empty string")

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        problems.append("metrics must be an object")
    else:
        for key in METRIC_FIELDS:
            if key not in metrics:
                problems.append(f"metrics missing key: {key}")

    return problems
