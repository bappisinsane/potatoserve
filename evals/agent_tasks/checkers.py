"""Programmatic checkers for the agent-eval suite (Phase 3, spec section 6).

Each task in ``evals/agent_tasks/tasks/*.json`` names a ``checker`` object with a
``type`` key resolved here. Checkers are pure functions over three inputs:

* the task's **sandbox** (a temp directory the agent worked in),
* the **recorded tool calls** the model made (for ``tool_call_args``),
* the model's **final answer** text (for ``answer_contains``).

Design rules (docs/rules.md):
* stdlib only — the JSON-schema validator below is a deliberate subset
  (type / required / properties / enum / items / minimum / maximum); no
  ``jsonschema`` package (dependency policy);
* every check returns ``(passed, detail)`` — a failing check always explains
  why, so a recorded run is auditable;
* missing state (file never created) is a FAIL, never a harness error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DETAIL_LIMIT = 500  # max chars of any detail string kept in records


class CheckerError(Exception):
    """A task references an unknown checker or malformed parameters."""


def _read_text(root: Path, rel: str) -> Optional[str]:
    """File content relative to the sandbox, or None if it does not exist.

    Rejects paths escaping the sandbox (defense in depth — tools already do).
    """
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        return None
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _norm(text: str) -> str:
    """Normalize newlines only; whitespace inside lines is meaningful."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _clip(text: str) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= DETAIL_LIMIT else text[:DETAIL_LIMIT] + "..."


# --------------------------------------------------------------------------
# JSON-schema subset validator (no external dependency by design).
# Supported keywords: type, required, properties, enum, items, minimum,
# maximum. Unknown keywords are ignored rather than silently enforced.
# --------------------------------------------------------------------------

_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _is_number(v: Any) -> bool:
    # bool is an int in Python but never a JSON number.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_schema(value: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Validate *value* against a JSON-schema subset; return error strings."""
    errors: List[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        if expected_type == "integer":
            if not (isinstance(value, int) and not isinstance(value, bool)):
                errors.append(f"{path}: expected integer, got {type(value).__name__}")
                return errors
        elif expected_type == "number":
            if not _is_number(value):
                errors.append(f"{path}: expected number, got {type(value).__name__}")
                return errors
        else:
            js_type = _JSON_TYPES.get(expected_type)
            if js_type is None:
                errors.append(f"{path}: schema uses unsupported type {expected_type!r}")
            elif not isinstance(value, js_type):
                errors.append(
                    f"{path}: expected {expected_type}, got {type(value).__name__}"
                )
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if expected_type == "object" and isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                errors.extend(validate_schema(value[key], sub, f"{path}.{key}"))
    if expected_type == "array" and isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))
    return errors



# --------------------------------------------------------------------------
# Checkers. Signature: (task, state) -> (passed, detail).
# state = {"root": Path, "calls": [...], "final_answer": str}
# --------------------------------------------------------------------------

CheckerFn = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]]


def _file_equals(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    c = task["checker"]
    text = _read_text(state["root"], c["path"])
    if text is None:
        return False, f"{c['path']} does not exist"
    expected = _norm(c["content"])
    got = _norm(text)
    # Tolerate a trailing-newline difference (shell echo vs editor writes).
    if got.rstrip("\n") == expected.rstrip("\n"):
        return True, "exact content matches"
    return False, f"content differs: expected {_clip(expected)!r}, got {_clip(got)!r}"


def _file_contains(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    c = task["checker"]
    text = _read_text(state["root"], c["path"])
    if text is None:
        return False, f"{c['path']} does not exist"
    for needle in c.get("all_of", []):
        if needle not in text:
            return False, f"{c['path']}: missing {_clip(needle)!r}"
    for needle in c.get("none_of", []):
        if needle in text:
            return False, f"{c['path']}: still contains {_clip(needle)!r}"
    return True, "all required substrings present"


def _file_contains_any(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    c = task["checker"]
    for rel in c["paths"]:
        text = _read_text(state["root"], rel)
        if text is None:
            continue
        if all(n in text for n in c.get("all_of", [])):
            return True, f"{rel} matches"
    return False, f"none of {c['paths']} contains {c.get('all_of', [])!r}"


def _file_ends_with(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    c = task["checker"]
    text = _read_text(state["root"], c["path"])
    if text is None:
        return False, f"{c['path']} does not exist"
    if _norm(text).rstrip("\n").endswith(_norm(c["suffix"]).rstrip("\n")):
        return True, "suffix present at end"
    return False, f"{c['path']} does not end with {_clip(c['suffix'])!r}"


def _file_valid_json(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    import json as _json

    c = task["checker"]
    text = _read_text(state["root"], c["path"])
    if text is None:
        return False, f"{c['path']} does not exist"
    try:
        data = _json.loads(text)
    except ValueError as exc:
        return False, f"{c['path']} is not valid JSON: {exc}"
    for key, want in c.get("expect", {}).items():
        if not isinstance(data, dict) or key not in data:
            return False, f"missing key {key!r}"
        if data[key] != want:
            return False, f"key {key!r} is {data[key]!r}, expected {want!r}"
    return True, "valid JSON" + (" with expected keys" if c.get("expect") else "")


def _file_missing(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    target = (state["root"] / task["checker"]["path"]).resolve()
    if not target.is_relative_to(state["root"].resolve()):
        return False, "checker path escapes sandbox"
    if target.exists():
        return False, "file still exists"
    return True, "file removed"


def _files_exist(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    missing = []
    for rel in task["checker"]["paths"]:
        target = (state["root"] / rel).resolve()
        if not target.is_relative_to(state["root"].resolve()) or not target.exists():
            missing.append(rel)
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, f"all {len(task['checker']['paths'])} paths exist"


def _lines_sorted(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    c = task["checker"]
    text = _read_text(state["root"], c["path"])
    if text is None:
        return False, f"{c['path']} does not exist"
    lines = _norm(text).split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline is not a data line
    if c.get("expect_count") is not None and len(lines) != c["expect_count"]:
        return False, f"expected {c['expect_count']} lines, found {len(lines)}"
    if lines == sorted(lines):
        return True, f"{len(lines)} lines sorted"
    return False, "lines are not in ascending order"


def _answer_contains(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    c = task["checker"]
    answer = state.get("final_answer") or ""
    cs = c.get("case_sensitive", False)
    hay = answer if cs else answer.lower()
    for needle in c.get("all_of", []):
        n = needle if cs else needle.lower()
        if n not in hay:
            return False, f"final answer missing {_clip(needle)!r}"
    for needle in c.get("none_of", []):
        n = needle if cs else needle.lower()
        if n in hay:
            return False, f"final answer contains distractor {_clip(needle)!r}"
    if not answer.strip():
        return False, "no final answer (agent never stopped calling tools)"
    return True, "answer contains required phrases"


def _tool_call_args(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    """The model must have called ``checker.tool`` with schema-valid args that
    match ``checker.expect`` (subset match — extra args are allowed)."""
    c = task["checker"]
    tool = c["tool"]
    matching = [x for x in state.get("calls", []) if x.get("name") == tool]
    if not matching:
        return False, f"tool {tool!r} was never called"
    last = matching[-1]
    args = last.get("args")
    if not isinstance(args, dict):
        return False, f"args for {tool!r} are not an object: {args!r}"
    if c.get("validate_schema", True):
        for tool_def in task.get("tools", []):
            if tool_def.get("name") == tool and tool_def.get("parameters"):
                errors = validate_schema(args, tool_def["parameters"])
                if errors:
                    return False, "schema violations: " + "; ".join(errors[:5])
    for key, want in c.get("expect", {}).items():
        got = args.get(key)
        if _is_number(want) and _is_number(got):
            if float(want) != float(got):
                return False, f"{key}={got!r}, expected {want!r}"
        elif got != want:
            return False, f"{key}={got!r}, expected {want!r}"
    return True, f"valid call to {tool} with expected args"


def _all(task: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    """Composite: every sub-checker in ``checker.of`` must pass; the first
    failure is reported with its sub-checker type prefixed."""
    subs = task["checker"].get("of", [])
    if not subs:
        raise CheckerError(f"task {task.get('id')}: 'all' checker has no sub-checkers")
    for sub in subs:
        fn = CHECKERS.get(sub.get("type"))
        if fn is None:
            raise CheckerError(
                f"task {task.get('id')}: unknown checker type {sub.get('type')!r} "
                "inside 'all'"
            )
        try:
            ok, detail = fn({**task, "checker": sub}, state)
        except (KeyError, TypeError) as exc:
            raise CheckerError(
                f"task {task.get('id')}: malformed sub-checker {sub.get('type')!r}: {exc}"
            ) from exc
        if not ok:
            return False, f"[{sub.get('type')}] {detail}"
    return True, f"all {len(subs)} sub-checkers passed"


CHECKERS: Dict[str, CheckerFn] = {
    "all": _all,
    "file_equals": _file_equals,
    "file_contains": _file_contains,
    "file_contains_any": _file_contains_any,
    "file_ends_with": _file_ends_with,
    "file_valid_json": _file_valid_json,
    "file_missing": _file_missing,
    "files_exist": _files_exist,
    "lines_sorted": _lines_sorted,
    "answer_contains": _answer_contains,
    "tool_call_args": _tool_call_args,
}


def run_checker(
    task: Dict[str, Any],
    root: Path,
    calls: Sequence[Dict[str, Any]],
    final_answer: str,
) -> Tuple[bool, str]:
    """Dispatch to the task's checker. Raises CheckerError if unknown/malformed."""
    spec = task.get("checker") or {}
    kind = spec.get("type")
    fn = CHECKERS.get(kind)
    if fn is None:
        raise CheckerError(f"task {task.get('id')}: unknown checker type {kind!r}")
    state = {"root": root, "calls": list(calls), "final_answer": final_answer}
    try:
        return fn(task, state)
    except (KeyError, TypeError) as exc:
        raise CheckerError(
            f"task {task.get('id')}: malformed checker {kind!r}: {exc}"
        ) from exc

