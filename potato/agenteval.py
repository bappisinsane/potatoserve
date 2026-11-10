"""Phase 3 agent-eval harness (project-spec section 6).

Runs the 50-task suite in ``evals/agent_tasks/tasks/`` against a live
llama-server (OpenAI-compatible ``/v1/chat/completions``) and checks each task
programmatically. Design commitments:

* **Client-side tools in a sandbox.** The model emits ``tool_calls``; THIS
  harness executes them inside a per-task temp directory (path-traversal
  guarded) — never the server-side ``--tools`` runner, which would act on the
  real filesystem (docs/rules.md honesty + safety).
* **Reproducible by default:** temperature 0, fixed seed, fixed task order,
  deterministic error injection. ``--repeats``/``--temperature`` give the
  "small sampled run for robustness" the spec asks for.
* **Honest metrics:** success rate with a Wilson 95% interval, invalid
  tool-call %, mean steps, wall time, token usage — all from observed runs;
  missing measurements stay null.
* **Stdlib only** (urllib for HTTP): no dependency may be added without owner
  approval (docs/rules.md rule 2).

Checkers live next to the task data in ``evals/agent_tasks/checkers.py`` and
are loaded from there by path (``evals/`` is not an importable package).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from potato import runner as runner_mod
from potato import store

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_DIR = REPO_ROOT / "evals" / "agent_tasks" / "tasks"
_CHECKERS_PATH = REPO_ROOT / "evals" / "agent_tasks" / "checkers.py"

CATEGORIES = ("error_recovery", "file_edit", "search", "shell", "tool_call")

DEFAULT_MAX_STEPS = 14          # upper bound; each task may set its own lower
DEFAULT_CHAT_TIMEOUT_S = 120.0  # per HTTP completion
DEFAULT_TOOL_TIMEOUT_S = 30.0   # per run_shell invocation
DEFAULT_MAX_TOKENS = 1024       # per completion; caps runaway generations
TOOL_OUTPUT_LIMIT = 8000        # chars of tool output echoed back to the model

SYSTEM_PROMPT = (
    "You are an autonomous agent working in an isolated sandbox directory. "
    "Accomplish the user's task using the provided tools. All file paths are "
    "relative to the sandbox. run_shell executes with the operating system "
    "shell inside the sandbox; prefer portable commands (for example "
    "`python -c ...`) when the task involves file manipulation. "
    "If a tool returns an error, recover and try another approach. "
    "When the task is complete, respond with a brief final answer and no tool "
    "call."
)

# Built-in tools every task gets (JSON-schema arguments, spec section 6).
BUILTIN_TOOLS: List[Dict[str, Any]] = [
    {"name": "read_file", "description": "Read a text file from the sandbox.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}},
    {"name": "write_file", "description": "Create or overwrite a text file in the sandbox (parent dirs are created).",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "list_files", "description": "List all files in the sandbox.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "search", "description": "Case-insensitive substring search across sandbox files.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "run_shell", "description": "Run a shell command inside the sandbox. Returns exit code and output.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
]

BUILTIN_TOOL_NAMES = frozenset(t["name"] for t in BUILTIN_TOOLS)


def _load_checkers_module():
    """Import evals/agent_tasks/checkers.py by path (evals is not a package)."""
    spec = importlib.util.spec_from_file_location("potato_agent_checkers", _CHECKERS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load checkers from {_CHECKERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_checkers = _load_checkers_module()


def _validate_checker(spec: Dict[str, Any], task_id: str) -> None:
    """Raise ValueError on unknown/malformed checker (recursive into 'all')."""
    kind = spec.get("type")
    if kind == "all":
        subs = spec.get("of")
        if not subs or not isinstance(subs, list):
            raise ValueError(f"task {task_id}: 'all' checker needs a non-empty 'of' list")
        for sub in subs:
            _validate_checker(sub, task_id)
        return
    if kind not in _checkers.CHECKERS:
        raise ValueError(f"task {task_id}: unknown checker type {kind!r}")


def load_tasks(tasks_dir: Path) -> List[Dict[str, Any]]:
    """Load and validate every task; deterministic order (sorted file names).

    Raises ValueError with a precise message on any schema problem — a broken
    suite must fail loudly before any model is queried (rules.md).
    """
    tasks_dir = Path(tasks_dir)
    files = sorted(tasks_dir.glob("*.json"))
    if not files:
        raise ValueError(
            f"no task files in {tasks_dir} (expected *.json, one per category)"
        )
    tasks: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{path.name}: invalid JSON: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError(f"{path.name}: expected a JSON array of tasks")
        for idx, task in enumerate(raw):
            tid = task.get("id") or f"{path.stem}[{idx}]"
            if tid in seen:
                raise ValueError(f"duplicate task id {tid!r} ({seen[tid]} and {path.name})")
            seen[tid] = path.name
            for key in ("prompt", "setup", "max_steps", "checker"):
                if key not in task:
                    raise ValueError(f"task {tid}: missing required field {key!r}")
            if task["category"] != path.stem:
                raise ValueError(
                    f"task {tid}: category {task['category']!r} does not match file {path.stem!r}"
                )
            if not isinstance(task["prompt"], str) or not task["prompt"].strip():
                raise ValueError(f"task {tid}: prompt must be a non-empty string")
            if not isinstance(task["setup"], dict) or not all(
                isinstance(v, str) for v in task["setup"].values()
            ):
                raise ValueError(f"task {tid}: setup must map relative paths to strings")
            for rel in task["setup"]:
                if rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/"):
                    raise ValueError(f"task {tid}: setup path {rel!r} escapes the sandbox")
            if not (isinstance(task["max_steps"], int) and task["max_steps"] >= 1):
                raise ValueError(f"task {tid}: max_steps must be a positive integer")
            task.setdefault("tools", [])
            task.setdefault("inject_errors", [])
            for td in task["tools"]:
                if not isinstance(td.get("name"), str) or not isinstance(td.get("parameters"), dict):
                    raise ValueError(f"task {tid}: each tool needs name + parameters")
            for ie in task["inject_errors"]:
                if set(ie) != {"tool", "fail_times", "message"} or ie["fail_times"] < 1:
                    raise ValueError(
                        f"task {tid}: inject_errors entries need tool/fail_times/message"
                    )
                if ie["tool"] not in BUILTIN_TOOL_NAMES:
                    raise ValueError(
                        f"task {tid}: inject_errors targets unknown tool {ie['tool']!r}"
                    )
            _validate_checker(task["checker"], tid)
            if task["category"] == "tool_call":
                # A tool_call task must actually check a tool call, and that
                # tool must be one the task offers - otherwise the task is
                # unsatisfiable (it would fail for every model).
                c = task["checker"]
                if c.get("type") != "tool_call_args":
                    raise ValueError(
                        f"task {tid}: tool_call tasks need a 'tool_call_args' checker, "
                        f"got {c.get('type')!r}"
                    )
                names = {td["name"] for td in task["tools"]}
                if c.get("tool") not in names:
                    raise ValueError(
                        f"task {tid}: checker watches tool {c.get('tool')!r} "
                        f"which the task does not offer ({sorted(names)})"
                    )
            tasks.append(task)
    if not tasks:
        raise ValueError(f"no tasks found in {tasks_dir}")
    return tasks


def suite_fingerprint(tasks: Sequence[Dict[str, Any]]) -> str:
    """Stable sha256 over the loaded suite (provenance, ADR-2 spirit)."""
    canonical = json.dumps(tasks, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Sandbox: one temp directory per task, setup files applied, cleanup after.
# --------------------------------------------------------------------------

class Sandbox:
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def create(cls, task: Dict[str, Any]) -> "Sandbox":
        root = Path(tempfile.mkdtemp(prefix=f"potato-agent-{task['id']}-"))
        for rel, content in task.get("setup", {}).items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return cls(root)

    def resolve(self, rel: str) -> Path:
        """Path inside the sandbox; raises ValueError on traversal/absolute."""
        candidate = Path(rel)
        if candidate.is_absolute() or candidate.drive or candidate.anchor:
            raise ValueError(f"absolute paths are not allowed: {rel!r}")
        target = (self.root / rel).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError(f"path escapes the sandbox: {rel!r}")
        return target

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _clip(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} chars truncated]"


def _execute_builtin(
    name: str, args: Dict[str, Any], sandbox: Sandbox, tool_timeout_s: float
) -> Dict[str, Any]:
    if name == "read_file":
        target = sandbox.resolve(args["path"])
        if not target.is_file():
            return {"ok": False, "error": f"file not found: {args['path']}"}
        return {"ok": True, "content": _clip(target.read_text(encoding="utf-8", errors="replace"))}
    if name == "write_file":
        target = sandbox.resolve(args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        return {"ok": True, "bytes": len(args["content"].encode("utf-8"))}
    if name == "list_files":
        files = sorted(
            str(p.relative_to(sandbox.root)).replace("\\", "/")
            for p in sandbox.root.rglob("*") if p.is_file()
        )
        return {"ok": True, "files": files[:500]}
    if name == "search":
        query = str(args["query"]).lower()
        hits: List[str] = []
        for p in sorted(sandbox.root.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(sandbox.root)).replace("\\", "/")
            try:
                for lineno, line in enumerate(
                    p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if query in line.lower():
                        hits.append(f"{rel}:{lineno}:{_clip(line, 200)}")
                        if len(hits) >= 50:
                            break
            except OSError:
                continue
            if len(hits) >= 50:
                break
        return {"ok": True, "matches": hits}
    if name == "run_shell":
        try:
            proc = subprocess.run(
                str(args["command"]), shell=True, cwd=str(sandbox.root),
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=tool_timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"command timed out after {tool_timeout_s:g}s"}
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": _clip(proc.stdout or ""),
            "stderr": _clip(proc.stderr or ""),
        }
    return {"ok": False, "error": f"unknown built-in tool {name!r}"}


def execute_tool(
    name: str,
    args: Dict[str, Any],
    sandbox: Sandbox,
    injections: Dict[str, Dict[str, Any]],
    task_tools: Sequence[Dict[str, Any]],
    tool_timeout_s: float,
) -> Tuple[Dict[str, Any], bool]:
    """Execute one tool call. Returns (result, was_injected_failure).

    *injections* maps tool name -> {"fail_times": N, "message": str, "used": n};
    the first N calls to a tool fail deterministically without side effects —
    the "error recovery" mechanic of spec section 6.
    Custom (task-declared) tools are recorded but have no side effects; their
    arguments are checked by the task's ``tool_call_args`` checker.
    """
    inj = injections.get(name)
    if inj is not None and inj["used"] < inj["fail_times"]:
        inj["used"] += 1
        return {"ok": False, "error": inj["message"]}, True
    if name in BUILTIN_TOOL_NAMES:
        try:
            return _execute_builtin(name, args, sandbox, tool_timeout_s), False
        except ValueError as exc:  # path guard
            return {"ok": False, "error": str(exc)}, False
        except OSError as exc:
            return {"ok": False, "error": f"os error: {exc}"}, False
    for td in task_tools:
        if td.get("name") == name:
            return {"ok": True, "message": f"recorded call to {name}"}, False
    return {"ok": False, "error": f"unknown tool {name!r}"}, False


# --------------------------------------------------------------------------
# OpenAI-compatible client (stdlib urllib).
# --------------------------------------------------------------------------

class EndpointError(RuntimeError):
    """The server could not be reached or answered incorrectly."""


def _chat_url(endpoint: str) -> str:
    url = endpoint.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _models_url(endpoint: str) -> str:
    url = endpoint.rstrip("/")
    if url.endswith("/v1"):
        return url + "/models"
    return url + "/v1/models"


def _http_json(url: str, payload: Optional[Dict[str, Any]], timeout_s: float) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise EndpointError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise EndpointError(
            f"cannot reach {url} ({exc.reason}). Is llama-server running with "
            "--jinja (tool-calling template)? e.g. "
            "llama-server -m model.gguf --port 8080 --jinja"
        ) from exc
    except TimeoutError as exc:
        raise EndpointError(f"timeout after {timeout_s:g}s waiting for {url}") from exc
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
        # http.client.RemoteDisconnected is a ConnectionResetError (NOT a
        # URLError), so the handler above misses it. A server that drops the
        # connection mid-request (e.g. the model filled the context window
        # and llama-server reset the slot) must fail ONE task softly, not
        # sink the whole suite run.
        raise EndpointError(
            f"connection to {url} was reset by the server ({exc.__class__.__name__}). "
            "The server may have closed the slot (e.g. context window filled) or "
            "crashed; check the llama-server log."
        ) from exc
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise EndpointError(f"{url} returned non-JSON body: {body[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise EndpointError(f"{url} returned unexpected JSON: {body[:200]!r}")
    return parsed


def list_models(endpoint: str, timeout_s: float = 15.0) -> str:
    """The model id the server reports (used as the OpenAI `model` field)."""
    info = _http_json(_models_url(endpoint), None, timeout_s)
    data = info.get("data") or []
    if not data or not isinstance(data[0], dict) or not data[0].get("id"):
        raise EndpointError(f"{_models_url(endpoint)} listed no models")
    return str(data[0]["id"])


def chat_completion(
    endpoint: str, model: str, messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]], temperature: float, seed: int,
    timeout_s: float, max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": [{"type": "function", "function": t} for t in tools],
        "temperature": temperature,
        "seed": seed,
        "stream": False,
        # Cap the completion length. Without this the model may generate until
        # it fills the server's whole context window, which makes llama-server
        # reset the slot mid-request (observed: n_tokens=4095, truncated=1)
        # and kills the connection. A tool call is short; a runaway generation
        # is a model failure, not a reason to lose the run.
        "max_tokens": max_tokens,
    }
    resp = _http_json(_chat_url(endpoint), payload, timeout_s)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise EndpointError(f"response has no choices: {json.dumps(resp)[:300]}")
    return resp


@dataclass
class TaskResult:
    task_id: str
    category: str
    passed: bool
    detail: str
    steps: int
    wall_s: float
    prompt_tokens: int
    completion_tokens: int
    invalid_tool_calls: int
    injected_failures_used: int
    final_answer: str
    harness_error: Optional[str] = None
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "category": self.category,
            "passed": self.passed, "detail": self.detail, "steps": self.steps,
            "wall_s": round(self.wall_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "invalid_tool_calls": self.invalid_tool_calls,
            "injected_failures_used": self.injected_failures_used,
            "harness_error": self.harness_error,
        }


def _tool_schema(tools: Sequence[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for td in tools:
        if td.get("name") == name:
            return td.get("parameters")
    return None


def run_task(
    task: Dict[str, Any],
    endpoint: str,
    model: str,
    temperature: float = 0.0,
    seed: int = 42,
    chat_timeout_s: float = DEFAULT_CHAT_TIMEOUT_S,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    max_steps_override: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> TaskResult:
    """Run one task to completion in a fresh sandbox, then evaluate it.

    Loop: chat -> if tool_calls: validate args (schema subset), apply error
    injection, execute in sandbox, append tool results; else the content is
    the final answer. Fails soft per task (harness_error) so one broken task
    cannot sink a whole suite run.
    """
    sandbox = Sandbox.create(task)
    tools = list(BUILTIN_TOOLS) + list(task.get("tools", []))
    injections: Dict[str, Dict[str, Any]] = {
        ie["tool"]: {"fail_times": ie["fail_times"], "message": ie["message"], "used": 0}
        for ie in task.get("inject_errors", [])
    }
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    calls: List[Dict[str, Any]] = []
    invalid = 0
    prompt_tokens = completion_tokens = 0
    injected_used = 0
    final_answer = ""
    steps = 0
    harness_error: Optional[str] = None
    limit = max_steps_override if max_steps_override else task.get("max_steps", DEFAULT_MAX_STEPS)
    started = time.monotonic()
    try:
        for step in range(1, limit + 1):
            steps = step
            resp = chat_completion(
                endpoint, model, messages, tools, temperature, seed, chat_timeout_s,
                max_tokens=max_tokens,
            )
            usage = resp.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            msg = resp["choices"][0].get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            assistant: Dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)
            if not tool_calls:
                final_answer = msg.get("content") or ""
                break
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments", "{}")
                args: Optional[Dict[str, Any]] = None
                was_injected = False
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args) if raw_args.strip() else {}
                    except ValueError:
                        parsed = None
                else:
                    parsed = raw_args
                if isinstance(parsed, dict):
                    args = parsed
                    schema = _tool_schema(tools, name)
                    schema_errors = _checkers.validate_schema(args, schema) if schema else []
                    if schema_errors:
                        invalid += 1
                        result: Dict[str, Any] = {
                            "ok": False,
                            "error": "schema violations: " + "; ".join(schema_errors[:5]),
                        }
                    else:
                        result, was_injected = execute_tool(
                            name, args, sandbox, injections, task.get("tools", []),
                            tool_timeout_s,
                        )
                else:
                    invalid += 1
                    result = {"ok": False, "error": "arguments were not valid JSON"}
                if was_injected:
                    injected_used += 1
                calls.append({
                    "name": name,
                    "args": args if args is not None else str(raw_args)[:500],
                    "injected_failure": was_injected,
                    "ok": bool(result.get("ok")),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            harness_error = f"no final answer within {limit} steps"
    except EndpointError as exc:
        harness_error = str(exc)
    wall = time.monotonic() - started

    passed = False
    if harness_error is None:
        try:
            passed, detail = _checkers.run_checker(task, sandbox.root, calls, final_answer)
        except _checkers.CheckerError as exc:
            harness_error = str(exc)
            detail = f"checker error: {exc}"
    else:
        detail = harness_error
    sandbox.cleanup()
    return TaskResult(
        task_id=task["id"], category=task["category"], passed=passed,
        detail=detail, steps=steps, wall_s=wall,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        invalid_tool_calls=invalid, injected_failures_used=injected_used,
        final_answer=_clip(final_answer, 1000), harness_error=harness_error,
        calls=calls,
    )


# --------------------------------------------------------------------------
# Metrics: Wilson 95% interval + aggregation over repeats.
# --------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Chosen because it behaves correctly at the extremes (0/n and n/n), where
    the textbook normal approximation gives intervals leaving [0, 1].
    Returns (lower, upper); (0.0, 0.0) for n == 0 ("not measured" downstream).
    """
    if n <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes {successes} out of range for n={n}")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate_results(
    all_runs: Sequence[Sequence[TaskResult]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fold per-repeat task results into (metrics, spread) for a §8 record.

    * ``metrics.agent_success`` = median of per-repeat success rates (each
      repeat is the full suite, so with ``--repeats 1`` it is just the rate);
    * ``spread.agent_samples`` = every per-repeat rate (report shows median +
      spread like every other metric);
    * ``spread.agent_detail`` = spec-section-6 metrics: Wilson 95% CI over all
      task trials, invalid tool-call %, mean steps, mean wall time, tokens.
    """
    if not all_runs:
        return (
            {"agent_success": None},
            {"agent_samples": [], "agent_detail": {"tasks_total": 0}},
        )
    rates: List[float] = []
    trials = passed_trials = 0
    invalid_calls = steps_total = wall_total = 0
    prompt_tokens = completion_tokens = 0
    harness_errors = 0
    detail_rows: List[Dict[str, Any]] = []
    for run in all_runs:
        n = len(run)
        ok = sum(1 for r in run if r.passed)
        rates.append(ok / n if n else 0.0)
        trials += n
        passed_trials += ok
        for r in run:
            invalid_calls += r.invalid_tool_calls
            steps_total += r.steps
            wall_total += r.wall_s
            prompt_tokens += r.prompt_tokens
            completion_tokens += r.completion_tokens
            if r.harness_error:
                harness_errors += 1
            detail_rows.append(r.to_dict())
    lo, hi = wilson_interval(passed_trials, trials)
    # Invalid tool-call % over every tool call the model attempted: we only
    # counted invalid ones, so express it per task trial (honest denominator).
    detail = {
        "repeats": len(all_runs),
        "tasks_total": trials,
        "tasks_passed": passed_trials,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "invalid_tool_call_pct": round(100.0 * invalid_calls / trials, 2) if trials else None,
        "invalid_tool_calls_total": invalid_calls,
        "mean_steps": round(steps_total / trials, 2) if trials else None,
        "mean_wall_s": round(wall_total / trials, 3) if trials else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "harness_errors": harness_errors,
        "per_task": detail_rows,
    }
    import statistics

    metrics = {"agent_success": round(statistics.median(rates), 4)}
    spread = {"agent_samples": [round(r, 4) for r in rates], "agent_detail": detail}
    return metrics, spread


# --------------------------------------------------------------------------
# Suite runner + §8 record builder.
# --------------------------------------------------------------------------

def _model_sha256(model_path: str, chunk: int = 1 << 20) -> Optional[str]:
    """SHA-256 of a local file, or None if it cannot be read (never guessed)."""
    p = Path(model_path)
    if not p.is_absolute():
        # llama-server may report a path relative to ITS cwd, not ours.
        candidates = [p, REPO_ROOT / p, Path.cwd() / p]
    else:
        candidates = [p]
    for cand in candidates:
        if cand.is_file():
            h = hashlib.sha256()
            try:
                with open(cand, "rb") as fh:
                    while True:
                        block = fh.read(chunk)
                        if not block:
                            break
                        h.update(block)
                return h.hexdigest()
            except OSError:
                return None
    return None


def run_suite(
    tasks: Sequence[Dict[str, Any]],
    endpoint: str,
    repeats: int = 1,
    temperature: float = 0.0,
    seed: int = 42,
    chat_timeout_s: float = DEFAULT_CHAT_TIMEOUT_S,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    max_steps_override: Optional[int] = None,
    progress: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Tuple[List[List[TaskResult]], str]:
    """Run the whole suite *repeats* times; returns (runs, model_id).

    Model id comes from ``GET /v1/models`` once — hard fail with guidance if
    the endpoint is dead (never fabricate a run that did not happen).
    """
    model = list_models(endpoint, timeout_s=min(chat_timeout_s, 30.0))
    all_runs: List[List[TaskResult]] = []
    total = len(tasks)
    for rep in range(1, repeats + 1):
        if progress:
            print(f"-- repeat {rep}/{repeats} against {model} @ {endpoint}")
        run: List[TaskResult] = []
        for i, task in enumerate(tasks, 1):
            result = run_task(
                task, endpoint, model, temperature=temperature, seed=seed,
                chat_timeout_s=chat_timeout_s, tool_timeout_s=tool_timeout_s,
                max_steps_override=max_steps_override, max_tokens=max_tokens,
            )
            run.append(result)
            if progress:
                status = "PASS" if result.passed else (
                    "ERROR" if result.harness_error else "FAIL"
                )
                print(
                    f"   [{i:>2}/{total}] {task['id']:<6} {status:<5} "
                    f"steps={result.steps} {result.wall_s:6.1f}s  {result.detail[:80]}"
                )
        all_runs.append(run)
    return all_runs, model


def build_record(
    run_id: str,
    all_runs: Sequence[Sequence[TaskResult]],
    model_id: str,
    endpoint: str,
    tasks_dir: str,
    suite_sha: str,
    repeats: int,
    temperature: float,
    seed: int,
    category_filter: Optional[str],
) -> Dict[str, Any]:
    """Assemble a project-spec §8 record with agent_success + §6 metrics."""
    from potato import probe as probe_mod
    from potato.schema import new_record, validate

    metrics, spread = aggregate_results(all_runs)
    fingerprint: Optional[Dict[str, Any]] = None
    try:
        fingerprint = probe_mod.probe()
    except Exception:  # noqa: BLE001 - fingerprint is enrichment, never fatal
        fingerprint = None
    driver = os_name = gpu_name = None
    if fingerprint:
        nvidia = (fingerprint.get("gpu") or {}).get("nvidia") or {}
        driver = nvidia.get("driver")
        gpu_name = nvidia.get("name")
        os_name = (fingerprint.get("os") or {}).get("platform")
    record = new_record(
        run_id,
        llama_cpp_commit=runner_mod.pinned_commit(),
        build_flags={"tool": "llama-server", "client": "potato agent-eval"},
        driver=driver, os=os_name, gpu=gpu_name,
        model_id=model_id, model_sha256=_model_sha256(model_id),
        config={
            "endpoint": endpoint,
            "tasks_dir": tasks_dir,
            "suite_sha256": suite_sha,
            "repeats": repeats,
            "temperature": temperature,
            "seed": seed,
            "category_filter": category_filter,
            "model_size_bytes": (
                Path(model_id).stat().st_size if Path(model_id).is_file() else None
            ),
        },
        metrics=metrics,
        repeats=repeats,
        spread=spread,
    )
    problems = validate(record)
    if problems:
        raise ValueError(f"agent-eval record invalid: {problems}")
    return record.to_dict()


def render_summary(record: Dict[str, Any]) -> str:
    """Human-readable section-6 summary of a stored/derived record."""
    detail = (record.get("spread") or {}).get("agent_detail") or {}
    metrics = record.get("metrics") or {}
    samples = (record.get("spread") or {}).get("agent_samples") or []
    lo, hi = detail.get("wilson95", [None, None])
    lines = ["PotatoServe agent-eval summary", "=" * 56]
    succ = metrics.get("agent_success")
    lines.append(f"run_id          : {record.get('run_id', '?')}")
    lines.append(f"model           : {record.get('model_id', '?')}")
    if detail:
        rate = f" = {succ:.1%}" if isinstance(succ, float) else ""
        lines.append(
            f"success         : {detail.get('tasks_passed')}/"
            f"{detail.get('tasks_total')}{rate}"
        )
        lines.append(
            f"Wilson 95% CI   : [{lo}, {hi}]" if lo is not None
            else "Wilson 95% CI   : not measured"
        )
        lines.append(f"invalid tool %  : {detail.get('invalid_tool_call_pct')}")
        lines.append(f"mean steps      : {detail.get('mean_steps')}")
        lines.append(f"mean wall time  : {detail.get('mean_wall_s')} s")
        lines.append(
            f"tokens          : {detail.get('prompt_tokens')} prompt + "
            f"{detail.get('completion_tokens')} completion"
        )
        lines.append(f"harness errors  : {detail.get('harness_errors')}")
    if len(samples) > 1:
        lines.append(f"repeat rates    : {samples}")
    failures = [r for r in detail.get("per_task", []) if not r.get("passed")]
    if failures:
        lines.append("")
        lines.append("failed tasks:")
        for r in failures:
            lines.append(
                f"  {r['task_id']:<6} {r['category']:<14} {r['detail'][:70]}"
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="potato agent-eval",
        description="Run the 50-task agent suite against a llama-server endpoint "
                    "(project-spec section 6) and store a section-8 record.",
    )
    parser.add_argument("--endpoint", default=None,
                        help="Base URL of llama-server, e.g. http://127.0.0.1:8080")
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR),
                        help="Directory of per-category task JSON files.")
    parser.add_argument("--results-dir", default=store.DEFAULT_RESULTS_DIR,
                        help="Where to store the resulting record (append-only).")
    parser.add_argument("--run-id", default=None,
                        help="Override the generated run id.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Full suite repeats (default 1; use 3+ for spread).")
    parser.add_argument("--category", default=None, choices=list(CATEGORIES),
                        help="Run only one category (debug / partial runs).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N tasks after filtering (debug).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0 = reproducible).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed sent with each request (default 42).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override every task's step budget.")
    parser.add_argument("--chat-timeout", type=float, default=DEFAULT_CHAT_TIMEOUT_S,
                        help="Per-request timeout in seconds.")
    parser.add_argument("--tool-timeout", type=float, default=DEFAULT_TOOL_TIMEOUT_S,
                        help="Per run_shell invocation timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Max tokens per completion; caps runaway generations "
                             "that would otherwise fill the server's context window.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the suite and print the plan; no HTTP, no writes.")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    try:
        tasks = load_tasks(Path(args.tasks_dir))
    except ValueError as exc:
        print(f"potato agent-eval: {exc}", file=sys.stderr)
        return 1
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.limit is not None:
        tasks = tasks[: max(args.limit, 0)]
    suite_sha = suite_fingerprint(tasks)

    if args.dry_run:
        by_cat: Dict[str, int] = {}
        for t in tasks:
            by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        print(f"dry run: {len(tasks)} task(s) loaded from {args.tasks_dir}")
        print(f"  by category : {by_cat}")
        print(f"  suite sha256: {suite_sha}")
        print(f"  repeats={args.repeats} temperature={args.temperature} seed={args.seed}")
        print(f"  endpoint    : {args.endpoint or '(not given; --endpoint required to run)'}")
        if args.endpoint:
            print("endpoint given; would now start the suite (dry run stops here).")
        return 0

    if not args.endpoint:
        parser.error("--endpoint is required (or pass --dry-run)")
    if args.repeats < 3:
        print(
            "note: spec asks for >= 3 repeats per point for trustworthy spread.",
            file=sys.stderr,
        )
    try:
        all_runs, model = run_suite(
            tasks, args.endpoint, repeats=args.repeats,
            temperature=args.temperature, seed=args.seed,
            chat_timeout_s=args.chat_timeout, tool_timeout_s=args.tool_timeout,
            max_steps_override=args.max_steps, max_tokens=args.max_tokens,
        )
        run_id = args.run_id or time.strftime("agent-%Y%m%dT%H%M%SZ", time.gmtime())
        record = build_record(
            run_id, all_runs, model, args.endpoint, str(args.tasks_dir),
            suite_sha, args.repeats, args.temperature, args.seed, args.category,
        )
        path = store.save_record(record, results_dir=args.results_dir)
    except (EndpointError, ValueError, OSError, RuntimeError) as exc:
        print(f"potato agent-eval: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    print()
    print(render_summary(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









