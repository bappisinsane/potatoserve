"""Unit tests for the Phase 3 agent-eval harness (potato/agenteval.py).

No GPU and no llama-server needed: the agent loop is exercised against a fake
OpenAI-compatible server (stdlib http.server on a random port) with scripted
responses, so tool execution, error injection, invalid-argument accounting and
the checker integration are all covered end-to-end without a model.
"""

import json
import tempfile
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from potato import agenteval
from potato.agenteval import (
    Sandbox,
    aggregate_results,
    load_tasks,
    run_task,
    suite_fingerprint,
    wilson_interval,
)


class FakeOpenAIServer:
    """Scripted OpenAI-compatible endpoint.

    * ``/v1/models`` always answers one model id;
    * ``/v1/chat/completions`` pops the next scripted body (a full response
      dict). An empty queue answers HTTP 500 so unexpected calls fail loudly.
    """

    def __init__(self, model_id: str = "fake-model.gguf"):
        self.model_id = model_id
        self.queue = []
        self.chat_requests = []
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "FakeOpenAIServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _make_handler(self):
        ref = self  # closure

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence request logging
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"data": [{"id": ref.model_id}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    ref.chat_requests.append(json.loads(raw.decode("utf-8")))
                except ValueError:
                    ref.chat_requests.append({"unparseable": raw[:200]})
                if not ref.queue:
                    self._send(500, {"error": "script queue exhausted"})
                    return
                self._send(200, ref.queue.pop(0))

        return Handler


def chat_resp(message, prompt_tokens=10, completion_tokens=5):
    """A minimal but schema-shaped chat completion body."""
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def tool_call_resp(name, arguments, tc_id="call-1"):
    return chat_resp({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id, "type": "function",
            "function": {"name": name,
                         "arguments": arguments if isinstance(arguments, str)
                         else json.dumps(arguments)},
        }],
    })


def final_resp(text):
    return chat_resp({"role": "assistant", "content": text})


def simple_task(task_id="tt-01", category="file_edit", prompt="do it",
                setup=None, checker=None, inject_errors=None, max_steps=6,
                tools=None):
    return {
        "id": task_id, "category": category, "name": "t", "prompt": prompt,
        "setup": setup or {}, "tools": tools or [],
        "inject_errors": inject_errors or [], "max_steps": max_steps,
        "checker": checker or {"type": "files_exist", "paths": ["out.txt"]},
    }


class TaskSuiteIntegrityTest(unittest.TestCase):
    """The committed suite must always load: 50 tasks, 5x10, known checkers."""

    def test_full_suite_loads(self) -> None:
        tasks = load_tasks(agenteval.DEFAULT_TASKS_DIR)
        self.assertEqual(len(tasks), 50)
        by_cat = {}
        for t in tasks:
            by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        self.assertEqual(
            by_cat,
            {"error_recovery": 10, "file_edit": 10, "search": 10,
             "shell": 10, "tool_call": 10},
        )

    def test_ids_unique_and_fingerprint_stable(self) -> None:
        tasks = load_tasks(agenteval.DEFAULT_TASKS_DIR)
        ids = [t["id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        fp1 = suite_fingerprint(tasks)
        fp2 = suite_fingerprint(load_tasks(agenteval.DEFAULT_TASKS_DIR))
        self.assertEqual(fp1, fp2)
        self.assertRegex(fp1, r"^[0-9a-f]{16}$")

    def test_unknown_checker_rejected(self) -> None:
        bad = simple_task(checker={"type": "no_such_checker"})
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "file_edit.json").write_text(json.dumps([bad]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_tasks(Path(tmp))
            self.assertIn("unknown checker", str(ctx.exception))

    def test_duplicate_id_rejected(self) -> None:
        a = simple_task(task_id="dup-1")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "file_edit.json").write_text(json.dumps([a]), encoding="utf-8")
            Path(tmp, "shell.json").write_text(
                json.dumps([{**a, "category": "shell"}]), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                load_tasks(Path(tmp))
            self.assertIn("duplicate", str(ctx.exception))

    def test_category_must_match_file(self) -> None:
        bad = simple_task()  # category file_edit, saved as shell.json
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "shell.json").write_text(json.dumps([bad]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tasks(Path(tmp))

    def test_missing_directory_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_tasks(Path("no/such/tasks/dir"))

    def test_tool_call_task_must_check_an_offered_tool(self) -> None:
        # The checker watches a tool the task never offers -> unsatisfiable.
        bad = simple_task(
            task_id="tc-bad", category="tool_call",
            tools=[{"name": "get_weather", "parameters": {"type": "object"}}],
            checker={"type": "tool_call_args", "tool": "other_tool",
                     "expect": {"city": "Tokyo"}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "tool_call.json").write_text(json.dumps([bad]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_tasks(Path(tmp))
            self.assertIn("does not offer", str(ctx.exception))

    def test_tool_call_task_needs_a_tool_call_checker(self) -> None:
        bad = simple_task(
            task_id="tc-bad2", category="tool_call",
            tools=[{"name": "get_weather", "parameters": {"type": "object"}}],
            checker={"type": "files_exist", "paths": ["out.txt"]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "tool_call.json").write_text(json.dumps([bad]), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_tasks(Path(tmp))
            self.assertIn("tool_call_args", str(ctx.exception))


class WilsonIntervalTest(unittest.TestCase):
    """Reference values computed from the Wilson formula (z=1.96)."""

    def test_empty_is_not_measured(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_half_successes(self) -> None:
        lo, hi = wilson_interval(25, 50)
        self.assertAlmostEqual(lo, 0.3664, places=3)
        self.assertAlmostEqual(hi, 0.6336, places=3)

    def test_all_successes_stays_in_unit_interval(self) -> None:
        lo, hi = wilson_interval(50, 50)
        self.assertAlmostEqual(lo, 0.9287, places=3)
        self.assertEqual(hi, 1.0)

    def test_no_successes(self) -> None:
        lo, hi = wilson_interval(0, 50)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0713, places=3)

    def test_invalid_counts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(51, 50)


class SandboxAndToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = Sandbox.create(simple_task(setup={"a.txt": "hello\n"}))
        self.addCleanup(self.sandbox.cleanup)

    def test_setup_files_written(self) -> None:
        self.assertEqual((self.sandbox.root / "a.txt").read_text(), "hello\n")

    def test_path_traversal_rejected(self) -> None:
        for bad in ("../escape.txt", "/etc/passwd", "a/../../x.txt"):
            with self.assertRaises(ValueError):
                self.sandbox.resolve(bad)

    def test_write_read_list_roundtrip(self) -> None:
        from potato.agenteval import execute_tool

        result, injected = execute_tool(
            "write_file", {"path": "sub/out.txt", "content": "data"},
            self.sandbox, {}, [], 30.0,
        )
        self.assertTrue(result["ok"]); self.assertFalse(injected)
        result, _ = execute_tool(
            "read_file", {"path": "sub/out.txt"}, self.sandbox, {}, [], 30.0
        )
        self.assertEqual(result["content"], "data")
        result, _ = execute_tool("list_files", {}, self.sandbox, {}, [], 30.0)
        self.assertIn("sub/out.txt", result["files"])

    def test_write_rejects_escape(self) -> None:
        from potato.agenteval import execute_tool

        result, _ = execute_tool(
            "write_file", {"path": "../evil.txt", "content": "x"},
            self.sandbox, {}, [], 30.0,
        )
        self.assertFalse(result["ok"])
        self.assertIn("escapes", result["error"])
        self.assertFalse((self.sandbox.root.parent / "evil.txt").exists())

    def test_search_finds_case_insensitive(self) -> None:
        from potato.agenteval import execute_tool

        result, _ = execute_tool(
            "search", {"query": "HELLO"}, self.sandbox, {}, [], 30.0
        )
        self.assertTrue(result["ok"])
        self.assertTrue(any("a.txt" in m for m in result["matches"]))

    def test_shell_runs_in_sandbox(self) -> None:
        from potato.agenteval import execute_tool

        # Portable across Windows cmd and POSIX sh: python is always on PATH.
        result, _ = execute_tool(
            "run_shell",
            {"command": 'python -c "open(\'shell.txt\',\'w\').write(\'ran\')"'},
            self.sandbox, {}, [], 30.0,
        )
        self.assertEqual(result["exit_code"], 0, result)
        self.assertEqual((self.sandbox.root / "shell.txt").read_text(), "ran")

    def test_injection_fails_then_recovers(self) -> None:
        from potato.agenteval import execute_tool

        injections = {"write_file": {
            "fail_times": 1, "message": "quota exceeded", "used": 0,
        }}
        args = {"path": "x.txt", "content": "1"}
        result, injected = execute_tool(
            "write_file", args, self.sandbox, injections, [], 30.0
        )
        self.assertTrue(injected)
        self.assertFalse(result["ok"])
        self.assertIn("quota", result["error"])
        self.assertFalse((self.sandbox.root / "x.txt").exists())
        # Second attempt succeeds and has no side effect from the first.
        result, injected = execute_tool(
            "write_file", args, self.sandbox, injections, [], 30.0
        )
        self.assertFalse(injected)
        self.assertTrue(result["ok"])
        self.assertTrue((self.sandbox.root / "x.txt").exists())

    def test_custom_tool_records_without_side_effects(self) -> None:
        from potato.agenteval import execute_tool

        tools = [{"name": "get_weather", "parameters": {"type": "object"}}]
        result, injected = execute_tool(
            "get_weather", {"city": "Tokyo"}, self.sandbox, {}, tools, 30.0
        )
        self.assertTrue(result["ok"]); self.assertFalse(injected)
        result, _ = execute_tool(
            "no_such_tool", {}, self.sandbox, {}, tools, 30.0
        )
        self.assertFalse(result["ok"])


class AgentLoopFakeServerTest(unittest.TestCase):
    """End-to-end loop: scripted responses -> sandbox -> checker verdict."""

    def setUp(self) -> None:
        self.server = FakeOpenAIServer().start()
        self.addCleanup(self.server.stop)

    def _run(self, task, chat_timeout_s=5.0):
        return run_task(
            task, self.server.endpoint, "fake-model.gguf",
            chat_timeout_s=chat_timeout_s,
        )

    def test_write_then_finish_passes(self) -> None:
        self.server.queue = [
            tool_call_resp("write_file", {"path": "out.txt", "content": "all good"}),
            final_resp("finished"),
        ]
        task = simple_task(
            checker={"type": "all", "of": [
                {"type": "file_contains", "path": "out.txt", "all_of": ["all good"]},
                {"type": "answer_contains", "all_of": ["finished"]},
            ]}
        )
        result = self._run(task)
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(result.completion_tokens, 10)
        self.assertEqual(result.invalid_tool_calls, 0)
        self.assertIsNone(result.harness_error)
        self.assertEqual(len(result.calls), 1)
        self.assertTrue(result.calls[0]["ok"])
        # Request shape: OpenAI tools envelope, temperature 0, seed 42.
        first = self.server.chat_requests[0]
        self.assertEqual(first["temperature"], 0.0)
        self.assertEqual(first["seed"], 42)
        self.assertTrue(first["tools"])
        self.assertEqual(first["tools"][0]["type"], "function")
        self.assertEqual(first["messages"][0]["role"], "system")

    def test_invalid_json_arguments_counted_and_recovered(self) -> None:
        self.server.queue = [
            tool_call_resp("write_file", "{not valid json"),
            final_resp("done despite error"),
        ]
        task = simple_task(checker={"type": "answer_contains", "all_of": ["done"]})
        result = self._run(task)
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.invalid_tool_calls, 1)
        self.assertEqual(result.steps, 2)
        # The tool error was fed back to the model on request 2.
        tool_msgs = [
            m for m in self.server.chat_requests[1]["messages"]
            if m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("not valid JSON", tool_msgs[0]["content"])

    def test_schema_violation_counted_and_explained(self) -> None:
        self.server.queue = [
            tool_call_resp("write_file", {"path": 123, "content": "x"}),
            final_resp("could not write"),
        ]
        task = simple_task(checker={"type": "answer_contains", "all_of": ["could not"]})
        result = self._run(task)
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.invalid_tool_calls, 1)
        tool_msgs = [
            m for m in self.server.chat_requests[1]["messages"]
            if m.get("role") == "tool"
        ]
        self.assertIn("schema violations", tool_msgs[0]["content"])

    def test_injected_failure_recovers_on_retry(self) -> None:
        self.server.queue = [
            tool_call_resp("write_file", {"path": "ok.txt", "content": "persistence"}),
            tool_call_resp("write_file", {"path": "ok.txt", "content": "persistence"},
                           tc_id="call-2"),
            final_resp("written"),
        ]
        task = simple_task(
            inject_errors=[{"tool": "write_file", "fail_times": 1,
                            "message": "transient storage error"}],
            checker={"type": "file_contains", "path": "ok.txt",
                     "all_of": ["persistence"]},
            max_steps=8,
        )
        result = self._run(task)
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.injected_failures_used, 1)
        self.assertEqual(result.steps, 3)
        # The first tool result fed back to the model carries the error.
        first_tool_msg = [
            m for m in self.server.chat_requests[1]["messages"]
            if m.get("role") == "tool"
        ][0]
        self.assertIn("transient storage error", first_tool_msg["content"])

    def test_step_budget_exhaustion_fails_soft(self) -> None:
        self.server.queue = [
            tool_call_resp("list_files", {}, tc_id="c1"),
            tool_call_resp("list_files", {}, tc_id="c2"),
        ]
        task = simple_task(max_steps=2,
                           checker={"type": "answer_contains", "all_of": ["never"]})
        result = self._run(task)
        self.assertFalse(result.passed)
        self.assertIn("no final answer", result.harness_error or "")
        self.assertEqual(result.steps, 2)

    def test_custom_tool_call_checked(self) -> None:
        weather_schema = {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "unit": {"type": "string",
                                    "enum": ["celsius", "fahrenheit"]}},
            "required": ["city", "unit"],
        }
        self.server.queue = [
            tool_call_resp("get_weather", {"city": "Tokyo", "unit": "celsius"}),
            final_resp("It is mild in Tokyo."),
        ]
        task = simple_task(
            category="tool_call",
            tools=[{"name": "get_weather", "description": "d",
                    "parameters": weather_schema}],
            checker={"type": "tool_call_args", "tool": "get_weather",
                     "expect": {"city": "Tokyo", "unit": "celsius"}},
        )
        result = self._run(task)
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.invalid_tool_calls, 0)

    def test_custom_tool_schema_violation_then_retry(self) -> None:
        weather_schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        self.server.queue = [
            tool_call_resp("get_weather", {"city": 123}),
            tool_call_resp("get_weather", {"city": "Tokyo"}, tc_id="call-2"),
            final_resp("Mild."),
        ]
        task = simple_task(
            category="tool_call",
            tools=[{"name": "get_weather", "description": "d",
                    "parameters": weather_schema}],
            checker={"type": "tool_call_args", "tool": "get_weather",
                     "expect": {"city": "Tokyo"}},
        )
        result = self._run(task)
        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.invalid_tool_calls, 1)

    def test_unreachable_endpoint_fails_soft(self) -> None:
        result = run_task(simple_task(), "http://127.0.0.1:9", "m",
                          chat_timeout_s=2.0)
        self.assertFalse(result.passed)
        self.assertIn("cannot reach", result.harness_error or "")

    def test_connection_reset_fails_soft(self) -> None:
        """A server that drops the connection must fail ONE task, not the run.

        http.client.RemoteDisconnected is a ConnectionResetError, NOT a
        urllib.error.URLError, so a naive `except URLError` lets it escape
        and abort the whole suite (observed against a real llama-server
        that reset the slot when the context window filled).
        """
        import http.client

        from potato import agenteval as agent_mod

        with unittest.mock.patch.object(
            agent_mod.urllib.request, "urlopen",
            side_effect=http.client.RemoteDisconnected("connection reset"),
        ):
            result = run_task(simple_task(), "http://127.0.0.1:8080", "m",
                              chat_timeout_s=2.0)
        self.assertFalse(result.passed)
        self.assertIn("reset by the server", result.harness_error or "")
        # It must be an EndpointError so run_suite's per-task handling sees it.
        self.assertNotIn("Traceback", result.harness_error or "")

    def test_max_tokens_is_sent_on_every_request(self) -> None:
        """Every completion must carry a max_tokens cap.

        Without it the model can generate until it fills the server's whole
        context window, which makes llama-server reset the slot mid-request
        and kills the connection.
        """
        self.server.queue = [final_resp("ok")]
        self._run(simple_task(checker={"type": "answer_contains",
                                        "all_of": ["ok"]}))
        self.assertEqual(len(self.server.chat_requests), 1)
        self.assertIn("max_tokens", self.server.chat_requests[0])
        self.assertGreater(self.server.chat_requests[0]["max_tokens"], 0)

    def test_max_tokens_override_is_honoured(self) -> None:
        self.server.queue = [final_resp("ok")]
        run_task(simple_task(checker={"type": "answer_contains",
                                      "all_of": ["ok"]}),
                 self.server.endpoint, "fake-model.gguf",
                 chat_timeout_s=5.0, max_tokens=64)
        self.assertEqual(self.server.chat_requests[0]["max_tokens"], 64)


def _tr(tid, passed, steps=3, wall=1.0, invalid=0, pt=10, ct=5, err=None):
    from potato.agenteval import TaskResult

    return TaskResult(
        task_id=tid, category="file_edit", passed=passed,
        detail="ok" if passed else "nope", steps=steps, wall_s=wall,
        prompt_tokens=pt, completion_tokens=ct, invalid_tool_calls=invalid,
        injected_failures_used=0, final_answer="", harness_error=err,
    )


class AggregateResultsTest(unittest.TestCase):
    def test_median_rate_and_samples_across_repeats(self) -> None:
        run_a = [_tr("a", True), _tr("b", True), _tr("c", True), _tr("d", False)]
        run_b = [_tr("a", True), _tr("b", True), _tr("c", False), _tr("d", False)]
        metrics, spread = aggregate_results([run_a, run_b])
        # rates 0.75 and 0.5 -> median 0.625 (even count -> mean of the two middles)
        self.assertAlmostEqual(metrics["agent_success"], 0.625)
        self.assertEqual(spread["agent_samples"], [0.75, 0.5])
        detail = spread["agent_detail"]
        self.assertEqual(detail["tasks_total"], 8)
        self.assertEqual(detail["tasks_passed"], 5)
        self.assertEqual(len(detail["wilson95"]), 2)
        self.assertLess(detail["wilson95"][0], 5 / 8)
        self.assertGreater(detail["wilson95"][1], 5 / 8)
        self.assertEqual(detail["invalid_tool_calls_total"], 0)
        self.assertEqual(detail["mean_steps"], 3.0)
        self.assertEqual(detail["prompt_tokens"], 80)
        self.assertEqual(detail["completion_tokens"], 40)

    def test_failures_and_invalid_calls_counted(self) -> None:
        run = [_tr("a", True, invalid=2), _tr("b", False, invalid=1, err="boom")]
        metrics, spread = aggregate_results([run])
        detail = spread["agent_detail"]
        self.assertAlmostEqual(metrics["agent_success"], 0.5)
        self.assertEqual(detail["invalid_tool_calls_total"], 3)
        self.assertEqual(detail["harness_errors"], 1)
        self.assertEqual(len(detail["per_task"]), 2)

    def test_empty_is_null_not_zero(self) -> None:
        metrics, spread = aggregate_results([])
        self.assertIsNone(metrics["agent_success"])
        self.assertEqual(spread["agent_detail"]["tasks_total"], 0)


class CliTest(unittest.TestCase):
    def test_dry_run_full_suite(self) -> None:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = agenteval.main(["--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("50 task(s)", buf.getvalue())
        self.assertIn("suite sha256", buf.getvalue())

    def test_dry_run_category_filter(self) -> None:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = agenteval.main(["--dry-run", "--category", "shell", "--limit", "3"])
        self.assertEqual(code, 0)
        self.assertIn("3 task(s)", buf.getvalue())

    def test_missing_endpoint_is_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            agenteval.main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_dead_endpoint_returns_1_without_writes(self) -> None:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = agenteval.main([
                    "--endpoint", "http://127.0.0.1:9",
                    "--limit", "1", "--chat-timeout", "2",
                    "--results-dir", tmp, "--run-id", "dead-1",
                ])
            self.assertEqual(code, 1)
            self.assertIn("potato agent-eval", err.getvalue())
            self.assertEqual(list(Path(tmp).glob("*.json")), [])


class BuildRecordTest(unittest.TestCase):
    def test_record_validates_and_carries_agent_metrics(self) -> None:
        runs = [[_tr("a", True), _tr("b", False)]]
        record = agenteval.build_record(
            "agent-test-1", runs, "fake-model.gguf", "http://127.0.0.1:8080",
            "evals/agent_tasks/tasks", "deadbeefdeadbeef", 1, 0.0, 42, None,
        )
        from potato.schema import validate

        self.assertEqual(validate(record), [])
        self.assertEqual(record["run_id"], "agent-test-1")
        self.assertAlmostEqual(record["metrics"]["agent_success"], 0.5)
        self.assertEqual(record["spread"]["agent_samples"], [0.5])
        detail = record["spread"]["agent_detail"]
        # Wilson 95% for 1 success out of 2 trials. Verified against the
        # published Wilson score interval (z=1.96): [0.0945, 0.9055].
        self.assertEqual(detail["wilson95"], [0.0945, 0.9055])
        self.assertEqual(record["config"]["suite_sha256"], "deadbeefdeadbeef")
        self.assertEqual(record["config"]["temperature"], 0.0)
        # model file does not exist -> sha/size stay null, never invented
        self.assertIsNone(record["model_sha256"])


class ReportIntegrationTest(unittest.TestCase):
    def test_agent_samples_flow_into_report(self) -> None:
        from potato.report import _metric_summary_from_record

        rec = {
            "metrics": {"agent_success": 0.64},
            "spread": {"agent_samples": [0.62, 0.64, 0.66]},
        }
        s = _metric_summary_from_record(rec, "agent_success")
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["median"], 0.64)
        self.assertGreater(s["stdev"], 0.0)


if __name__ == "__main__":
    unittest.main()





