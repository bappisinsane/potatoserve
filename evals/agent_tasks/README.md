# evals/agent_tasks/ — agent task suite

Task definitions and programmatic checkers (Phase 3). Per docs/project-spec.md
section 6: 50 tasks, each with a pass/fail check — file edit, multi-step shell in
a sandbox dir, JSON-schema tool calls, search-and-summarize over a local corpus,
and error recovery.

Format (planned): one YAML task definition + a Python checker per task. New tasks
are added as `YAML + checker` (architecture.md section 7).

Nothing to run yet (Phase 0).
