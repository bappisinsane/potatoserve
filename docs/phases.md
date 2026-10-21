# Phases — PotatoServe

Durations are estimates for focused part-time work; adjust after Phase 0.

## Phase 0 — Foundations (≈ 1 week)
- [ ] Build llama.cpp from source: CUDA (SM 6.1) and Vulkan; record flags
- [ ] Pick 2 models; record exact HF IDs + licenses in `memory.md`
- [ ] Baseline `llama-bench` numbers; measure real VRAM overhead
- [ ] Repo skeleton, CI, results schema
**Exit:** baseline table committed, reproducible by script.

## Phase 1 — Benchmark harness (≈ 1–2 weeks)
- [x] `probe`, `runner`, `bench` (speed + VRAM)
- [x] Results store + `report`
**Exit:** `potato bench` produces a report with ≥ 3 repeats and spread.
**Status (2026-10-13):** exit criterion met. `potato bench` launches via
`potato.runner` (live `nvidia-smi` VRAM sampling), persists through the
append-only `potato.store` (ADR-5), and prints a median+spread report; `potato
report` renders stored records and A/B comparisons. 46 unit tests pass.

## Phase 2 — Quality gate (≈ 2 weeks)
- [x] Reference logits + KL/perplexity pipeline
- [x] `quant` with imatrix variants
- [ ] Decide KL threshold from data (plot KL vs. agent success on a small sample)
**Exit:** quality-vs-size curve for both models.
**Status (2026-10-16):** code complete — `potato bench --corpus [--kl-base]`
runs llama-perplexity (reference / KL modes, ≥3 repeats, live VRAM sampling
during the pass), `potato quant` builds imatrix-variant GGUFs +
`quant_manifest.json`, and `potato report --curve` renders the quality-vs-size
table. 89 unit tests pass (stdlib only). The measurement-dependent items (KL
threshold decision, curve for both models) **await GPU runs** — exact commands
in `memory.md`; no numbers were fabricated.

## Phase 3 — Agent-eval (≈ 2 weeks)
- [x] 50-task suite + checkers
- [ ] Validate checkers with a strong reference model
- [x] Integrate into bench (tune lands in Phase 4)
**Exit:** success rates with confidence intervals for ≥ 5 configs.
**Status (2026-10-21):** code complete — `potato/agenteval.py` runs the
50-task suite (`evals/agent_tasks/tasks/`, 5 categories × 10) against a live
`llama-server`, executing model-emitted tool calls **client-side in a sandbox**
(path-traversal guarded) and scoring them with 11 programmatic checkers
(`evals/agent_tasks/checkers.py`). Metrics: `agent_success` (median of per-repeat
rates) + Wilson 95% interval over all trials, invalid tool-call %, mean steps,
wall time, tokens. Wired as `potato agent-eval` and `potato bench
--agent-endpoint`; the metric flows into the §8 record and `potato report`.
130 unit tests pass (stdlib only) — the whole loop is covered against a fake
OpenAI server, so **no GPU/model is needed for the tests**. The remaining
checkbox (*validate checkers with a strong reference model*) and the exit
criterion (≥ 5 configs) **await GPU runs** — exact commands in `memory.md`;
no numbers were fabricated.

## Phase 4 — Auto-tuner (≈ 2 weeks)
- [ ] Memory model validated against measured VRAM (±5%)
- [ ] Staged search + Pareto output + `serve`
**Exit:** `potato tune` recommends configs that beat hand-tuned defaults on dev GPU.

## Phase 5 — Decoding & caching on old GPUs (≈ 2 weeks)
- [ ] Prompt-cache / slot reuse effect on agent loops
- [ ] Speculative decoding (draft + n-gram) — when does it help, when does it break on Pascal?
**Exit:** documented matrix of what works/fails on each backend.

## Phase 6 — Upstream contribution (≈ 3–4 weeks)
- [ ] Read CONTRIBUTING + AI-assistance policy; build `test-backend-ops`
- [ ] Pick one scoped issue (bug repro, kernel tuning for older arch, docs+tests)
- [ ] Small PR with perf numbers; respond to review
**Exit:** PR opened with evidence (merged = bonus).

## Phase 7 — Publish (≈ 1–2 weeks)
- [ ] Write-up, charts, demo video, README
- [ ] Reproducibility check by fresh machine/VM
**Exit:** v1.0 tagged.

## Definition of done (every phase)
Tests pass • results reproducible • `memory.md` updated • no unexplained numbers.
