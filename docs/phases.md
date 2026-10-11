# Phases — PotatoServe

Durations are estimates for focused part-time work; adjust after Phase 0.

## Phase 0 — Foundations (≈ 1 week)
- [ ] Build llama.cpp from source: CUDA (SM 6.1) and Vulkan; record flags
- [ ] Pick 2 models; record exact HF IDs + licenses in `memory.md`
- [ ] Baseline `llama-bench` numbers; measure real VRAM overhead
- [ ] Repo skeleton, CI, results schema
**Exit:** baseline table committed, reproducible by script.

## Phase 1 — Benchmark harness (≈ 1–2 weeks)
- [ ] `probe`, `runner`, `bench` (speed + VRAM)
- [ ] Results store + `report`
**Exit:** `potato bench` produces a report with ≥ 3 repeats and spread.

## Phase 2 — Quality gate (≈ 2 weeks)
- [ ] Reference logits + KL/perplexity pipeline
- [ ] `quant` with imatrix variants
- [ ] Decide KL threshold from data (plot KL vs. agent success on a small sample)
**Exit:** quality-vs-size curve for both models.

## Phase 3 — Agent-eval (≈ 2 weeks)
- [ ] 50-task suite + checkers; validate checkers with a strong reference model
- [ ] Integrate into bench + tune
**Exit:** success rates with confidence intervals for ≥ 5 configs.

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
