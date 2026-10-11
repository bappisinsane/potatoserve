# PRD — PotatoServe

> Make capable local AI agents usable on cheap, old GPUs — and prove it with honest numbers.

## 1. Vision
PotatoServe is a toolkit that finds the best quantization / offload / context / decoding configuration for a given GPU + model, verifies it with rigorous benchmarks (speed, memory, quality drift, agent success), and feeds the wins back upstream to llama.cpp.

## 2. Problem
- Most "run an agent locally" guides assume 12–24 GB of VRAM. Millions of users have 4–8 GB cards (GTX 10-series, RX 5xx/5000, laptop GPUs).
- Picking weight quant, KV-cache type, GPU layers, context length and speculative decoding is trial and error, and some combinations silently break on older hardware.
- Published numbers usually report speed only — not whether the agent still *works* after aggressive compression.

## 3. Goals
| ID | Goal |
|----|------|
| G1 | Run a tool-calling agent model at usable speed inside 6 GB VRAM at ≥ 16k context |
| G2 | Auto-tuner that emits a ready-to-run `llama-server` command for any NVIDIA/AMD/Intel GPU |
| G3 | Benchmark suite covering speed, peak memory, quality drift (KL divergence) and agent task success |
| G4 | ≥ 1 merged upstream contribution to llama.cpp/ggml (stretch: ≥ 3) |
| G5 | Public, reproducible write-up + results repo |

## 4. Non-goals
- Not a new inference engine, not a model trainer, not a chat UI.
- Not datacenter serving (vLLM / SGLang territory).

## 5. Personas
- **Budget builder** — 6 GB GPU, wants a local coding/agent assistant that actually works.
- **Edge engineer** — evaluating llama.cpp for embedded targets, needs trustworthy numbers.
- **Maintainer / hiring manager** — judging engineering depth from the repo; needs clean, reproducible work.

## 6. Functional requirements
- **FR-1 `potato probe`** — detect GPUs, VRAM, driver, available backends (CUDA / Vulkan / CPU), RAM, CPU features.
- **FR-2 `potato quant`** — produce GGUF variants (selectable quant types, optional imatrix) and record size.
- **FR-3 `potato bench`** — prefill + decode tok/s at several context depths, peak VRAM, quality (KL divergence + perplexity vs. a high-precision baseline).
- **FR-4 `potato agent-eval`** — run the agent task suite against a running server; report success rate, tool-call validity, steps, wall time.
- **FR-5 `potato tune`** — search the config space under a VRAM budget; output best configs per objective (speed / quality / long-context).
- **FR-6 `potato serve`** — launch `llama-server` with a chosen config and health-check it.
- **FR-7 `potato report`** — Markdown + JSON + charts with a full environment fingerprint.
- **FR-8 Results store** — every run records llama.cpp commit, build flags, driver, model hash, config.

## 7. Non-functional requirements
- **Reproducible:** one command regenerates any published table.
- **Honest:** ≥ 3 runs per point, report median + spread; no cherry-picking.
- **Portable:** Windows + Linux.
- **Practical:** full tuning of one model finishes in < 2 h on the dev machine.

## 8. Success metrics (all targets are hypotheses — revise after Phase 0 baselines)
| Metric | Target |
|--------|--------|
| Decode speed, 7–9B Q4-class model, 16k ctx, 6 GB GPU | ≥ 15 tok/s |
| Agent-suite success vs. best-fit baseline | within 5 points |
| KL divergence of recommended config | under threshold set in Phase 2 |
| Upstream PRs merged | ≥ 1 |
| Reproduction | single command |

## 9. Risks & open questions
- Consumer Pascal GPUs have no tensor cores — expect CUDA kernel paths to differ from newer cards; verify early.
- Upstream review is slow and selective → start with a small, well-tested PR.
- llama.cpp moves daily → pin commits; re-baseline monthly.
- Open: which two models? (decide in Phase 0; record in `memory.md`)

## 10. Release criteria (v1.0)
All FRs pass on ≥ 2 GPUs of different vendors; results repo reproducible by a stranger; write-up published; ≥ 1 upstream PR opened with evidence.
