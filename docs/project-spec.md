# Project Spec — PotatoServe

## 1. Targets
| Item | Spec |
|------|------|
| Primary dev GPU (assumed — confirm) | GTX 1060 6 GB (Pascal, CUDA SM 6.1) |
| Secondary | any AMD/Intel GPU reachable via Vulkan; CPU-only run |
| OS | Windows 10/11 and Ubuntu 22.04+ |
| Model classes | one dense 7–9B instruct model with tool calling; optionally one small MoE (expert offload to CPU) |

## 2. Stack
- Orchestration: Python 3.11+, `typer` (CLI), `pydantic` (config), `sqlite3`, `matplotlib`, `jinja2`.
- Inference: llama.cpp (git submodule, pinned commit), built from source with CUDA and Vulkan; verify flag names against the pinned build — they change.
- Upstream track: C++17, CUDA, GLSL (Vulkan), `test-backend-ops`.

## 3. Config space
| Axis | Values to explore |
|------|-------------------|
| Weight quant | Q3_K_M, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K (± imatrix) |
| KV cache K / V | f16, q8_0, q4_0 (quantized V needs flash attention) |
| GPU layers (`-ngl`) | 0 … all |
| MoE expert placement | CPU vs GPU via tensor overrides |
| Context | 4k, 8k, 16k, 32k |
| Batch / ubatch | 128 … 2048 |
| Speculative decoding | off, draft model, n-gram/lookup |
| Threads | physical cores ± 1 |

## 4. Memory model (used by tuner to prune)
`KV_bytes = 2 × n_layers × n_ctx × n_kv_heads × head_dim × bytes_per_elem`
`VRAM ≈ weights_on_gpu + KV_on_gpu + compute_buffers + driver_overhead`
Treat `compute_buffers` and `driver_overhead` as measured, not guessed (record in Phase 0).

## 5. Quality measurement
1. Produce reference logits with the highest-precision variant that runs (CPU/RAM offload allowed): `llama-perplexity --kl-divergence-base`.
2. For each candidate: `--kl-divergence` on the same text → mean KL, top-1 agreement, perplexity delta.
3. Agent-eval is the final gate; KL is the cheap filter.

## 6. Agent-eval suite
- 50 tasks, each with a programmatic pass/fail check: file edit, multi-step shell in a sandbox dir, JSON-schema tool calls, search-and-summarize over a local corpus, error recovery.
- Metrics: success %, invalid tool-call %, mean steps, wall time, tokens used.
- Fixed seeds, temperature 0 for reproducibility, plus a small sampled run for robustness.

## 7. CLI contract
```
potato probe                       -> hardware.json
potato quant  <hf-id> --types ...  -> models/*.gguf + quant_manifest.json
potato bench  <gguf> --cfg cfg.yaml-> runs/<id>/bench.json
potato agent-eval --endpoint URL   -> runs/<id>/agent.json
potato tune   <hf-id> --vram 6GB --objective speed|quality|longctx
potato serve  --cfg best.yaml
potato report --runs runs/ --out report/
```

## 8. Result record (JSON)
`{run_id, date, llama_cpp_commit, build_flags, driver, os, gpu, model_id, model_sha256, config, metrics:{prefill_tps, decode_tps, vram_peak_mb, kl_mean, top1_agree, ppl, agent_success}, repeats, spread}`

## 9. Repo layout
```
potatoserve/
  potato/            # python package
  third_party/llama.cpp  # submodule (pinned)
  lab/               # upstream experiments, patches, perf notes
  evals/agent_tasks/ # task definitions + checkers
  runs/              # gitignored raw runs; curated results in results/
  results/           # published tables + JSON
  docs/  tests/  scripts/
```

## 10. Testing
Unit tests for parsers/memory model; integration smoke test with a tiny model in CI; golden-file test for report generation; every benchmark script has a `--dry-run`.

## 11. Licensing
Code MIT (matches llama.cpp). Models keep their own licenses; never redistribute weights.

## 12. Deliverables
CLI toolkit • results repo • ≥ 1 upstream PR (with perf evidence) • long-form write-up • 5-min demo video.
