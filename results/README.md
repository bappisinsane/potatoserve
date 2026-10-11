# results/ — curated, published results

Curated result tables and JSON that are committed to the repo (docs/project-spec.md
section 9). Raw, gitignored run output lives in `runs/`.

Every published number must be reproducible by a single command and must record the
environment fingerprint (llama.cpp commit, build flags, driver, OS, GPU, model hash).

## Phase 0 baseline (real, measured)

`baseline-20261011T133403Z.json` — first baseline, measured on the dev GPU:

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti (6 GB, Vulkan backend) |
| llama.cpp | v0.6.0, commit `d812350` |
| Driver / OS | 616.56 / Windows |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M (sha256 `74a4da8c…d7a9db`) |
| Config | `-ngl 99`, flash-attn auto, ctx f16 KV, pp 512 / tg 128, 3 repeats |
| prefill_tps | 4227.0 (stddev 90.0) |
| decode_tps | 152.7 (stddev 17.8) |

Regenerate with:
`python -m potato.cli bench -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf --ngl 99 -fa auto --results-dir results`

Remaining §8 metrics (`vram_peak_mb`, `kl_mean`, `top1_agree`, `ppl`, `agent_success`) are
`null` here by design — they are produced by Phase 1+ (llama-server / perplexity / agent-eval),
never fabricated.
