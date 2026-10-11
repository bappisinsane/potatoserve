# Memory — PotatoServe

> Read this at the start of every session. Update it at the end. Keep it short and current.

## Owner context
- Bappi — self-taught AI/ML engineer, Bangladesh. Goal: remote ML / edge-AI roles; prefers deep technical work over commodity coding.
- Primary dev GPU (assumed — confirm): GTX 1060 6 GB.

## One-paragraph summary
Toolkit around llama.cpp that auto-tunes quantization, KV-cache, offload and decoding for low-VRAM GPUs, measures speed/quality/agent success honestly, and contributes fixes upstream.

## Status
- Phase: 0 (not started) — created 2026-10-11
- Next action: build llama.cpp (CUDA + Vulkan), record baselines.

## Decisions log
| Date | Decision | Why |
|------|----------|-----|
| 2026-10-11 | Wrap llama.cpp rather than fork | Less maintenance; fork only in `lab/` |
| 2026-10-11 | KL divergence as quality filter | More sensitive than perplexity alone |

## Models chosen (fill in)
- Dense: `<HF id>` — license: `<…>`
- MoE (optional): `<HF id>` — license: `<…>`

## Environment fingerprint (fill in)
GPU / driver / CUDA toolkit / OS / llama.cpp commit / build flags.

## Open questions
- Which models give the best tool-calling at ≤ 9B?
- Does speculative decoding help on Pascal at all?
- Which upstream issue is the best first PR?

## Gotchas learned
(append as you go — e.g. flag renames, backend quirks, thermal behavior)

## Useful commands (verify against pinned build)
- Bench: `llama-bench -m model.gguf -ngl 99 -p 512 -n 128 -r 3`
- KL base: `llama-perplexity -m ref.gguf -f text.txt --kl-divergence-base ref.kld`
- Server: `llama-server -m model.gguf -ngl 99 -c 16384 -fa on --cache-type-k q8_0 --cache-type-v q8_0`

## Session log
- 2026-10-11 — project docs created.
