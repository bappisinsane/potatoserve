# Memory — PotatoServe

> Read this at the start of every session. Update it at the end. Keep it short and current.

## Owner context
- Bappi — self-taught AI/ML engineer, Bangladesh. Goal: remote ML / edge-AI roles; prefers deep technical work over commodity coding.
- Dev GPU (confirmed via `nvidia-smi`, 2026-10-11): **GTX 1660 Ti 6 GB (TU116, Turing, SM 7.5)** — not the GTX 1060 previously assumed. All CUDA arch targets are `75`.

## One-paragraph summary
Toolkit around llama.cpp that auto-tunes quantization, KV-cache, offload and decoding for low-VRAM GPUs, measures speed/quality/agent success honestly, and contributes fixes upstream.

## Status
- Phase: 0 (steps 1–4 done) — updated 2026-10-11
- Done: repo scaffolded per project-spec §9; llama.cpp submodule pinned to v0.6.0; CUDA + Vulkan build scripts (flags verified against pinned source); `potato probe` + section-8 result schema; baseline runner (`potato bench` / `scripts/baseline_bench.*`) with ≥3 repeats. 16 unit tests pass (stdlib only, no deps added).
- Blocked on hardware/toolchain: **CUDA Toolkit (nvcc) is not installed**, so a CUDA build + first baseline cannot run yet. Vulkan runtime + dev SDK headers also not confirmed.
- Next action: install CUDA Toolkit, build via `scripts/build_cuda.ps1` (arch 75), then run `potato bench -m <model.gguf> --ngl 99` and paste the JSON record back.

## Decisions log
| Date | Decision | Why |
|------|----------|-----|
| 2026-10-11 | Wrap llama.cpp rather than fork | Less maintenance; fork only in `lab/` |
| 2026-10-11 | KL divergence as quality filter | More sensitive than perplexity alone |
| 2026-10-11 | Pin llama.cpp to **v0.6.0** (commit `d8123504…`), not master | "Recent stable" = a tagged release; master build tags (e.g. `b11553`) are rolling |
| 2026-10-11 | CLI on stdlib `argparse`, not `typer` | rules.md forbids deps without owner approval; single swap point in `potato/cli.py` if `typer` approved later |
| 2026-10-11 | probe/schema/bench use stdlib only (no psutil/GPUtil) | Same no-deps rule; shell out to `nvidia-smi`/`vulkaninfo` + WMI/`/proc` |
| 2026-10-11 | `CMAKE_CUDA_ARCHITECTURES=75` set explicitly, `GGML_NATIVE=OFF` | Reproducible regardless of which GPU is attached at build time |
| 2026-10-11 | CUDA build must fail loudly, never fabricate | Honest measurement (PRD pillar); `potato bench` exits non-zero with install guidance if `llama-bench` is missing |

## Models chosen (fill in)
- Dense: `<HF id>` — license: `<…>`
- MoE (optional): `<HF id>` — license: `<…>`

## Environment fingerprint (detected 2026-10-11 via `potato probe`)
- OS: Windows 11
- CPU: Intel Core i5-9400F @ 2.90GHz (6C/6T, no HT)
- RAM: 15.9 GB
- GPU: NVIDIA GeForce GTX 1660 Ti, 6144 MiB, driver 616.56, compute_cap 7.5 → CUDA arch **75** (TU116, Turing)
- Vulkan: runtime present (`vulkaninfo`); device = GTX 1660 Ti (dev SDK headers unconfirmed)
- CUDA Toolkit (nvcc): **NOT installed** — CUDA build blocked until installed
- Python: 3.14.0 (3.10 also present)
- Build tools on PATH: git 2.51.0, ninja (pip); **cmake not on PATH**; `cl` not on PATH (VS 2019 Build Tools present)
- llama.cpp: submodule `third_party/llama.cpp` @ **v0.6.0** = `d81235049384534c167caea52b85a694f6103d14`
- Build flags (from scripts, not yet run): `-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF` (CUDA); `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF` (Vulkan)

## Open questions
- Which models give the best tool-calling at ≤ 9B?
- Does speculative decoding help on Pascal at all?
- Which upstream issue is the best first PR?

## Gotchas learned
(append as you go — e.g. flag renames, backend quirks, thermal behavior)

## Gotchas learned
- PowerShell treats a native command's stderr (incl. git's LF/CRLF warnings and unittest's progress) as an error exit — pipe to `Out-File`/`cmd /c … 2>&1` to read true output.
- `platform.processor()` on Windows returns an arch string, not the CPU name; use `(Get-CimInstance Win32_Processor).Name` for the fingerprint.
- The annotated tag object for v0.6.0 (`69ecae9…`) differs from the commit it points to (`d8123504…`); the submodule gitlink correctly stores the commit.
- `git submodule add` timed out at the 30s shell limit on a large repo; clone as a detached `Start-Process` and register the gitlink manually instead.

## Useful commands (verified against pinned v0.6.0 source)
- Probe: `python -m potato.cli probe --out hardware.json`
- Baseline (≥3 repeats): `python -m potato.cli bench -m model.gguf --ngl 99 -fa auto --results-dir results` (or `scripts/baseline_bench.ps1 -Model model.gguf`)
- Raw bench: `llama-bench -m model.gguf -ngl 99 -p 512 -n 128 -r 3 -o json` (default reps is 5; `-r 3` is required for the spec minimum)
- Build CUDA: `scripts/build_cuda.ps1` (arch 75) — needs CUDA Toolkit (nvcc)
- Build Vulkan: `scripts/build_vulkan.ps1` — needs Vulkan SDK (glslc)
- KL base: `llama-perplexity -m ref.gguf -f text.txt --kl-divergence-base ref.kld`
- Server: `llama-server -m model.gguf -ngl 99 -c 16384 -fa on --cache-type-k q8_0 --cache-type-v q8_0`
- Verified flag names (grep'd from pinned source, not memory): llama-bench `-r/--repetitions -p/--n-prompt -n/--n-gen -o/--output -ngl -fa -ctk -ctv`; common/arg.cpp `-c/--ctx-size -ctk -ctv -fa -ngl/--gpu-layers --kl-divergence(-base)`.

## Session log
- 2026-10-11 — project docs created.
- 2026-10-11 — Phase 0 steps 1–4 implemented (scaffold, submodule v0.6.0, build scripts, probe+schema, baseline). All flag names checked against the pinned source. Environment fingerprinted via `potato probe`. CUDA build blocked: nvcc not installed.
