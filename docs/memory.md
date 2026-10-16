# Memory — PotatoServe

> Read this at the start of every session. Update it at the end. Keep it short and current.

## Owner context
- Bappi — self-taught AI/ML engineer, Bangladesh. Goal: remote ML / edge-AI roles; prefers deep technical work over commodity coding.
- Dev GPU (confirmed via `nvidia-smi`, 2026-10-11): **GTX 1660 Ti 6 GB (TU116, Turing, SM 7.5)** — not the GTX 1060 previously assumed. All CUDA arch targets are `75`.

## One-paragraph summary
Toolkit around llama.cpp that auto-tunes quantization, KV-cache, offload and decoding for low-VRAM GPUs, measures speed/quality/agent success honestly, and contributes fixes upstream.

## Status
- Phase: 2 (quality gate — code complete, measurements pending) — updated 2026-10-16
- Done: repo scaffolded per project-spec §9; llama.cpp submodule pinned to v0.6.0; CUDA + Vulkan build scripts; `potato probe` + section-8 result schema; Phase 1 harness (live VRAM sampling, append-only store, median+spread reports); **Phase 2 quality gate code complete**: `potato/quality.py` (llama-perplexity reference/KL parsing + ≥3-repeat aggregation), `potato/quant.py` (llama-quantize + llama-imatrix → `quant_manifest.json`), `potato bench --corpus/--kl-base/--quality-only/--dry-run`, `potato quant --dry-run`, `potato report --curve` (quality-vs-size table). **89 unit tests pass** (stdlib only, no deps added).
- **Vulkan build works end-to-end** (2026-10-11): `llama-bench --list-devices` reports `Vulkan0: NVIDIA GeForce GTX 1660 Ti (6180 MiB)`. Vulkan SDK `1.4.363.0` installed at `C:\VulkanSDK` (per-user, no admin). The `load_backend: failed to find ggml_backend_init` lines are harmless duplicate-scan warnings — the device registers and runs fine.
- **First real baseline measured** → `results/baseline-20261011T133403Z.json`: Qwen2.5-0.5B-Instruct Q4_K_M, `-ngl 99`, fa auto, pp512/tg128, 3 repeats → prefill **4227.0 t/s** (σ90.0), decode **152.7 t/s** (σ17.8). All numbers real (measured on the 1660 Ti); arithmetic verified against the 3 samples. §8 metrics `vram_peak_mb/kl_mean/top1_agree/ppl/agent_success` are `null` by design (Phase 1+).
- Toolchain (2026-10-11, done by Cline with owner permission): **cmake 4.4.4 installed** (`C:\Program Files\CMake\bin`, on user PATH). **CUDA 12.4.1 installer downloaded byte-exact** (`runs/cuda_12.4.1_551.78_windows.exe`, 3,194,092,728 B) but **not yet installed** — needs one **UAC approval only the owner can click**. Build scripts now re-exec inside `vcvars64.bat` so MSVC `cl` is on PATH.
- Next action: **Phase 2 needs real GPU measurements** (Vulkan build already works — no UAC needed):
  1. Reference logits (once): `python -m potato.cli bench -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf -ngl 99 --corpus data/wiki.test.raw --kl-base results/qwen-ref.kld --save-kl-base --quality-only --run-id qwen-ref-1`
  2. Quality per variant: same command without `--save-kl-base`, per candidate model.
  3. Curve: `python -m potato.cli report --curve` after running 2+ quant variants (`potato quant <f16.gguf> --types Q4_K_M Q3_K_M ...` needs an F16 source GGUF, e.g. downloaded from HF — no HF/convert deps without owner approval).
  Corpus files (wikitext) still need downloading — rules.md: verify real numbers only. The KL-threshold decision (last Phase 2 checkbox) needs this data.

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
| 2026-10-11 | Install **CUDA 12.4** (not 13.x) for the 1660 Ti | llama.cpp pinned v0.6.0 Windows CUDA CI tests exactly `12.4` and `13.4`; 12.x unambiguously supports Turing SM 7.5 (13.x dropped Maxwell/Pascal/Volta). Owner approved 12.4 as the safe choice. winget pkg `12.4` = installer `cuda_12.4.1_551.78` |
| 2026-10-11 | Omit `-DGGML_CUDA_CCCL_VERSION` in our build script | Pinned CMake: empty = "use the CCCL bundled with the installed CUDA Toolkit"; CI pins it only for reproducibility + to avoid a CCCL network fetch |

## Models chosen (fill in)
- Dense: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (Q4_K_M, 397 MB) — license: Apache-2.0. Used for the Phase 0 Vulkan baseline (`models/qwen2.5-0.5b-instruct-q4_k_m.gguf`, gitignored).
- MoE (optional): `<HF id>` — license: `<…>`

## Environment fingerprint (detected 2026-10-11 via `potato probe`)
- OS: Windows 11
- CPU: Intel Core i5-9400F @ 2.90GHz (6C/6T, no HT)
- RAM: 15.9 GB
- GPU: NVIDIA GeForce GTX 1660 Ti, 6144 MiB, driver 616.56, compute_cap 7.5 → CUDA arch **75** (TU116, Turing)
- Vulkan: runtime present (`vulkaninfo`); device = GTX 1660 Ti. **Vulkan SDK `1.4.363.0` installed** at `C:\VulkanSDK` (per-user, no admin) → Vulkan build works.
- CUDA Toolkit (nvcc): **NOT installed** — CUDA build blocked until installed. CUDA 12.4.1 installer staged byte-exact in `runs/` but not run (needs the one owner UAC click). Decided: install **CUDA 12.4** (safest for Turing SM 7.5, in pinned CI).
- Python: 3.14.0 (3.10 also present); package installed editable (`pip install -e .`) → `potato` console command available.
- Build tools: git 2.51.0, ninja (pip), **cmake 4.4.4 installed** (`C:\Program Files\CMake\bin`, user PATH). `cl` not on raw PATH (VS 2019 Build Tools present) — build scripts re-exec inside `vcvars64.bat` to expose it.
- llama.cpp: submodule `third_party/llama.cpp` @ **v0.6.0** = `d81235049384534c167caea52b85a694f6103d14`
- Build flags: CUDA (not yet run) `-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF`; Vulkan (built OK) `-DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF`.

## Open questions
- Which models give the best tool-calling at ≤ 9B?
- Does speculative decoding help on Pascal at all?
- Which upstream issue is the best first PR?

## Gotchas learned
- PowerShell treats a native command's stderr (incl. git's LF/CRLF warnings and unittest's progress) as an error exit — pipe to `Out-File`/`cmd /c … 2>&1` to read true output.
- `platform.processor()` on Windows returns an arch string, not the CPU name; use `(Get-CimInstance Win32_Processor).Name` for the fingerprint.
- The annotated tag object for v0.6.0 (`69ecae9…`) differs from the commit it points to (`d8123504…`); the submodule gitlink correctly stores the commit.
- `git submodule add` timed out at the 30s shell limit on a large repo; clone as a detached `Start-Process` and register the gitlink manually instead.

## Useful commands (verified against pinned v0.6.0 source)
- Probe: `python -m potato.cli probe --out hardware.json`
- Baseline (≥3 repeats): `python -m potato.cli bench -m model.gguf --ngl 99 -fa auto --results-dir results` (or `scripts/baseline_bench.ps1 -Model model.gguf`)
- Raw bench: `llama-bench -m model.gguf -ngl 99 -p 512 -n 128 -r 3 -o json` (default reps is 5; `-r 3` is required for the spec minimum)
- Report (median+spread) a stored record: `python -m potato.cli report --run-id <id>` (or no flag → all records)
- Compare two stored records: `python -m potato.cli report --compare BASE_ID CAND_ID [--metric decode_tps]`
- List stored records: `python -m potato.cli report --list`
- Quality (Phase 2): reference logits once — `python -m potato.cli bench -m model.gguf -ngl 99 --corpus wiki.test.raw --kl-base results/ref.kld --save-kl-base --quality-only --run-id ref-1`; then per candidate drop `--save-kl-base` (adds `--kl-divergence` automatically). `--dry-run` prints the exact llama-bench/llama-perplexity commands without running.
- Quant (Phase 2): `python -m potato.cli quant src-f16.gguf --types Q4_K_M Q3_K_M --imatrix-from wiki.train.raw --dry-run` (real run needs the source GGUF present; writes `quant_manifest.json`).
- Quality-vs-size curve (Phase 2 exit artifact): `python -m potato.cli report --curve`
- Build CUDA: `scripts/build_cuda.ps1` (arch 75) — needs CUDA Toolkit (nvcc)
- Build Vulkan: `scripts/build_vulkan.ps1` — needs Vulkan SDK (glslc)
- KL base: `llama-perplexity -m ref.gguf -f text.txt --kl-divergence-base ref.kld`
- Server: `llama-server -m model.gguf -ngl 99 -c 16384 -fa on --cache-type-k q8_0 --cache-type-v q8_0`
- Verified flag names (grep'd from pinned source, not memory): llama-bench `-r/--repetitions -p/--n-prompt -n/--n-gen -o/--output -ngl -fa -ctk -ctv`; common/arg.cpp `-c/--ctx-size -ctk -ctv -fa -ngl/--gpu-layers --kl-divergence(-base)`.

## Session log
- 2026-10-11 — project docs created.
- 2026-10-11 — Phase 0 steps 1–4 implemented (scaffold, submodule v0.6.0, build scripts, probe+schema, baseline). All flag names checked against the pinned source. Environment fingerprinted via `potato probe`. CUDA build blocked: nvcc not installed.
- 2026-10-11 — Wired `potato = "potato.cli:main"` console entry point; fixed `potato bench` to accept llama-bench short flags (`-fa`); 16 tests pass. Verified CUDA 12.4/13.4 from pinned v0.6.0 Windows CI; owner chose **CUDA 12.4**. Exact winget install commands issued (cmake + CUDA 12.4); awaiting owner to install + paste build/bench output.
- 2026-10-11 — Owner granted permission for Cline to run the toolchain installs. cmake 4.4.4 installed. CUDA winget stalled (exe half-downloaded, `winget` wouldn't exit) → killed it and **resumed the byte-exact download with `curl -C -`**, staged `runs/cuda_12.4.1_551.78_windows.exe` (3,194,092,728 B verified). Installer demands admin even for `-?`/extract, so the **one UAC click is unavoidable**; wrote `runs/install_cuda.ps1` (silent, excludes the bundled driver so 616.56 is kept).
- 2026-10-11 — Pivoted to **Vulkan** to get a real baseline without UAC: installed Vulkan SDK `1.4.363.0` per-user; build scripts now re-exec inside `vcvars64.bat`; `llama-bench --list-devices` → `Vulkan0: GTX 1660 Ti (6180 MiB)`. Downloaded Qwen2.5-0.5B-Instruct Q4_K_M (GGUF magic verified) and measured the first baseline → `results/baseline-20261011T133403Z.json`: prefill **4227.0 t/s** (σ90.0), decode **152.7 t/s** (σ17.8), 3 repeats, `-ngl 99`, fa auto. `bench.py` now records per-kind pp/tg sizes, model sha256, and live driver/OS/GPU into the §8 record. `results/` README documents the baseline; `.gitignore` excludes `models/` + `*.gguf`. 16 tests pass.
- 2026-10-13 — **Phase 1 benchmark harness completed** (exit criterion met). Wired the three previously-untracked modules into the pipeline: `potato.bench` now launches via `potato.runner` (live `nvidia-smi` VRAM sampling during the run → real `vram_peak_mb`, recorded `null` if unavailable) and persists via `potato.store` (append-only, ADR-5 — re-running an existing `run_id` refuses to clobber). Added `potato report` CLI (`--run-id`/`--compare`/`--metric`/`--list`). **Fixed a latent bug**: `report._metric_summary_from_record` looked for `spread["decode_tps"]` but `bench` writes `spread["decode_samples_ts"]`, so reports would have shown `n=1, stdev=0` for the very metrics with 3 real repeats; it now derives median/stdev from the per-repeat sample arrays. Added `render_compare_all` (all-metric A/B table) and a min/max column. 30 new stdlib-only tests (`test_runner`/`test_store`/`test_report`); **46 tests pass** under both pytest and `unittest discover` (CI). Stdlib only, no deps added.
- 2026-10-16 — **Phase 2 quality gate code complete** (89 tests pass under pytest and `unittest discover`). Added `potato/quality.py` (llama-perplexity runner: reference/KL/plain modes, output parser written from the pinned v0.6.0 printf formats — incl. UTF-8 `±`/`Δ` lines, ≥3-repeat aggregation keeping every sample) and `potato/quant.py` (llama-quantize + optional llama-imatrix variants, deterministic variant paths, append-only `quant_manifest.json` with measured sizes/SHA-256, `--dry-run` never executes or writes). Wired `potato bench --corpus/--kl-base/--save-kl-base/--chunks/--quality-repeats/--quality-only/--quality-ctx/--quality-binary/--dry-run` and the `quant`/`report --curve` CLIs; quality metrics merge into the speed record (or a quality-only record) with `config.model_size_bytes` for the curve. Added `runner.find_llama_binary` (searches build-cuda/build-vulkan/build then PATH) + `runner.pinned_commit` (git provenance) and UTF-8-safe subprocess decoding (Windows console mojibake). Build scripts now also build `llama-quantize` + `llama-imatrix`. Curve/quality/quant covered by `test_report`/`test_quality`/`test_quant` (53 new tests). Measurement-dependent exit items (KL threshold, curve for both models) await GPU runs — commands issued in Next action; nothing fabricated.

