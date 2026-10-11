# scripts/ — build & helper scripts

`llama.cpp` is pinned as a submodule at `third_party/llama.cpp` (tag **v0.6.0**,
commit `d81235049384534c167caea52b85a694f6103d14`). All CMake / CLI flag names
below are verified against that pinned source, not memory.

## Build scripts (Phase 0 step 2)

| Script | Backend | Notes |
|---|---|---|
| `build_cuda.sh` / `build_cuda.ps1` | CUDA | `CMAKE_CUDA_ARCHITECTURES=75` (GTX 1660 Ti, TU116 Turing). `GGML_CUDA=ON`, `GGML_NATIVE=OFF`. |
| `build_vulkan.sh` / `build_vulkan.ps1` | Vulkan | `GGML_VULKAN=ON`. Portable to AMD/Intel GPUs. Needs Vulkan SDK (`glslc`) for shader-gen. |

Targets built: `llama-bench`, `llama-server`, `llama-perplexity`.

## Verified flags (pinned v0.6.0)
- **llama-bench**: `-r/--repetitions` (default **5** — always pass `-r 3`),
  `-p/--n-prompt`, `-n/--n-gen`, `-ngl/--n-gpu-layers`,
  `-fa/--flash-attn [on|off|auto]`, `-ctk/--cache-type-k`, `-ctv/--cache-type-v`,
  `-o <csv|json|jsonl|md|sql>`, `--no-warmup`.
- **common/arg.cpp**: `-c/--ctx-size`, `-ctk/-ctv`, `-fa`, `-ngl/--gpu-layers/--n-gpu-layers`,
  `--kl-divergence`, `--kl-divergence-base`.

## Baseline runner (Phase 0 step 4)
`scripts/baseline_bench.ps1` / `scripts/baseline_bench.sh` wrap `llama-bench`
with 3 repeats and store JSON. See that file's header for exact usage.

## Local build prerequisites (dev machine, checked 2026-10-11)
| Tool | State |
|---|---|
| git 2.51.0 | present |
| cmake | **MISSING** — install CMake ≥ 3.21 |
| ninja | present (pip) |
| CUDA toolkit (nvcc) | **MISSING** — `CUDA_PATH` unset; required for `build_cuda.*` |
| Vulkan SDK | runtime only (`vulkaninfo` present), `VULKAN_SDK` unset — SDK needed for `build_vulkan.*` shader-gen |
| MSVC (VS 2019 Build Tools) | installed (not on PATH until a Developer prompt) |

Until `cmake` + the CUDA Toolkit are installed, `build_cuda.*` cannot run. See
`docs/memory.md` for the full environment fingerprint and install commands.
