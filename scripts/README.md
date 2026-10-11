# scripts/ — build & helper scripts

Home for the llama.cpp build scripts added in Phase 0 step 2:

- `build_cuda.sh` / `build_cuda.ps1` — CUDA build (SM version matched to the dev GPU)
- `build_vulkan.sh` / `build_vulkan.ps1` — Vulkan build

Build flags are verified against the **pinned** llama.cpp commit, never from memory —
flag names change between releases.
