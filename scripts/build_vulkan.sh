#!/usr/bin/env bash
# Build llama.cpp with the Vulkan backend (secondary target, portable to AMD/Intel).
# Pinned build: third_party/llama.cpp @ v0.6.0.
# Flag names verified against pinned v0.6.0 source: top-level CMakeLists.txt GGML_VULKAN.
set -euo pipefail

GENERATOR="${GENERATOR:-Ninja}"
CONFIG="${CONFIG:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/third_party/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build-vulkan"

echo "== PotatoServe Vulkan build =="
echo "llama.cpp : $LLAMA_DIR"
echo "pinned    : $(git -C "$LLAMA_DIR" describe --tags) @ $(git -C "$LLAMA_DIR" rev-parse HEAD)"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake not found on PATH." >&2
  exit 1
fi
if [ -z "${VULKAN_SDK:-}" ]; then
  echo "WARNING: VULKAN_SDK not set. Install the LunarG Vulkan SDK so glslc is available for shader compilation." >&2
else
  echo "VULKAN_SDK: $VULKAN_SDK"
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE="$CONFIG" \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF

cmake --build "$BUILD_DIR" --target llama-bench llama-server llama-perplexity

echo "== Build complete =="
echo "Binaries in: $BUILD_DIR/bin"
echo "Verify with: $BUILD_DIR/bin/llama-bench --version"
