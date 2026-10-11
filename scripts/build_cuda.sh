#!/usr/bin/env bash
# Build llama.cpp with the CUDA backend for the PotatoServe dev GPU.
# Pinned build: third_party/llama.cpp @ v0.6.0.
# Dev GPU: NVIDIA GeForce GTX 1660 Ti (TU116, Turing) -> compute capability 7.5.
#
# Flag names verified against pinned v0.6.0 source (not memory):
#   - ggml-cuda/CMakeLists.txt uses standard CMAKE_CUDA_ARCHITECTURES; 75 is valid.
#   - Top-level CMakeLists.txt: GGML_NATIVE, GGML_CUDA, GGML_CUDA_FA_ALL_QUANTS.
set -euo pipefail

CUDA_ARCH="${1:-75}"          # override: ./build_cuda.sh <arch>
GENERATOR="${GENERATOR:-Ninja}"
CONFIG="${CONFIG:-Release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/third_party/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build-cuda"

echo "== PotatoServe CUDA build =="
echo "llama.cpp : $LLAMA_DIR"
echo "pinned    : $(git -C "$LLAMA_DIR" describe --tags) @ $(git -C "$LLAMA_DIR" rev-parse HEAD)"
echo "CUDA arch : $CUDA_ARCH (GTX 1660 Ti / Turing)"
echo "generator : $GENERATOR"

for tool in cmake nvcc; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool' not found on PATH. Install the CUDA Toolkit + CMake, then retry." >&2
    exit 1
  fi
done

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE="$CONFIG" \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=OFF

cmake --build "$BUILD_DIR" --target llama-bench llama-server llama-perplexity

echo "== Build complete =="
echo "Binaries in: $BUILD_DIR/bin"
echo "Verify with: $BUILD_DIR/bin/llama-bench --version"
