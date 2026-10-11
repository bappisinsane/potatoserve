#requires -Version 5.1
<#
.SYNOPSIS
    Build llama.cpp with the Vulkan backend (secondary target for PotatoServe).

.DESCRIPTION
    Pinned build: third_party/llama.cpp @ v0.6.0.
    Vulkan runs on the same GTX 1660 Ti and is the portable path for any
    AMD/Intel GPU (project-spec section 1: "Secondary | any AMD/Intel GPU
    reachable via Vulkan").

    Flag names verified against the pinned v0.6.0 source:
      - Top-level CMakeLists.txt: GGML_VULKAN (default OFF).
      - ggml-vulkan-shaders-gen compiles at configure time and needs the Vulkan
        SDK (glslc) on PATH, OR it falls back to a CPU build of the generator.

.PARAMETER Generator
    CMake generator. Default "Ninja".

.EXAMPLE
    .\scripts\build_vulkan.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet("Ninja", "Visual Studio 17 2022", "Visual Studio 16 2019", "NMake Makefiles")]
    [string]$Generator = "Ninja",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$llamaDir = (Resolve-Path (Join-Path $PSScriptRoot "..\third_party\llama.cpp")).Path
$buildDir = Join-Path $llamaDir "build-vulkan"

Write-Host "== PotatoServe Vulkan build ==" -ForegroundColor Cyan
Write-Host "llama.cpp : $llamaDir"
Write-Host "pinned    : $(git -C $llamaDir describe --tags) @ $(git -C $llamaDir rev-parse HEAD)"
Write-Host "generator : $Generator"

# --- Preconditions ---
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake not found on PATH. Install CMake, then retry."
}
# A Vulkan *runtime* is not enough; shader compilation needs the SDK's glslc.
if (-not $env:VULKAN_SDK) {
    Write-Warning "VULKAN_SDK is not set. Install the LunarG Vulkan SDK so glslc (shader compiler) is available; otherwise the Vulkan shader-gen step may fail."
} else {
    Write-Host "VULKAN_SDK: $env:VULKAN_SDK"
}

$cmakeArgs = @(
    "-S", $llamaDir,
    "-B", $buildDir,
    "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DGGML_VULKAN=ON",
    "-DGGML_NATIVE=OFF",
    "-DLLAMA_CURL=OFF"
)
if ($Generator -eq "Ninja") {
    $cmakeArgs += @("-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl")
}

Write-Host "`n>> cmake $($cmakeArgs -join ' ')" -ForegroundColor Yellow
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }

$buildArgs = @("--build", $buildDir, "--config", $Config, "--target", "llama-bench", "llama-server", "llama-perplexity")

Write-Host "`n>> cmake $($buildArgs -join ' ')" -ForegroundColor Yellow
cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }

$bin = Join-Path $buildDir "bin"
Write-Host "`n== Build complete ==" -ForegroundColor Green
Write-Host "Binaries in: $bin"
Write-Host "Verify with : & '$(Join-Path $bin 'llama-bench.exe')' --version"
