#requires -Version 5.1
<#
.SYNOPSIS
    Build llama.cpp with the CUDA backend for the PotatoServe dev GPU.

.DESCRIPTION
    Pinned build: third_party/llama.cpp @ v0.6.0.
    Dev GPU: NVIDIA GeForce GTX 1660 Ti (TU116, Turing) -> CUDA compute
    capability 7.5. Set explicitly via CMAKE_CUDA_ARCHITECTURES so the build is
    reproducible and does not depend on which GPU is attached at build time.

    Flag names below are verified against the pinned v0.6.0 source, not memory:
      - ggml-cuda/CMakeLists.txt uses the standard CMAKE_CUDA_ARCHITECTURES
        and accepts 75 (Turing) as a target.
      - Top-level CMakeLists.txt: GGML_NATIVE, GGML_CUDA, GGML_CUDA_FA_ALL_QUANTS.

.PARAMETER CudaArch
    CUDA compute capability to compile for. Default 75 (GTX 1660 Ti, Turing).
    Change this if you build for a different GPU.

.PARAMETER Generator
    CMake generator. Default "Ninja" (ninja is on PATH). Use "Visual Studio 17
    2022" (or your installed VS) if you prefer MSBuild.

.EXAMPLE
    .\scripts\build_cuda.ps1
    .\scripts\build_cuda.ps1 -CudaArch 75 -Generator Ninja
#>
[CmdletBinding()]
param(
    [string]$CudaArch = "75",
    [ValidateSet("Ninja", "Visual Studio 17 2022", "Visual Studio 16 2019", "NMake Makefiles")]
    [string]$Generator = "Ninja",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# --- Load the MSVC toolchain ---
# cl.exe/nmake.exe are not on PATH in a normal shell. Re-exec this script inside
# the vcvars64 environment so cmake sees a fully populated cl/INCLUDE/LIB setup.
if ($env:LLAMA_BUILD_IN_VCVARS -ne "1") {
    $vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
        }
    }
    if (-not (Test-Path $vcvars)) {
        throw "Could not find vcvars64.bat. Install the 'Desktop development with C++' workload (VS Build Tools), then retry."
    }
    Write-Host "Re-launching inside MSVC environment via vcvars64..." -ForegroundColor DarkCyan
    $env:LLAMA_BUILD_IN_VCVARS = "1"
    cmd /c "`"$vcvars`" >nul 2>&1 && powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit $LASTEXITCODE
}

$llamaDir = Join-Path $PSScriptRoot "..\third_party\llama.cpp"
$llamaDir = (Resolve-Path $llamaDir).Path
$buildDir = Join-Path $llamaDir "build-cuda"

Write-Host "== PotatoServe CUDA build ==" -ForegroundColor Cyan
Write-Host "llama.cpp : $llamaDir"
Write-Host "pinned    : $(git -C $llamaDir describe --tags) @ $(git -C $llamaDir rev-parse HEAD)"
Write-Host "CUDA arch : $CudaArch (GTX 1660 Ti / Turing)"
Write-Host "generator : $Generator"

# --- Preconditions (do not hide a missing toolchain) ---
foreach ($tool in @("cmake", "nvcc")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required tool '$tool' not found on PATH. Install the CUDA Toolkit (nvcc) and CMake, then retry. See docs/memory.md environment notes."
    }
}

$nvccVer = (nvcc --version) -join "`n"
Write-Host ("nvcc      : " + (($nvccVer -split "`n") | Select-Object -Last 1).Trim())

# --- Configure ---
# GGML_CUDA=ON, GGML_NATIVE=OFF (explicit arch is more reproducible than native).
# GGML_CUDA_FA_ALL_QUANTS=OFF keeps the build small/fast; enable only for the
# flash-attention quant sweep in later phases.
$cmakeArgs = @(
    "-S", $llamaDir,
    "-B", $buildDir,
    "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DGGML_CUDA=ON",
    "-DGGML_NATIVE=OFF",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DLLAMA_CURL=OFF"
)
if ($Generator -eq "Ninja") {
    $cmakeArgs += @("-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl")
}
# See build_vulkan.ps1: without explicit Release flags the MSVC build has no
# /DNDEBUG (asserts stay live) and no /EHsc (C++ exceptions abort the
# process). Both were observed crashing llama-server with 0xc0000409.
if ($Config -eq "Release") {
    $cmakeArgs += @(
        "-DCMAKE_C_FLAGS_RELEASE=/O2 /DNDEBUG",
        "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /DNDEBUG /EHsc"
    )
}

Write-Host "`n>> cmake $($cmakeArgs -join ' ')" -ForegroundColor Yellow
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }

# --- Build (targets we actually use in Phase 0-2) ---
$buildArgs = @("--build", $buildDir, "--config", $Config, "--target", "llama-bench", "llama-server", "llama-perplexity", "llama-quantize", "llama-imatrix")
if ($Generator -eq "Ninja") { $buildArgs = @("--build", $buildDir, "--target", "llama-bench", "llama-server", "llama-perplexity", "llama-quantize", "llama-imatrix") }

Write-Host "`n>> cmake $($buildArgs -join ' ')" -ForegroundColor Yellow
cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }

$bin = Join-Path $buildDir "bin"
Write-Host "`n== Build complete ==" -ForegroundColor Green
Write-Host "Binaries in: $bin"
Write-Host "Verify with : & '$(Join-Path $bin 'llama-bench.exe')' --version"
