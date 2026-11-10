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

# --- Load the MSVC toolchain ---
# cl.exe/nmake.exe live in the VS Build Tools but are NOT on PATH in a normal
# shell. We must run inside the environment set up by vcvars64.bat. The cleanest
# way from PowerShell is to re-exec this whole script inside cmd /c after
# vcvars64, so cmake sees a fully populated cl/INCLUDE/LIB environment.
if ($env:LLAMA_BUILD_IN_VCVARS -ne "1") {
    $vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        # Fall back to vswhere so this also works for VS 2022 / other editions.
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
    # Re-invoke under cmd so vcvars populates PATH/INCLUDE/LIB, then run this script again.
    cmd /c "`"$vcvars`" >nul 2>&1 && powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit $LASTEXITCODE
}

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
# MSVC + CMake leaves the C/C++ flag variables empty unless the toolchain sets
# them, so this build was missing three things that matter:
#
#   /O2       - no optimisation at all.
#   /DNDEBUG  - assert() stays live, so a failed assertion aborts the process
#               with 0xc0000409 in ucrtbase.dll.
#   /EHsc     - C++ exceptions have NO unwind semantics without it. When an
#               exception propagates the runtime calls abort(), which is
#               EXACTLY the crash observed killing llama-server mid-eval
#               (reproduced deterministically on task sh-10: the process died
#               at n_gen = 1005 with no message). The build log is full of
#               "warning C4530: C++ exception handler used, but unwind
#               semantics are not enabled. Specify /EHsc".
#
# Define all three explicitly.
if ($Config -eq "Release") {
    $cmakeArgs += @(
        "-DCMAKE_C_FLAGS_RELEASE=/O2 /DNDEBUG",
        "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /DNDEBUG /EHsc"
    )
}
if ($Generator -eq "Ninja") {
    $cmakeArgs += @("-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl")
}

Write-Host "`n>> cmake $($cmakeArgs -join ' ')" -ForegroundColor Yellow
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }

$buildArgs = @("--build", $buildDir, "--config", $Config, "--target", "llama-bench", "llama-server", "llama-perplexity", "llama-quantize", "llama-imatrix")

Write-Host "`n>> cmake $($buildArgs -join ' ')" -ForegroundColor Yellow
cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }

$bin = Join-Path $buildDir "bin"
Write-Host "`n== Build complete ==" -ForegroundColor Green
Write-Host "Binaries in: $bin"
Write-Host "Verify with : & '$(Join-Path $bin 'llama-bench.exe')' --version"
