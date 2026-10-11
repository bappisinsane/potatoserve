#requires -Version 5.1
<#
.SYNOPSIS
    Baseline benchmark runner (Phase 0 step 4): run llama-bench with 3 repeats
    and store a project-spec section-8 result record under results/.

.DESCRIPTION
    This is a thin wrapper around `python -m potato.cli bench`, which:
      * invokes llama-bench with verified v0.6.0 flags
        (-r/--repetitions, -p/--n-prompt, -n/--n-gen, -o json, -ngl, -fa),
      * defaults to 3 repeats (llama-bench's own default is 5, so we set it
        explicitly to satisfy the "3 repeats" requirement),
      * parses the JSON and writes results/<run_id>.json in the section-8 schema.

    No numbers are invented: if llama-bench is not built or the model is
    missing, the command fails with a clear message and writes nothing.

.PARAMETER Model
    Path to a .gguf model (required).

.PARAMETER Repeats
    Number of repeats. Default 3 (spec minimum). Must be >= 3 for a valid baseline.

.PARAMETER ResultsDir
    Directory for result records. Default "results".

.EXAMPLE
    .\scripts\baseline_bench.ps1 -Model ..\models\qwen3-0.6b-q4_k_m.gguf
.EXAMPLE
    .\scripts\baseline_bench.ps1 -Model models\model.gguf -Repeats 3 -Ngl 99 -FlashAttn on
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [int]$Repeats = 3,
    [int]$Pp = 512,
    [int]$Tg = 128,
    [int]$Ngl = 99,
    [string]$FlashAttn = "on",
    [string]$ResultsDir = "results",
    [string]$Binary = "",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$cliArgs = @("bench", "--model", $Model, "--repeats", "$Repeats", "--pp", "$Pp",
             "--tg", "$Tg", "--ngl", "$Ngl", "--flash-attn", $FlashAttn,
             "--results-dir", $ResultsDir)
if ($Binary)    { $cliArgs += @("--binary", $Binary) }
if ($RunId)     { $cliArgs += @("--run-id", $RunId) }

Write-Host ">> python -m potato.cli $($cliArgs -join ' ')" -ForegroundColor Yellow
python -m potato.cli @cliArgs
if ($LASTEXITCODE -ne 0) {
    throw "baseline_bench failed (exit $LASTEXITCODE). Is llama.cpp built and the model path correct?"
}
Write-Host "Baseline complete. Records in: $ResultsDir" -ForegroundColor Green
