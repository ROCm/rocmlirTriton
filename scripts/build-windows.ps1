<#
.SYNOPSIS
    Windows build entry point for rocmlirTriton (in-tree LLVM/MLIR + Triton +
    rocMLIR, using a ROCm clang-cl / lld-link toolchain).

.DESCRIPTION
    A single CMake configure + build; the vendored trees compile in-tree via
    add_subdirectory (see cmake/triton.cmake). Needs CMake >= 3.20, Ninja, Python 3.

    Supports either ROCm distribution, selected via -RocmPath (forwarded as
    -DROCM_PATH); the toolchain layout is auto-detected:
      * AMD HIP SDK   - clang-cl / lld-link under <root>/bin
      * TheRock ROCm  - clang-cl / lld-link under <root>/lib/llvm/bin
    The root resolves from -RocmPath, then ROCM_PATH, HIP_PATH, then C:/opt/rocm.

    Run this from an "x64 Native Tools Command Prompt for VS": it provides the
    MASM assembler (for LLVM's BLAKE3 sources) and mt.exe (the manifest tool
    CMake's clang-cl toolchain auto-detects).

.EXAMPLE
    # HIP SDK (default C:/opt/rocm)
    pwsh scripts/build-windows.ps1 -ConfigureOnly -GpuTargets gfx1201

.EXAMPLE
    # TheRock ROCm, into a separate build dir
    pwsh scripts/build-windows.ps1 -BuildDir build-therock -RocmPath C:/opt/therock-rocm -GpuTargets gfx1201
#>
[CmdletBinding()]
param(
    # Build directory, relative to the repo root unless absolute.
    [string]$BuildDir = "build",

    # HIP SDK / ROCm root. Falls back to ROCM_PATH, HIP_PATH, then C:/opt/rocm.
    [string]$RocmPath = $(
        if ($env:ROCM_PATH) { $env:ROCM_PATH }
        elseif ($env:HIP_PATH) { $env:HIP_PATH }
        else { "C:/opt/rocm" }
    ),

    # GPU arch(s), semicolon-separated. Defaults to the full RDNA3/RDNA4 set;
    # override for a faster single-arch dev build (e.g. -GpuTargets gfx1201).
    [string]$GpuTargets = "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1170;gfx1171;gfx1172;gfx1200;gfx1201",

    # Integration-test chipset; auto-detected, with single GpuTargets as fallback.
    [string]$RocmTestChipset = "",

    [ValidateSet("Release", "RelWithDebInfo", "Debug", "MinSizeRel")]
    [string]$BuildType = "RelWithDebInfo",

    # Parallel build jobs. 0 -> all logical processors.
    [int]$Jobs = 0,

    # Extra -D options passed verbatim to the configure step.
    [string[]]$CMakeArgs = @(),

    # Configure only; skip the build step.
    [switch]$ConfigureOnly,

    # Specific build targets.
    [string[]]$Targets = @("check-rocmlir-build-only")
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $repoRoot $BuildDir
}
# Forward slashes: CMake accepts them and avoids backslash-escaping surprises.
$RocmPath = $RocmPath -replace '\\', '/'

function Find-RocmTestChipset([string]$root) {
    foreach ($relativePath in @("bin/amdgpu-arch.exe", "lib/llvm/bin/amdgpu-arch.exe")) {
        $tool = Join-Path $root $relativePath
        if (-not (Test-Path $tool)) { continue }
        $output = & $tool 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
        $arch = $output | Where-Object { $_ -match '^gfx[0-9A-Za-z:+_-]+$' } |
            Select-Object -First 1
        if ($arch) { return $arch }
    }
    return $null
}

if ($Jobs -le 0) {
    $Jobs = [int]$env:NUMBER_OF_PROCESSORS
    if ($Jobs -le 0) { $Jobs = 1 }
}
if (-not $RocmTestChipset) {
    $RocmTestChipset = Find-RocmTestChipset $RocmPath
    if (-not $RocmTestChipset) {
        $gpuTargetList = @($GpuTargets -split ';' | Where-Object { $_ })
        if ($gpuTargetList.Count -ne 1) {
            throw "Could not detect the test chipset; pass -RocmTestChipset when building multiple GPU targets."
        }
        $RocmTestChipset = $gpuTargetList[0]
    }
}

# cmake writes warnings to stderr; under 'Stop' PowerShell would treat that as a
# terminating error, so relax it here (function-scoped) and let callers check
# $LASTEXITCODE for real failures.
function Invoke-CMake([string[]]$cmakeArgs) {
    Write-Host "`n> cmake $($cmakeArgs -join ' ')" -ForegroundColor DarkGray
    $ErrorActionPreference = "Continue"
    & cmake @cmakeArgs
}

Write-Host "rocmlirTriton Windows build -> $BuildDir  (ROCm: $RocmPath, $BuildType, -j $Jobs)" -ForegroundColor Cyan

$configureArgs = @(
    "-S", $repoRoot,
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DROCM_PATH=$RocmPath",
    "-DGPU_TARGETS=$GpuTargets",
    "-DROCM_TEST_CHIPSET=$RocmTestChipset",
    "-DBUILD_FAT_LIBROCKCOMPILER=ON",
    "-DLLD_BUILD_TOOLS=ON",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
) + $CMakeArgs

Invoke-CMake $configureArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)." }

if ($ConfigureOnly) {
    Write-Host "Configure complete (configure-only)." -ForegroundColor Green
    return
}

$buildArgs = @("--build", $BuildDir, "-j", $Jobs)
if ($Targets.Count -gt 0) { $buildArgs += @("--target") + $Targets }

Invoke-CMake $buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)." }

Write-Host "Build complete." -ForegroundColor Green
