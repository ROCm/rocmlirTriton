<#
.SYNOPSIS
    Windows companion to scripts/build-llvm.sh.

.DESCRIPTION
    Bootstraps the LLVM/MLIR/LLD build that rocmlirTriton depends on, for
    Windows hosts using the HIP SDK + clang-cl + LLD. Unlike the Linux
    counterpart (which delegates to Triton's bash build-llvm-project.sh and
    sed-patches it in place), this script invokes CMake directly so there is
    no hard dependency on Git Bash or GNU sed.

    Steps performed, in order:
      1. git submodule update --init --recursive  (brings in external/triton)
      2. Apply any triton-patches/*.patch, normalizing CRLF -> LF first
         (Windows git core.autocrlf=true otherwise breaks `git apply`)
      3. Clone llvm/llvm-project into external/triton/llvm-project and reset
         to the commit in external/triton/cmake/llvm-hash.txt
      4. CMake-configure LLVM/MLIR/LLD with clang-cl + LLD
      5. ninja the LLVM build

.PARAMETER BuildType
    CMake build type (default: Release). RelWithDebInfo is a reasonable
    alternative; Debug is untested and very large on Windows.

.PARAMETER HipPath
    Path to the HIP SDK install. Defaults to $env:HIP_PATH, then $env:ROCM_PATH,
    then C:\opt\rocm.

.PARAMETER Chipset
    AMDGPU chipset used to gate MLIR_ENABLE_ROCM_RUNNER's integration-test
    lookup (there is no rocm_agent_enumerator.exe in the Windows HIP SDK).
    Defaults to gfx1201.

.PARAMETER Clean
    If set, wipe external/triton/llvm-project before cloning.

.NOTES
    This script MUST be run from an x64 Native Tools Command Prompt for
    VS 2022 (or an equivalent vcvars64.bat-activated shell), so that the
    MSVC toolchain headers/libraries (INCLUDE, LIB) are on PATH for
    clang-cl and lld-link.
#>

[CmdletBinding()]
param(
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug', 'MinSizeRel')]
    [string]$BuildType = 'Release',

    [string]$HipPath,

    [string]$Chipset = 'gfx1201',

    [switch]$Clean,

    # Windows-only override of the Triton submodule.
    # rocmlirTriton's .gitmodules points at upstream triton-lang/triton so
    # Linux is unaffected. On Windows we redirect the *local* clone to the
    # windows-enablement fork; .gitmodules and the recorded gitlink are not
    # modified.
    [string]$TritonRemote = 'https://github.com/triton-lang/triton-windows.git',
    # Empty -> read pinned SHA from triton-windows-hash.txt (see below).
    # Pass -TritonRef <sha> or -TritonRef main-windows to override.
    [string]$TritonRef    = ''
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
try {
    $RepoRoot = (git -C $ScriptDir rev-parse --show-toplevel 2>$null).Trim()
    if ([string]::IsNullOrEmpty($RepoRoot)) { throw 'git rev-parse failed' }
} catch {
    $RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
}
$RepoRoot = $RepoRoot -replace '\\', '/'
Write-Host "=== rocmlirTriton LLVM build wrapper (Windows) ===" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"

# Pinned triton-windows commit (analogous to external/triton/cmake/llvm-hash.txt).
# Bump by editing triton-windows-hash.txt and verifying the build.
if (-not $TritonRef) {
    $TritonHashFile = "$RepoRoot/triton-windows-hash.txt"
    if (-not (Test-Path $TritonHashFile)) {
        throw "Pinned triton-windows hash not found: $TritonHashFile"
    }
    $TritonRef = (Get-Content $TritonHashFile -TotalCount 1).Trim()
}
Write-Host "Triton ref: $TritonRef"

# ---------------------------------------------------------------------------
# Resolve HIP SDK
# ---------------------------------------------------------------------------
if (-not $HipPath) {
    if ($env:HIP_PATH)       { $HipPath = $env:HIP_PATH }
    elseif ($env:ROCM_PATH)  { $HipPath = $env:ROCM_PATH }
    else                     { $HipPath = 'C:\opt\rocm' }
}
$HipPath = $HipPath -replace '\\', '/'
if (-not (Test-Path "$HipPath/bin/clang-cl.exe")) {
    throw "HIP SDK clang-cl.exe not found at $HipPath/bin/clang-cl.exe. " +
          "Set HIP_PATH, ROCM_PATH, or pass -HipPath."
}
Write-Host "HIP SDK: $HipPath"

# ---------------------------------------------------------------------------
# Sanity-check MSVC dev env (INCLUDE/LIB must be set)
# ---------------------------------------------------------------------------
if (-not $env:INCLUDE -or -not $env:LIB) {
    throw 'MSVC environment not active. Run this from an "x64 Native Tools ' +
          'Command Prompt for VS 2022" or first call vcvars64.bat.'
}

$TritonDir  = "$RepoRoot/external/triton"
$PatchesDir = "$RepoRoot/triton-patches"

# ---------------------------------------------------------------------------
# Step 1: bring in external/triton (Windows-only fork override).
#
# rocmlirTriton's .gitmodules pins upstream triton-lang/triton, which is the
# Linux baseline. On Windows the upstream tree does not currently build
# cleanly, so we redirect the *local* clone of external/triton to the
# windows-enablement fork triton-lang/triton-windows at branch main-windows.
#
# This override is purely local to the clone -- it does NOT modify
# .gitmodules and does NOT modify the parent repo's recorded gitlink, so
# Linux clones using build-llvm.sh continue to get upstream triton.
# `git submodule status` on Windows will report external/triton as "+<sha>"
# (modified) by design.
# ---------------------------------------------------------------------------
Write-Host "--- Bringing in external/triton ($TritonRef from fork) ---" `
    -ForegroundColor Cyan
Write-Host "  remote: $TritonRemote"
Write-Host "  ref:    $TritonRef"

if (-not (Test-Path "$TritonDir/.git")) {
    Write-Host "  external/triton/.git absent -- direct-cloning fork" `
        -ForegroundColor Yellow
    if (Test-Path $TritonDir) { Remove-Item -Recurse -Force $TritonDir }
    git clone --filter=blob:none $TritonRemote $TritonDir
    if ($LASTEXITCODE -ne 0) { throw "git clone fork failed ($LASTEXITCODE)" }
} else {
    git -C $TritonDir remote set-url origin $TritonRemote
    if ($LASTEXITCODE -ne 0) { throw "remote set-url failed ($LASTEXITCODE)" }
}
git -C $TritonDir fetch --depth 1 origin $TritonRef
if ($LASTEXITCODE -ne 0) { throw "git fetch $TritonRef failed ($LASTEXITCODE)" }
git -C $TritonDir reset --hard $TritonRef
if ($LASTEXITCODE -ne 0) { throw "git reset $TritonRef failed ($LASTEXITCODE)" }

git -C $TritonDir submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "nested submodule update failed ($LASTEXITCODE)" }

# ---------------------------------------------------------------------------
# Step 2: Apply triton patches.
#
# Originally we applied every triton-patches/*.patch on top of the gitlinked
# Triton tree. An audit against triton-windows@main-windows showed that on
# the Windows path:
#   - patch1:         logically merged with context drift in the fork.
#   - patch2, patch3: cleanly merged into the fork.
#   - patch4, patch5: apply cleanly but proven not to be compile-time critical.
#   - patch6:         hard compile blocker on clang-cl (overload ambiguity in
#                     PartitionLoops.cpp). MUST be applied.
#
# We therefore disable the generic patch loop on Windows (commented out
# below for easy reversion) and apply only patch6. The other .patch files
# remain on disk so the Linux build path (scripts/build-llvm.sh) keeps
# applying them against upstream Triton.
# ---------------------------------------------------------------------------

# --- Generic patch loop (disabled on Windows; uncomment to re-enable) ------
# if (Test-Path $PatchesDir) {
#     $patches = Get-ChildItem -Path $PatchesDir -Filter '*.patch' -File |
#                Sort-Object Name
#     if ($patches.Count -gt 0) {
#         Write-Host "--- Applying triton patches ---" -ForegroundColor Cyan
#         $TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "rocmlir-patches-$PID"
#         New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
#         # Lower ErrorActionPreference for the patch loop. Under 'Stop',
#         # PowerShell wraps any native-command stderr in a NativeCommandError
#         # *at invocation time* -- the `2>$null` / `2>&1` redirections happen
#         # too late to suppress it. We manage failures ourselves via
#         # $LASTEXITCODE inside the loop.
#         $savedPref = $ErrorActionPreference
#         $ErrorActionPreference = 'Continue'
#         try {
#             foreach ($patch in $patches) {
#                 # Rewrite patch to LF before handing to `git apply`.
#                 $bytes = [System.IO.File]::ReadAllBytes($patch.FullName)
#                 $text  = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "`r`n", "`n"
#                 $lfPath = Join-Path $TmpDir $patch.Name
#                 [System.IO.File]::WriteAllBytes(
#                     $lfPath, [System.Text.Encoding]::UTF8.GetBytes($text))
#
#                 & git -C $TritonDir apply --check $lfPath 2>$null | Out-Null
#                 if ($LASTEXITCODE -eq 0) {
#                     Write-Host "Applying:  $($patch.Name)"
#                     & git -C $TritonDir apply $lfPath 2>&1 | Write-Host
#                     if ($LASTEXITCODE -ne 0) {
#                         throw "git apply failed for $($patch.Name)"
#                     }
#                     continue
#                 }
#                 & git -C $TritonDir apply --check --reverse $lfPath 2>$null | Out-Null
#                 if ($LASTEXITCODE -eq 0) {
#                     Write-Host "Skipping:  $($patch.Name) (already applied)"
#                 } else {
#                     # The Windows build targets triton-windows@main-windows
#                     # rather than the gitlink upstream triton recorded in
#                     # .gitmodules. Some patches authored against the older
#                     # upstream pin no longer apply forward or reverse here
#                     # because their logical change is present but the
#                     # surrounding context drifted. Warn loudly but continue;
#                     # the per-patch context is documented in
#                     # triton-patches/*.patch headers.
#                     Write-Warning ("Patch does not apply cleanly: " +
#                         "$($patch.Name) -- assuming the change is " +
#                         "present in this Triton tree (Windows-only override).")
#                 }
#             }
#         } finally {
#             $ErrorActionPreference = $savedPref
#             Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
#         }
#     }
# }

# --- Apply only patch6 (the hard compile blocker on clang-cl) -------------
$Patch6 = Join-Path $PatchesDir 'patch6.patch'
if (Test-Path $Patch6) {
    Write-Host "--- Applying triton patch6 ---" -ForegroundColor Cyan
    $TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "rocmlir-patches-$PID"
    New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
    # Lower ErrorActionPreference for the patch block. Under 'Stop',
    # PowerShell wraps any native-command stderr in a NativeCommandError
    # *at invocation time* -- the `2>$null` / `2>&1` redirections happen
    # too late to suppress it. We manage failures ourselves via $LASTEXITCODE.
    $savedPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Rewrite patch to LF before handing to `git apply`.
        $bytes  = [System.IO.File]::ReadAllBytes($Patch6)
        $text   = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "`r`n", "`n"
        $lfPath = Join-Path $TmpDir 'patch6.patch'
        [System.IO.File]::WriteAllBytes(
            $lfPath, [System.Text.Encoding]::UTF8.GetBytes($text))

        & git -C $TritonDir apply --check $lfPath 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Applying:  patch6.patch"
            & git -C $TritonDir apply $lfPath 2>&1 | Write-Host
            if ($LASTEXITCODE -ne 0) {
                throw "git apply failed for patch6.patch"
            }
        } else {
            & git -C $TritonDir apply --check --reverse $lfPath 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Skipping:  patch6.patch (already applied)"
            } else {
                Write-Warning ("patch6.patch does not apply cleanly -- " +
                    "assuming the change is present in this Triton tree " +
                    "(Windows-only override).")
            }
        }
    } finally {
        $ErrorActionPreference = $savedPref
        Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# Step 3: Fetch LLVM at the pinned commit
# ---------------------------------------------------------------------------
$LlvmSrc    = "$TritonDir/llvm-project"
$LlvmBuild  = "$LlvmSrc/build"
$LlvmHashFile = "$TritonDir/cmake/llvm-hash.txt"
if (-not (Test-Path $LlvmHashFile)) {
    throw "$LlvmHashFile not found (did the Triton submodule init?)."
}
$LlvmHash = (Get-Content $LlvmHashFile -Raw).Trim()

if ($Clean -and (Test-Path $LlvmSrc)) {
    Write-Host "Removing $LlvmSrc (--Clean)" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $LlvmSrc
}

if (-not (Test-Path $LlvmSrc)) {
    Write-Host "--- Cloning llvm-project ---" -ForegroundColor Cyan
    git clone --filter=blob:none https://github.com/llvm/llvm-project $LlvmSrc
    if ($LASTEXITCODE -ne 0) { throw "git clone failed ($LASTEXITCODE)" }
}

Write-Host "--- Resetting llvm-project to $LlvmHash ---" -ForegroundColor Cyan
git -C $LlvmSrc fetch --depth 1 origin $LlvmHash
if ($LASTEXITCODE -ne 0) { throw "git fetch failed ($LASTEXITCODE)" }
git -C $LlvmSrc reset --hard $LlvmHash
if ($LASTEXITCODE -ne 0) { throw "git reset failed ($LASTEXITCODE)" }

# If a previous build was configured against a different LLVM commit, wipe
# its build dir to avoid stale-artifact / incremental-rebuild bugs (object
# files compiled against the old MLIR headers, leftover lit configs, etc.).
$LlvmHashSentinel = "$LlvmBuild/.rocmlir-llvm-hash"
if ((Test-Path $LlvmBuild) -and (Test-Path $LlvmHashSentinel)) {
    $previousHash = (Get-Content -Raw $LlvmHashSentinel).Trim()
    if ($previousHash -ne $LlvmHash) {
        Write-Host ("LLVM hash changed ($previousHash -> $LlvmHash); " +
                    "wiping $LlvmBuild") -ForegroundColor Yellow
        Remove-Item -Recurse -Force $LlvmBuild
    }
}

# ---------------------------------------------------------------------------
# Step 4: Configure LLVM/MLIR/LLD
# ---------------------------------------------------------------------------
Write-Host "--- Configuring LLVM/MLIR/LLD ---" -ForegroundColor Cyan
$cmakeArgs = @(
    '-G', 'Ninja',
    '-S', "$LlvmSrc/llvm",
    '-B', $LlvmBuild,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_C_COMPILER=$HipPath/bin/clang-cl.exe",
    "-DCMAKE_CXX_COMPILER=$HipPath/bin/clang-cl.exe",
    '-DCMAKE_LINKER_TYPE=LLD',
    '-DLLVM_ENABLE_ASSERTIONS=ON',
    '-DLLVM_CCACHE_BUILD=OFF',
    '-DBUILD_SHARED_LIBS=OFF',
    '-DLLVM_OPTIMIZED_TABLEGEN=ON',
    '-DMLIR_ENABLE_BINDINGS_PYTHON=OFF',
    '-DLLVM_ENABLE_ZSTD=OFF',
    '-DLLVM_TARGETS_TO_BUILD=Native;NVPTX;AMDGPU',
    '-DLLVM_ENABLE_PROJECTS=mlir;lld',
    '-DMLIR_ENABLE_ROCM_RUNNER=ON',
    "-DROCM_PATH=$HipPath",
    "-DROCM_TEST_CHIPSET=$Chipset",
    '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
)
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

# ---------------------------------------------------------------------------
# Step 5: Build
# ---------------------------------------------------------------------------
Write-Host "--- ninja -C $LlvmBuild ---" -ForegroundColor Cyan
& ninja -C $LlvmBuild
if ($LASTEXITCODE -ne 0) { throw "ninja build failed ($LASTEXITCODE)" }

# Record the hash this build dir is configured for, so a future Triton bump
# can detect the change and trigger a wipe (see Step 3 above).
Set-Content -Path $LlvmHashSentinel -Value $LlvmHash -NoNewline

Write-Host "=== LLVM build complete ===" -ForegroundColor Green
Write-Host "MLIR CMake config: $LlvmBuild/lib/cmake/mlir/MLIRConfig.cmake"
