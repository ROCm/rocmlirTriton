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
      1. Download external/triton via the GitHub CLI.
      2. Apply any triton-patches/*.patch, normalizing CRLF -> LF first.
      3. Download llvm/llvm-project into external/triton/llvm-project at the
         commit in external/triton/cmake/llvm-hash.txt.
      4. CMake-configure LLVM/MLIR/LLD with clang-cl + LLD
      5. ninja the LLVM build

.PARAMETER BuildType
    CMake build type (default: Release). RelWithDebInfo is a reasonable
    alternative; Debug is untested and very large on Windows.

.PARAMETER HipPath
    Path to the HIP SDK install. Defaults to $env:HIP_PATH, then $env:ROCM_PATH,
    then C:\opt\rocm.

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

    [switch]$Clean,

    # --- Triton source ----------------------------------------------------
    # If -TritonRemote is provided, the script switches to "alternative
    # Triton" mode: it clones $TritonRemote@$TritonRef into external/triton
    # and $LlvmRemote@$LlvmRef into external/triton/llvm-project. No
    # triton-patches/* or llvm-patches/* are applied -- alternative Triton
    # forks are expected to carry the equivalent changes in their source
    # directly.
    #
    # If -TritonRemote is empty (default), the script keeps the historical
    # windows-enablement behaviour: clones triton-lang/triton-windows at
    # the SHA pinned in triton-windows-hash.txt, applies triton-patches/*
    # (patch1, patch6) and llvm-patches/*.
    [string]$TritonRemote = '',
    [string]$TritonRef    = '',

    # --- LLVM source (only consulted in "alternative Triton" mode) --------
    # In default (windows-enablement) mode LLVM URL is hard-coded to
    # github.com/llvm/llvm-project and the commit is read from the Triton
    # tree's cmake/llvm-hash.txt, exactly as before.
    [string]$LlvmRemote = '',
    [string]$LlvmRef    = ''
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Helper: invoke a native command (gh, cmake, ninja) without letting
# PowerShell turn its stderr progress chatter (e.g. "Cloning into ...",
# "Updating files: 17% (...)") into a terminating NativeCommandError.
#
# Without this wrapper, $ErrorActionPreference='Stop' on Windows PowerShell 5.1
# aborts the script on the very first native stderr line, even though the
# command itself succeeds. PowerShell 7 has
# $PSNativeCommandUseErrorActionPreference for this; PS 5 does not.
#
# The caller is still responsible for checking $LASTEXITCODE -- this wrapper
# only fixes the stderr-as-error misclassification.
# ---------------------------------------------------------------------------
function Invoke-Native {
    [CmdletBinding()]
    param([Parameter(Mandatory)] [string]$Exe,
          [Parameter(ValueFromRemainingArguments)] [object[]]$Args)
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Args 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                Write-Host $_.Exception.Message
            } else {
                Write-Host $_
            }
        }
    } finally {
        $ErrorActionPreference = $saved
    }
}

function Invoke-GhText {
    [CmdletBinding()]
    param([Parameter(ValueFromRemainingArguments)] [object[]]$Args)

    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & gh @Args 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $saved
    }

    if ($exitCode -ne 0) {
        $message = ($output | Out-String).Trim()
        throw "gh $($Args -join ' ') failed ($exitCode): $message"
    }
    return @($output)
}

function ConvertTo-GithubRepoSlug {
    param([Parameter(Mandatory)] [string]$Remote)

    $trimmed = $Remote.Trim()
    if ($trimmed -match '^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$') {
        return "$($Matches[1])/$($Matches[2])"
    }
    if ($trimmed -match '^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$') {
        return "$($Matches[1])/$($Matches[2])"
    }
    if ($trimmed -match '^[^/\s]+/[^/\s]+$') {
        return ($trimmed -replace '\.git$', '')
    }

    throw "Only GitHub repositories are supported by the gh-based fetch path: $Remote"
}

function Get-GhAuthHeaders {
    $headers = @{}
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $tokenOutput = & gh auth token 2>$null
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $saved
    }

    if ($exitCode -eq 0) {
        $token = ($tokenOutput | Select-Object -First 1).Trim()
        if ($token) {
            $headers['Authorization'] = "Bearer $token"
        }
    }
    return $headers
}

function Clear-DirectoryExcept {
    param([Parameter(Mandatory)] [string]$Path,
          [string[]]$PreserveNames = @())

    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
        return
    }

    $preserve = @{}
    foreach ($name in $PreserveNames) {
        if ($name) {
            $preserve[$name.ToLowerInvariant()] = $true
        }
    }

    Get-ChildItem -LiteralPath $Path -Force | ForEach-Object {
        if (-not $preserve.ContainsKey($_.Name.ToLowerInvariant())) {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force
        }
    }
}

function Copy-DirectoryContents {
    param([Parameter(Mandatory)] [string]$Source,
          [Parameter(Mandatory)] [string]$Destination)

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName `
            -Destination (Join-Path $Destination $_.Name) `
            -Recurse -Force
    }
}

function Read-GitModules {
    param([Parameter(Mandatory)] [string]$Path)

    if (-not (Test-Path $Path)) { return @() }

    $entries = @()
    $current = $null
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*\[submodule\s+"(.+)"\]\s*$') {
            if ($current) { $entries += [pscustomobject]$current }
            $current = @{ Name = $Matches[1]; Path = $null; Url = $null }
        } elseif ($current -and $line -match '^\s*path\s*=\s*(.+?)\s*$') {
            $current.Path = $Matches[1]
        } elseif ($current -and $line -match '^\s*url\s*=\s*(.+?)\s*$') {
            $current.Url = $Matches[1]
        }
    }
    if ($current) { $entries += [pscustomobject]$current }
    return @($entries | Where-Object { $_.Path -and $_.Url })
}

function Install-GhRepositoryArchive {
    [CmdletBinding()]
    param([Parameter(Mandatory)] [string]$Remote,
          [Parameter(Mandatory)] [string]$Ref,
          [Parameter(Mandatory)] [string]$Destination,
          [string[]]$PreserveNames = @())

    $repoSlug = ConvertTo-GithubRepoSlug $Remote
    Write-Host "  gh repo: $repoSlug"
    Invoke-Native gh repo view $repoSlug --json nameWithOwner --jq .nameWithOwner
    if ($LASTEXITCODE -ne 0) { throw "gh repo view failed for $repoSlug ($LASTEXITCODE)" }

    # Download a GitHub archive instead of cloning so SHA pins do not require a
    # follow-up git fetch/checkout on machines where native git remotes fail.
    $tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) "rocmlir-gh-archive-$PID-$([guid]::NewGuid())"
    $zipPath = Join-Path $tmpRoot 'repo.zip'
    $extractRoot = Join-Path $tmpRoot 'extract'
    New-Item -ItemType Directory -Force -Path $tmpRoot, $extractRoot | Out-Null

    try {
        $encodedRef = [System.Uri]::EscapeDataString($Ref)
        $archiveUri = "https://codeload.github.com/$repoSlug/zip/$encodedRef"
        Write-Host "  ref:     $Ref"
        Invoke-WebRequest -Uri $archiveUri -OutFile $zipPath `
            -Headers (Get-GhAuthHeaders) -UseBasicParsing
        Expand-Archive -LiteralPath $zipPath -DestinationPath $extractRoot -Force
        $archiveRoot = Get-ChildItem -LiteralPath $extractRoot -Directory |
            Select-Object -First 1
        if (-not $archiveRoot) {
            throw "GitHub archive for $repoSlug@$Ref did not contain a root directory."
        }

        Clear-DirectoryExcept -Path $Destination -PreserveNames $PreserveNames
        Copy-DirectoryContents -Source $archiveRoot.FullName -Destination $Destination
        Install-GhSubmodules -RepoSlug $repoSlug -Ref $Ref -Destination $Destination
    } finally {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Install-GhSubmodules {
    [CmdletBinding()]
    param([Parameter(Mandatory)] [string]$RepoSlug,
          [Parameter(Mandatory)] [string]$Ref,
          [Parameter(Mandatory)] [string]$Destination)

    $gitmodulesPath = Join-Path $Destination '.gitmodules'
    $submodules = Read-GitModules $gitmodulesPath
    if ($submodules.Count -eq 0) { return }

    $encodedRef = [System.Uri]::EscapeDataString($Ref)
    $treeLines = Invoke-GhText api "repos/$RepoSlug/git/trees/$encodedRef" `
        --method GET -f recursive=1 `
        --jq '.tree[] | select(.type == \"commit\") | [.path, .sha] | @tsv'
    $shaByPath = @{}
    foreach ($line in $treeLines) {
        $parts = "$line" -split "`t"
        if ($parts.Count -eq 2) {
            $shaByPath[$parts[0]] = $parts[1]
        }
    }

    foreach ($submodule in $submodules) {
        if (-not $shaByPath.ContainsKey($submodule.Path)) {
            throw "Could not resolve submodule SHA for $($submodule.Path) in $RepoSlug@$Ref"
        }
        $submoduleDest = Join-Path $Destination $submodule.Path
        Write-Host "  submodule: $($submodule.Path) @ $($shaByPath[$submodule.Path])"
        Install-GhRepositoryArchive -Remote $submodule.Url `
            -Ref $shaByPath[$submodule.Path] `
            -Destination $submoduleDest
    }
}

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$RepoRoot = $RepoRoot -replace '\\', '/'
Write-Host "=== rocmlirTriton LLVM build wrapper (Windows) ===" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"

# ---------------------------------------------------------------------------
# Decide between "default windows-enablement" and "alternative Triton" modes.
# Sentinel: a non-empty $TritonRemote selects the alternative path.
# ---------------------------------------------------------------------------
$IsAlternativeStack = [bool]$TritonRemote

if ($IsAlternativeStack) {
    Write-Host "Mode: alternative Triton stack" -ForegroundColor Yellow
    if (-not $TritonRef) {
        throw '-TritonRef is required when -TritonRemote is set.'
    }
    if (-not $LlvmRemote) {
        throw '-LlvmRemote is required when -TritonRemote is set (the ' +
              'alternative Triton tree comes with its own matching LLVM).'
    }
    if (-not $LlvmRef) {
        throw '-LlvmRef is required when -TritonRemote is set (pin the ' +
              'exact LLVM commit so the build is reproducible).'
    }
    Write-Host "  Triton: $TritonRemote @ $TritonRef"
    Write-Host "  LLVM:   $LlvmRemote @ $LlvmRef"
} else {
    Write-Host "Mode: default windows-enablement" -ForegroundColor Cyan
    $TritonRemote = 'https://github.com/triton-lang/triton-windows.git'
    if (-not $TritonRef) {
        $TritonHashFile = "$RepoRoot/triton-windows-hash.txt"
        if (-not (Test-Path $TritonHashFile)) {
            throw "Pinned triton-windows hash not found: $TritonHashFile"
        }
        $TritonRef = (Get-Content $TritonHashFile -TotalCount 1).Trim()
    }
    # LlvmRemote / LlvmRef are derived later from the Triton tree's
    # cmake/llvm-hash.txt in Step 3.
    Write-Host "  Triton: $TritonRemote @ $TritonRef"
}

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
# cleanly, so we install the windows-enablement fork into external/triton.
#
# This override is purely local to the checkout -- it does NOT modify
# .gitmodules, so Linux clones using build-llvm.sh continue to get upstream
# triton.
# ---------------------------------------------------------------------------
Write-Host "--- Bringing in external/triton ($TritonRef from fork) ---" `
    -ForegroundColor Cyan
Write-Host "  remote: $TritonRemote"
Write-Host "  ref:    $TritonRef"

Install-GhRepositoryArchive -Remote $TritonRemote `
    -Ref $TritonRef `
    -Destination $TritonDir `
    -PreserveNames @('llvm-project')

# ---------------------------------------------------------------------------
# Step 2: Apply triton patches.
#
# Originally we applied every triton-patches/*.patch on top of the pinned
# Triton tree. An audit against triton-windows@main-windows showed that on
# the Windows path:
#   - patch1:         AtomicRMW fmax/smax disambiguation. Logically NOT
#                     present in the fork: emitAtomicRMW still concatenates
#                     `stringifyRMWOp(MAX) == "max"` and emits the nonexistent
#                     `llvm.amdgcn.raw.ptr.buffer.atomic.max` intrinsic.
#                     Required for fp/integer reduce_max correctness.
#   - patch2, patch3: cleanly merged into the fork.
#   - patch4, patch5: apply cleanly but proven not to be compile-time critical.
#   - patch6:         hard compile blocker on clang-cl (overload ambiguity in
#                     PartitionLoops.cpp). MUST be applied.
#
# We apply patch1 and patch6 explicitly on Windows. The other .patch files
# remain on disk so the Linux build path (scripts/build-llvm.sh) keeps
# applying them against upstream Triton.
# ---------------------------------------------------------------------------

$TritonPatchNames = @('patch1.patch', 'patch6.patch')
# In alternative-Triton mode, the patches are still attempted (they are
# generic Windows clang-cl fixes that the alternative tree may or may not
# already carry), but a "neither forward nor reverse apply" is downgraded
# from fatal to warning -- the alternative tree may have an equivalent fix
# that doesn't structurally match the upstream patch context.
$AnyPatchOnDisk = $TritonPatchNames | Where-Object { Test-Path (Join-Path $PatchesDir $_) }
if ($AnyPatchOnDisk) {
    Write-Host "--- Applying triton patches ---" -ForegroundColor Cyan
    $TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "rocmlir-patches-$PID"
    New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
    # Lower ErrorActionPreference for the patch block. Under 'Stop',
    # PowerShell wraps any native-command stderr in a NativeCommandError
    # *at invocation time* -- the `2>$null` / `2>&1` redirections happen
    # too late to suppress it. We manage failures ourselves via $LASTEXITCODE.
    $savedPref = $ErrorActionPreference
    $savedGitCeiling = $env:GIT_CEILING_DIRECTORIES
    $ErrorActionPreference = 'Continue'
    # Archives downloaded via gh do not have their own .git directory. Prevent
    # `git apply` from walking up to the parent rocmlirTriton repository and
    # treating Triton-relative patch paths as out-of-repo paths to skip.
    $env:GIT_CEILING_DIRECTORIES = $RepoRoot
    try {
        foreach ($patchName in $TritonPatchNames) {
            $patchPath = Join-Path $PatchesDir $patchName
            if (-not (Test-Path $patchPath)) {
                Write-Warning "$patchName not found on disk; skipping."
                continue
            }
            # Rewrite patch to LF before handing to `git apply`; Windows
            # core.autocrlf=true otherwise rejects the patch.
            $bytes  = [System.IO.File]::ReadAllBytes($patchPath)
            $text   = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "`r`n", "`n"
            $lfPath = Join-Path $TmpDir $patchName
            [System.IO.File]::WriteAllBytes(
                $lfPath, [System.Text.Encoding]::UTF8.GetBytes($text))

            & git -C $TritonDir apply --check $lfPath 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Applying:  $patchName"
                & git -C $TritonDir apply $lfPath 2>&1 | Write-Host
                if ($LASTEXITCODE -ne 0) {
                    throw "git apply failed for $patchName"
                }
                continue
            }
            & git -C $TritonDir apply --check --reverse $lfPath 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Skipping:  $patchName (already applied)"
            } elseif ($IsAlternativeStack) {
                # Alternative tree may carry an equivalent fix in a form that
                # doesn't match the upstream patch context. Warn loudly so the
                # developer can spot it but don't fail the build.
                Write-Warning ("$patchName does not apply cleanly to " +
                    "$TritonDir (neither forward nor reverse). The " +
                    "alternative Triton tree is presumed to carry an " +
                    "equivalent change. If you see build/test failures " +
                    "downstream that look like missing patch1 / patch6 " +
                    "fixes, re-check this tree.")
            } else {
                # Do NOT silently assume the patch is "absorbed". On Windows we
                # have repeatedly seen reset --hard leave garbage that makes both
                # forward and reverse `apply --check` fail -- which used to be
                # logged as a warning and then turned into mysterious test
                # failures (e.g. reduce_max -> missing buffer.atomic.max
                # intrinsic). Fail hard so the CI / developer notices.
                throw ("$patchName does not apply cleanly to $TritonDir -- " +
                    "neither forward nor reverse apply succeeds. Re-run with " +
                    "a clean external/triton checkout.")
            }
        }
    } finally {
        $ErrorActionPreference = $savedPref
        $env:GIT_CEILING_DIRECTORIES = $savedGitCeiling
        Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# Step 3: Fetch LLVM at the pinned commit.
#
# Default mode:        URL = github.com/llvm/llvm-project, ref read from
#                      $TritonDir/cmake/llvm-hash.txt.
# Alternative mode:    URL = $LlvmRemote, ref = $LlvmRef (both supplied by
#                      caller, validated at the top of this script).
# ---------------------------------------------------------------------------
$LlvmSrc    = "$TritonDir/llvm-project"
$LlvmBuild  = "$LlvmSrc/build"

if ($IsAlternativeStack) {
    $LlvmRemoteEffective = $LlvmRemote
    $LlvmHash            = $LlvmRef
} else {
    $LlvmRemoteEffective = 'https://github.com/llvm/llvm-project'
    $LlvmHashFile = "$TritonDir/cmake/llvm-hash.txt"
    if (-not (Test-Path $LlvmHashFile)) {
        throw "$LlvmHashFile not found (did the Triton source fetch finish?)."
    }
    $LlvmHash = (Get-Content $LlvmHashFile -Raw).Trim()
}

if ($Clean -and (Test-Path $LlvmSrc)) {
    Write-Host "Removing $LlvmSrc (--Clean)" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $LlvmSrc
}

Write-Host "--- Fetching llvm-project at $LlvmHash ---" -ForegroundColor Cyan
Install-GhRepositoryArchive -Remote $LlvmRemoteEffective `
    -Ref $LlvmHash `
    -Destination $LlvmSrc `
    -PreserveNames @('build')

# Apply llvm-patches/*.patch in sorted order, mirroring the Linux build-llvm.sh
# hook (which splices the same loop into Triton's build script). Each patch is
# CRLF-normalized so `git apply` accepts it under Windows core.autocrlf=true,
# and we check forward then reverse so re-runs are idempotent.
#
# In alternative-Triton mode the patches are still attempted (the alternative
# LLVM tree may or may not already carry them) but a "neither forward nor
# reverse apply" is downgraded from fatal to warning.
$LlvmPatchesDir = "$RepoRoot/llvm-patches"
if (Test-Path $LlvmPatchesDir) {
    $llvmPatches = Get-ChildItem -Path $LlvmPatchesDir -Filter '*.patch' -File |
        Sort-Object Name
    if ($llvmPatches.Count -gt 0) {
        Write-Host "--- Applying llvm-patches ---" -ForegroundColor Cyan
        $TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "rocmlir-llvm-patches-$PID"
        New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
        $savedPref = $ErrorActionPreference
        $savedGitCeiling = $env:GIT_CEILING_DIRECTORIES
        $ErrorActionPreference = 'Continue'
        # Keep `git apply` rooted at llvm-project when it was populated from a
        # GitHub archive rather than a native git clone.
        $env:GIT_CEILING_DIRECTORIES = $TritonDir
        try {
            foreach ($patch in $llvmPatches) {
                $bytes  = [System.IO.File]::ReadAllBytes($patch.FullName)
                $text   = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "`r`n", "`n"
                $lfPath = Join-Path $TmpDir $patch.Name
                [System.IO.File]::WriteAllBytes(
                    $lfPath, [System.Text.Encoding]::UTF8.GetBytes($text))

                & git -C $LlvmSrc apply --check $lfPath 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Applying:  $($patch.Name)"
                    & git -C $LlvmSrc apply $lfPath 2>&1 | Write-Host
                    if ($LASTEXITCODE -ne 0) {
                        throw "git apply failed for $($patch.Name)"
                    }
                } else {
                    & git -C $LlvmSrc apply --check --reverse $lfPath 2>$null | Out-Null
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "Skipping:  $($patch.Name) (already applied)"
                    } elseif ($IsAlternativeStack) {
                        Write-Warning ("$($patch.Name) does not apply " +
                            "cleanly to $LlvmSrc (neither forward nor " +
                            "reverse). The alternative LLVM tree is " +
                            "presumed to carry an equivalent change.")
                    } else {
                        # See the matching note in the triton-patches block:
                        # silently assuming presence has burned us before
                        # (430+ tests fell over with `tosa.target_env` parse
                        # errors when patch2 was quietly skipped). Fail hard.
                        throw ("$($patch.Name) does not apply cleanly to " +
                            "$LlvmSrc -- neither forward nor reverse apply " +
                            "succeeds. Re-run with a clean llvm-project " +
                            "checkout.")
                    }
                }
            }
        } finally {
            $ErrorActionPreference = $savedPref
            $env:GIT_CEILING_DIRECTORIES = $savedGitCeiling
            Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
        }
    }
}

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
    # MAX_PATH workaround: the upstream `mlir-minimal-opt-canonicalize` example
    # lives at tools\mlir\examples\minimal-opt\CMakeFiles\<dir>\<hash>\... which
    # easily exceeds Windows' 260-char rc.exe path limit when LLVM is built
    # under external\triton\llvm-project\build\. Examples are not used by
    # rocmlirTriton, so disable them outright.
    '-DLLVM_INCLUDE_EXAMPLES=OFF',
    '-DMLIR_INCLUDE_INTEGRATION_TESTS=OFF',
    '-DLLVM_ENABLE_ZSTD=OFF',
    '-DLLVM_TARGETS_TO_BUILD=Native;NVPTX;AMDGPU',
    '-DLLVM_ENABLE_PROJECTS=mlir;lld',
    '-DMLIR_ENABLE_ROCM_RUNNER=ON',
    "-DROCM_PATH=$HipPath",
    # ROCM_TEST_CHIPSET must be set whenever MLIR_ENABLE_ROCM_RUNNER=ON,
    # otherwise upstream MLIR's ExecutionEngine/CMakeLists.txt tries to
    # invoke rocm_agent_enumerator (a Linux-only ROCm helper missing from
    # the Windows HIP SDK) and aborts cmake configure. The value chosen
    # here doesn't affect the LLVM build artifacts: MLIR integration tests
    # are off and the LLVM AMDGPU backend itself is chipset-agnostic at
    # build time (the target is selected at llc invocation time with
    # -mcpu=...). rocmlirTriton's own Phase 3 configure picks the actual
    # test chipset via its own ROCK_TEST_CHIPSET / AMDGPU_TARGETS knobs.
    '-DROCM_TEST_CHIPSET=gfx1201',
    '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
)
Invoke-Native cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

# ---------------------------------------------------------------------------
# Step 5: Build
# ---------------------------------------------------------------------------
Write-Host "--- ninja -C $LlvmBuild ---" -ForegroundColor Cyan
Invoke-Native ninja -C $LlvmBuild
if ($LASTEXITCODE -ne 0) { throw "ninja build failed ($LASTEXITCODE)" }

# Record the hash this build dir is configured for, so a future Triton bump
# can detect the change and trigger a wipe (see Step 3 above).
Set-Content -Path $LlvmHashSentinel -Value $LlvmHash -NoNewline

Write-Host "=== LLVM build complete ===" -ForegroundColor Green
Write-Host "MLIR CMake config: $LlvmBuild/lib/cmake/mlir/MLIRConfig.cmake"
