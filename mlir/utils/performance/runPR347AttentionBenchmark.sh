#!/usr/bin/env bash
#
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

readonly DEFAULT_BASE_REF="0409d83a7f151269baad29592a4deac79bd39ae7"
readonly DEFAULT_CANDIDATE_REF="users/umayadav/pr347-attention-perf"

repo=""
workspace="${TMPDIR:-/tmp}/rocmlir-pr347-attention-${USER:-user}"
base_ref="$DEFAULT_BASE_REF"
candidate_ref="$DEFAULT_CANDIDATE_REF"
candidate_sha_override=""
gpu=""
samples=3
detach=true

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Fetch, build, quick-tune, and benchmark the PR #347 baseline and candidate in
isolated detached worktrees.

Options:
  --repo PATH           Existing rocmlirTriton clone (default: current repo)
  --workspace PATH      Worktrees and results directory
                        (default: $workspace)
  --gpu N               Require physical GPU N (default: first proven idle GPU)
  --samples N           Benchmark samples per config (default: 3)
  --base-ref REF        Baseline revision (default: $DEFAULT_BASE_REF)
  --candidate-ref REF   Candidate origin branch (default: $DEFAULT_CANDIDATE_REF)
  --candidate-sha SHA   Use an already-fetched exact candidate revision
  --foreground          Keep the benchmark attached to this shell
  -h, --help            Show this help

The script safely reuses matching worktrees, builds, tuning state, and completed
samples. It fails instead of replacing a mismatched or dirty worktree.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

while (($#)); do
  case "$1" in
  --repo)
    (($# >= 2)) || die "--repo requires a path"
    repo="$2"
    shift 2
    ;;
  --workspace)
    (($# >= 2)) || die "--workspace requires a path"
    workspace="$2"
    shift 2
    ;;
  --gpu)
    (($# >= 2)) || die "--gpu requires an index"
    gpu="$2"
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "--gpu must be nonnegative"
    shift 2
    ;;
  --samples)
    (($# >= 2)) || die "--samples requires a count"
    samples="$2"
    [[ "$samples" =~ ^[1-9][0-9]*$ ]] || die "--samples must be positive"
    shift 2
    ;;
  --base-ref)
    (($# >= 2)) || die "--base-ref requires a revision"
    base_ref="$2"
    shift 2
    ;;
  --candidate-ref)
    (($# >= 2)) || die "--candidate-ref requires a branch"
    candidate_ref="$2"
    shift 2
    ;;
  --candidate-sha)
    (($# >= 2)) || die "--candidate-sha requires a revision"
    candidate_sha_override="$2"
    shift 2
    ;;
  --foreground)
    detach=false
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    die "unknown option: $1"
    ;;
  esac
done

require_command bash
require_command cmake
require_command git
require_command ninja
require_command python3
require_command realpath

if ! command -v rocm-smi >/dev/null 2>&1; then
  rocm_bin="${ROCM_PATH:-/opt/rocm}/bin"
  [[ -x "$rocm_bin/rocm-smi" ]] || die "rocm-smi is required"
  export PATH="$rocm_bin:$PATH"
fi

if [[ -z "$repo" ]]; then
  repo="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null)" ||
    die "run from a rocmlirTriton clone or pass --repo"
fi
repo="$(git -C "$repo" rev-parse --show-toplevel 2>/dev/null)" ||
  die "not a Git repository: $repo"
repo="$(realpath "$repo")"
workspace="$(realpath -m "$workspace")"

if [[ -n "$candidate_sha_override" ]]; then
  candidate_sha="$(git -C "$repo" rev-parse "$candidate_sha_override^{commit}")" ||
    die "candidate revision is not available: $candidate_sha_override"
else
  echo "Fetching origin/$candidate_ref"
  git -C "$repo" fetch --no-tags origin \
    "refs/heads/$candidate_ref:refs/remotes/origin/$candidate_ref"
  candidate_sha="$(git -C "$repo" rev-parse "origin/$candidate_ref^{commit}")"
fi

if ! git -C "$repo" cat-file -e "$base_ref^{commit}" 2>/dev/null; then
  echo "Fetching baseline revision $base_ref"
  if ! git -C "$repo" fetch --no-tags origin "$base_ref"; then
    [[ "$(git -C "$repo" rev-parse --is-shallow-repository)" == "true" ]] ||
      die "unable to fetch baseline revision: $base_ref"
    git -C "$repo" fetch --no-tags --unshallow origin
  fi
fi
base_sha="$(git -C "$repo" rev-parse "$base_ref^{commit}")"

mkdir -p "$workspace/sources" "$workspace/configs" "$workspace/results"
base_source="$workspace/sources/base"
candidate_source="$workspace/sources/candidate"

ensure_worktree() {
  local label="$1"
  local path="$2"
  local revision="$3"

  if [[ -e "$path" ]]; then
    [[ -d "$path" && -e "$path/.git" ]] ||
      die "$label path exists but is not a worktree: $path"
    actual="$(git -C "$path" rev-parse HEAD)"
    [[ "$actual" == "$revision" ]] ||
      die "$label worktree is at $actual, expected $revision; use another --workspace"
    [[ -z "$(git -C "$path" status --porcelain)" ]] ||
      die "$label worktree is dirty: $path"
    echo "Reusing $label worktree at $revision"
    return
  fi

  git -C "$repo" worktree add --detach "$path" "$revision"
}

ensure_worktree base "$base_source" "$base_sha"
ensure_worktree candidate "$candidate_source" "$candidate_sha"

build_revision() {
  local label="$1"
  local source="$2"
  local revision="$3"
  local build="$source/build"
  local cache="$build/CMakeCache.txt"

  if [[ -f "$cache" ]] &&
    ! grep -Fqx "CMAKE_HOME_DIRECTORY:INTERNAL=$source" "$cache"; then
    die "$label build was configured from another source: $build"
  fi

  if [[ -f "$cache" && -f "$build/build.ninja" ]]; then
    echo "Reusing configured $label build"
  else
    echo "Configuring and building $label ($revision)"
    (cd "$source" && bash ./cmake.sh)
  fi

  cmake --build "$build" --target check-rocmlir-build-only
  cmake --build "$build" --target ci-performance-scripts

  # The pinned baseline predates the CMake-generated source stamp consumed by
  # runAttentionBranchBenchmark.py. Write it only after both build targets pass.
  printf '%s\n' "$revision" >"$build/rocmlir-source-sha"
}

build_revision base "$base_source" "$base_sha"
build_revision candidate "$candidate_source" "$candidate_sha"

configs="$workspace/configs/pr347-causal-kvcache-attention-configs"
python3 "$candidate_source/mlir/utils/performance/filterAttentionConfigs.py" \
  --input "$candidate_source/mlir/utils/performance/configs/tier1-attention-configs" \
  --output "$configs" \
  --mode performance

benchmark_command=(
  python3
  "$candidate_source/mlir/utils/performance/runAttentionBranchBenchmark.py"
  --base-source "$base_source"
  --base-build "$base_source/build"
  --candidate-source "$candidate_source"
  --candidate-build "$candidate_source/build"
  --configs "$configs"
  --output-dir "$workspace/results"
  --samples "$samples"
)
[[ -z "$gpu" ]] || benchmark_command+=(--gpu "$gpu")
$detach && benchmark_command+=(--detach)

echo "Launching PR #347 attention benchmark"
printf '  %q' "${benchmark_command[@]}"
printf '\n'
"${benchmark_command[@]}"

if $detach; then
  echo "Results: $workspace/results"
  echo "Log:     $workspace/results/detached.log"
fi
