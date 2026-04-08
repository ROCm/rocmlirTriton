#!/usr/bin/env bash
#
# Wrapper around Triton's build-llvm-project.sh that:
#   1. Initializes git submodules (to get external/triton)
#   2. Applies triton-patches/*.patch to the triton submodule
#   3. Patches Triton's script to enable MLIR_ENABLE_ROCM_RUNNER
#   4. Runs the patched script to build LLVM/MLIR
#
# Repo root is resolved in this order:
#   1. Pre-set REPO_ROOT environment variable (if non-empty)
#   2. git rev-parse --show-toplevel (if inside a git worktree)
#   3. Fallback: parent of the directory containing this script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/.." && pwd))"
fi

echo "=== rocMLIR LLVM build wrapper ==="
echo "Repo root: $REPO_ROOT"

# Step 1: Initialize submodules
echo "--- Initializing git submodules ---"
git -C "$REPO_ROOT" submodule update --init --recursive

TRITON_DIR="$REPO_ROOT/external/triton"
PATCHES_DIR="$REPO_ROOT/triton-patches"
TRITON_BUILD_SCRIPT="$TRITON_DIR/scripts/build-llvm-project.sh"

if [ ! -f "$TRITON_BUILD_SCRIPT" ]; then
    echo "ERROR: Triton build script not found at $TRITON_BUILD_SCRIPT"
    echo "Did the submodule init succeed?"
    exit 1
fi

# Step 2: Apply triton patches if any exist
if [ -d "$PATCHES_DIR" ] && [ -n "$(ls -A "$PATCHES_DIR"/*.patch 2>/dev/null)" ]; then
    echo "--- Applying triton patches from $PATCHES_DIR ---"
    cd "$TRITON_DIR"
    for patch in "$PATCHES_DIR"/*.patch; do
        if git apply --check "$patch" 2>/dev/null; then
            echo "Applying: $(basename "$patch")"
            git apply "$patch"
        elif git apply --check --reverse "$patch" 2>/dev/null; then
            echo "Skipping (already applied): $(basename "$patch")"
        else
            echo "ERROR: Patch cannot be applied (conflicts or other error): $(basename "$patch")"
            exit 1
        fi
    done
    cd "$REPO_ROOT"
fi

# Step 3: Ensure MLIR_ENABLE_ROCM_RUNNER=ON (in-place, idempotent)
if grep -q 'DMLIR_ENABLE_ROCM_RUNNER=OFF' "$TRITON_BUILD_SCRIPT"; then
    echo "--- Patching Triton's build-llvm-project.sh: enabling MLIR_ENABLE_ROCM_RUNNER ---"
    sed -i 's/DMLIR_ENABLE_ROCM_RUNNER=OFF/DMLIR_ENABLE_ROCM_RUNNER=ON/g' "$TRITON_BUILD_SCRIPT"
elif grep -q 'DMLIR_ENABLE_ROCM_RUNNER=ON' "$TRITON_BUILD_SCRIPT"; then
    echo "--- MLIR_ENABLE_ROCM_RUNNER already ON, no patch needed ---"
else
    echo "--- Adding MLIR_ENABLE_ROCM_RUNNER=ON to Triton's build script ---"
    sed -i '/DCMAKE_CXX_COMPILER/a\              -DMLIR_ENABLE_ROCM_RUNNER=ON' "$TRITON_BUILD_SCRIPT"
fi

# Step 4: Run Triton's build-llvm-project.sh
echo "--- Building LLVM/MLIR via Triton's script ---"
cd "$REPO_ROOT/external/triton/scripts"
bash build-llvm-project.sh "$@"

echo "=== LLVM build complete ==="
