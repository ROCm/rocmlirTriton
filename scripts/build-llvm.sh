#!/usr/bin/env bash
#
# Wrapper around Triton's build-llvm-project.sh that:
#   1. Initializes git submodules (to get external/triton)
#   2. Applies triton-patches/*.patch to the triton submodule
#   3. Patches Triton's script to enable MLIR_ENABLE_ROCM_RUNNER
#   4. Patches Triton's script to apply llvm-patches/*.patch after its
#      `git reset --hard` of the LLVM checkout (so our patches survive)
#   5. Runs the patched script to build LLVM/MLIR
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
TRITON_PATCHES_DIR="$REPO_ROOT/triton-patches"
LLVM_PATCHES_DIR="$REPO_ROOT/llvm-patches"
TRITON_BUILD_SCRIPT="$TRITON_DIR/scripts/build-llvm-project.sh"

if [ ! -f "$TRITON_BUILD_SCRIPT" ]; then
    echo "ERROR: Triton build script not found at $TRITON_BUILD_SCRIPT"
    echo "Did the submodule init succeed?"
    exit 1
fi

# Step 2: Apply triton patches if any exist
if [ -d "$TRITON_PATCHES_DIR" ] && [ -n "$(ls -A "$TRITON_PATCHES_DIR"/*.patch 2>/dev/null)" ]; then
    echo "--- Applying triton patches from $TRITON_PATCHES_DIR ---"
    cd "$TRITON_DIR"
    for patch in "$TRITON_PATCHES_DIR"/*.patch; do
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

# Step 4: Inject llvm-patches application into Triton's build script
#
# Triton's script does `git reset --hard $LLVM_COMMIT_HASH` on the LLVM
# checkout, which would wipe out any patches we apply ahead of time. Instead,
# we splice a hook into the script that runs immediately after the reset to
# apply our patches there (idempotently). The hook is delimited by markers so
# subsequent runs replace it cleanly.
ROCMLIR_HOOK_BEGIN="# >>> rocmlir llvm-patches hook >>>"
ROCMLIR_HOOK_END="# <<< rocmlir llvm-patches hook <<<"

# Remove any previously-injected hook (idempotent re-runs).
if grep -qF "$ROCMLIR_HOOK_BEGIN" "$TRITON_BUILD_SCRIPT"; then
    sed -i "/$ROCMLIR_HOOK_BEGIN/,/$ROCMLIR_HOOK_END/d" "$TRITON_BUILD_SCRIPT"
fi

if [ -d "$LLVM_PATCHES_DIR" ] && [ -n "$(ls -A "$LLVM_PATCHES_DIR"/*.patch 2>/dev/null)" ]; then
    echo "--- Injecting llvm-patches hook into Triton's build script ---"
    HOOK_FILE="$(mktemp)"
    cat > "$HOOK_FILE" <<EOF
$ROCMLIR_HOOK_BEGIN
echo "--- Applying llvm patches from $LLVM_PATCHES_DIR ---"
for patch in "$LLVM_PATCHES_DIR"/*.patch; do
    if git -C "\$LLVM_PROJECT_PATH" apply --check "\$patch" 2>/dev/null; then
        echo "Applying: \$(basename "\$patch")"
        git -C "\$LLVM_PROJECT_PATH" apply "\$patch"
    elif git -C "\$LLVM_PROJECT_PATH" apply --check --reverse "\$patch" 2>/dev/null; then
        echo "Skipping (already applied): \$(basename "\$patch")"
    else
        echo "ERROR: LLVM patch cannot be applied: \$(basename "\$patch")"
        exit 1
    fi
done
$ROCMLIR_HOOK_END
EOF
    # Insert the hook right after the `git ... reset --hard ...` line.
    sed -i "/git -C \"\$LLVM_PROJECT_PATH\" reset --hard/r $HOOK_FILE" "$TRITON_BUILD_SCRIPT"
    rm -f "$HOOK_FILE"
fi

# Step 5: Run Triton's build-llvm-project.sh
echo "--- Building LLVM/MLIR via Triton's script ---"
cd "$REPO_ROOT/external/triton/scripts"
bash build-llvm-project.sh "$@"

echo "=== LLVM build complete ==="
