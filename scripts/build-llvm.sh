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

# Step 3b: Link LLVM tools/libs with -Bsymbolic-functions (in-place, idempotent)
#
# When a statically-embedded LLVM coexists in a process with a second LLVM
# loaded at runtime -- e.g. ROCm Comgr's libLLVM.so reached via DT_NEEDED
# through libamd_comgr.so -- the two instances share LLVM's cl::opt /
# pass-registry singletons and abort during dynamic init with
#   SmallPtrSet.h: Assertion `Bucket < End' failed.
# Linking with -Bsymbolic-functions binds each module's own LLVM *function*
# references internally so the second LLVM can no longer interpose them. See
# ROCm/TheRock#4981, where the ROCm maintainers declined to make LLVM fully
# static and asked the LLVM-embedding consumers (us) to link this way instead.
#
# NOTE: We must use -Bsymbolic-functions, NOT full -Bsymbolic. MLIR forbids full
# -Bsymbolic (mlir/CMakeLists.txt FATAL_ERRORs on it) because binding *data*
# symbols locally breaks TypeID identity (http://llvm.org/pr51420) -- the exact
# cross-module symbol-identity problem we are trying to fix.
#
# We inject the flags into CMAKE_ARGS right before Triton's `cmake` call rather
# than splicing them into the hardcoded default array. Triton's script only uses
# that array when invoked with no positional args AND no preset CMAKE_ARGS env
# var; injecting at the call site instead covers all three construction paths.
#
# We seed the flag via the CMAKE_*_LINKER_FLAGS_INIT variables rather than the
# CMAKE_*_LINKER_FLAGS cache variables themselves. _INIT only provides the
# *initial* value when the cache variable is not set, so it never clobbers a
# value the caller passed explicitly (e.g. -DCMAKE_SHARED_LINKER_FLAGS=...) --
# passing those explicitly is the documented opt-out. No canonical repo/CI
# invocation passes custom LLVM linker flags, so this simple seed is sufficient.
SYMBOLIC_HOOK_BEGIN="# >>> rocmlir -Bsymbolic-functions hook >>>"
SYMBOLIC_HOOK_END="# <<< rocmlir -Bsymbolic-functions hook <<<"

# Remove any previously-injected hook (idempotent re-runs)...
if grep -qF "$SYMBOLIC_HOOK_BEGIN" "$TRITON_BUILD_SCRIPT"; then
    sed -i "/$SYMBOLIC_HOOK_BEGIN/,/$SYMBOLIC_HOOK_END/d" "$TRITON_BUILD_SCRIPT"
fi
# ...and drop the legacy inline injection (three -DCMAKE_*_LINKER_FLAGS lines an
# earlier version of this wrapper spliced into the default CMAKE_ARGS array).
sed -i '/-DCMAKE_\(EXE\|SHARED\|MODULE\)_LINKER_FLAGS="-Wl,-Bsymbolic-functions"/d' "$TRITON_BUILD_SCRIPT"

echo "--- Injecting -Bsymbolic-functions hook into Triton's build script ---"
SYMBOLIC_HOOK_FILE="$(mktemp)"
cat > "$SYMBOLIC_HOOK_FILE" <<EOF
$SYMBOLIC_HOOK_BEGIN
# Seed -Bsymbolic-functions as the initial value of each linker-flag cache
# variable. _INIT only seeds when the caller has not set the variable
# explicitly, so this never clobbers a caller-provided -DCMAKE_*_LINKER_FLAGS=...
CMAKE_ARGS+=(
    -DCMAKE_EXE_LINKER_FLAGS_INIT="-Wl,-Bsymbolic-functions"
    -DCMAKE_SHARED_LINKER_FLAGS_INIT="-Wl,-Bsymbolic-functions"
    -DCMAKE_MODULE_LINKER_FLAGS_INIT="-Wl,-Bsymbolic-functions"
)
echo "Seeded -Bsymbolic-functions via CMAKE_*_LINKER_FLAGS_INIT"
$SYMBOLIC_HOOK_END
EOF
# Insert the hook right after the "Configuring with ..." line, i.e. immediately
# before Triton's `cmake "\${CMAKE_ARGS[@]}"` invocation.
sed -i "/echo \"Configuring with/r $SYMBOLIC_HOOK_FILE" "$TRITON_BUILD_SCRIPT"
rm -f "$SYMBOLIC_HOOK_FILE"

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
# Triton's preceding \`git reset --hard\` restores tracked files but leaves
# untracked files behind. Patches that create new files would otherwise leave
# those files on disk across runs, so make sure to clean everything
# before applying patches.
git -C "\$LLVM_PROJECT_PATH" clean -fd
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
