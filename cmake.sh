#!/bin/bash

# The first time you set up the project, make sure you:
# 1. Download triton dependency
# $ git submodule update --init --recursive
#
# 2. Add -DMLIR_ENABLE_ROCM_RUNNER=ON to external/triton/scripts/build-llvm-project.sh
# $ nano external/triton/scripts/build-llvm-project.sh
#
# 3. Build triton's LLVM:
# $ cd external/triton/scripts/
# $ bash build-llvm-project.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/triton-patches"
TRITON_DIR="$SCRIPT_DIR/external/triton"

# Apply patches to the triton submodule if any exist
if [ -d "$PATCHES_DIR" ] && [ -n "$(ls -A "$PATCHES_DIR"/*.patch 2>/dev/null)" ]; then
    echo "Applying patches from $PATCHES_DIR to triton submodule..."
    cd "$TRITON_DIR"
    for patch in "$PATCHES_DIR"/*.patch; do
        if git apply --check "$patch" 2>/dev/null; then
            echo "Applying: $(basename "$patch")"
            git apply "$patch"
        elif git apply --check --reverse "$patch" 2>/dev/null; then
            echo "Skipping (already applied): $(basename "$patch")"
        else
            echo "ERROR: Patch cannot be applied (conflicts or other error): $(basename "$patch")"
        fi
    done
    cd "$SCRIPT_DIR"
fi

rm -rf build
mkdir build
cd build

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_FAT_LIBROCKCOMPILER=ON \
  -DLLD_BUILD_TOOLS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DROCK_E2E_TEST_ENABLED=ON \
  -DROCMLIR_DRIVER_PR_E2E_TEST_ENABLED=ON \
  -DROCMLIR_DRIVER_E2E_TEST_ENABLED=ON

ninja libconv-validation-wrappers.so; ninja check-rocmlir-build-only ci-performance-scripts
