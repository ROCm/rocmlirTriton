#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build LLVM/MLIR (handles submodule init, triton patches, ROCM_RUNNER, and building)
bash "$SCRIPT_DIR/scripts/build-llvm.sh"

rm -rf build
mkdir build
cd build

CXX_COMPILER=${CXX_COMPILER:-clang++-20}
C_COMPILER=${C_COMPILER:-clang-20}

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_FAT_LIBROCKCOMPILER=ON \
  -DLLD_BUILD_TOOLS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
  -DCMAKE_C_COMPILER=${C_COMPILER} \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DROCK_E2E_TEST_ENABLED=ON \
  -DROCMLIR_DRIVER_PR_E2E_TEST_ENABLED=ON \
  -DROCMLIR_DRIVER_E2E_TEST_ENABLED=ON

ninja libconv-validation-wrappers.so; ninja check-rocmlir-build-only ci-performance-scripts
