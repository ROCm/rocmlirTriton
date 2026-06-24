#!/bin/bash

# Exit immediately if some command fails
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build LLVM/MLIR (handles submodule init, triton patches, ROCM_RUNNER, and building)
bash "$SCRIPT_DIR/scripts/build-llvm.sh"

rm -rf build
mkdir build
cd build

CXX_COMPILER=${CXX_COMPILER:-clang++-20}
C_COMPILER=${C_COMPILER:-clang-20}

# Pin LLVM/MLIR to the LLVM we just built under external/triton. Without this,
# find_package(LLVM) inside MLIRConfig.cmake can be hijacked by another LLVM of
# the same version on CMAKE_PREFIX_PATH (e.g. the ROCm SDK's bundled LLVM, which
# is built without the NVPTX target), causing "imported targets ... missing:
# LLVMNVPTXCodeGen ..." configure failures. An explicit *_DIR cache var has the
# highest find_package priority and overrides CMAKE_PREFIX_PATH.
LLVM_BUILD_DIR="$SCRIPT_DIR/external/triton/llvm-project/build"
MLIR_DIR=${MLIR_DIR:-"$LLVM_BUILD_DIR/lib/cmake/mlir"}
LLVM_DIR=${LLVM_DIR:-"$LLVM_BUILD_DIR/lib/cmake/llvm"}

cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DBUILD_FAT_LIBROCKCOMPILER=ON \
  -DLLD_BUILD_TOOLS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
  -DCMAKE_C_COMPILER=${C_COMPILER} \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  "$@"

ninja check-rocmlir-build-only
