#!/bin/bash

# Exit immediately if some command fails
set -e

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
  "$@"

ninja check-rocmlir-build-only
