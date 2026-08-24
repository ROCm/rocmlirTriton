#!/bin/bash

# Exit immediately if some command fails
set -e

# --no-clean reconfigures an existing build directory instead of wiping it. The
# CMake cache is then sticky, so callers must pass every flag they care about.
clean_build_dir=1
cmake_args=()
for arg in "$@"; do
  case "$arg" in
    --no-clean) clean_build_dir=0 ;;
    *) cmake_args+=("$arg") ;;
  esac
done

if [ "$clean_build_dir" -eq 1 ]; then
  rm -rf build
fi
mkdir -p build
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
  "${cmake_args[@]}"

ninja check-rocmlir-build-only
