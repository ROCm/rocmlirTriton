#!/usr/bin/env bash
# Generate TableGen headers required by clang-tidy static checks.
#
# mlir-headers covers upstream MLIR and most chained targets. The targets
# below are rocmlirTriton orphan tablegen targets not hooked into
# mlir-headers. Any new add_public_tablegen_target() in mlir/ must be
# appended here.
set -euo pipefail

BUILD_DIR="${1:-build}"

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "Error: ${BUILD_DIR}/compile_commands.json not found; configure the project first." >&2
  exit 1
fi

cmake --build "${BUILD_DIR}" --target \
  mlir-headers \
  MLIRCpuPassIncGen \
  MLIRRockAttrDefsIncGen \
  MLIRRockTuningParamAttrInterfaceIncGen \
  MLIRRockPassIncGen \
  RocMLIRConversionPassIncGen \
  MLIRMIGraphXTypeIncGen \
  MLIRMIGraphXPassIncGen
