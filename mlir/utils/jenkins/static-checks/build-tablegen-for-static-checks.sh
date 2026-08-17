#!/usr/bin/env bash
# Generate TableGen headers required by clang-tidy static checks.
#
# mlir-headers covers add_mlir_dialect() / dialect tablegen; mlir-generic-headers
# covers add_mlir_interface() / generic tablegen. The targets below are
# rocmlirTriton orphan TableGen targets (bare add_public_tablegen_target() in
# mlir/) not hooked into either umbrella. Any new such orphan must be appended
# here.
set -euo pipefail

BUILD_DIR="${1:-build}"

if [[ ! -f "${BUILD_DIR}/compile_commands.json" ]]; then
  echo "Error: ${BUILD_DIR}/compile_commands.json not found; configure the project first." >&2
  exit 1
fi

cmake --build "${BUILD_DIR}" --target \
  mlir-headers \
  mlir-generic-headers \
  MLIRCpuPassIncGen \
  MLIRRockAttrDefsIncGen \
  MLIRRockTuningParamAttrInterfaceIncGen \
  MLIRRockPassIncGen \
  RocMLIRConversionPassIncGen \
  MLIRMIGraphXTypeIncGen \
  MLIRMIGraphXPassIncGen
