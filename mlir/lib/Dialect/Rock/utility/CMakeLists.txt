add_rocmlir_dialect_library(MLIRRockUtility
  attentionUtils.cpp
  builderUtils.cpp
  loweringUtils.cpp
  transformMapUtils.cpp
  fusionUtils.cpp
  tosaUtils.cpp
  RocmDeviceName.cpp
  compileUtils.cpp
  tritonUtils.cpp
  KnobUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRArithDialect
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRRockOps
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIRROCDLDialect

  PRIVATE
  TritonIR
  TritonAMDGPUTransforms
)

# Triton include directories needed by tritonUtils.cpp.
rocmlir_add_triton_includes(MLIRRockUtility)
