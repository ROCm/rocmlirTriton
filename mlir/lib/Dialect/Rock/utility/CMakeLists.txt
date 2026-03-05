add_rocmlir_dialect_library(MLIRRockUtility
  builderUtils.cpp
  loweringUtils.cpp
  transformMapUtils.cpp
  fusionUtils.cpp
  tosaUtils.cpp
  RocmDeviceName.cpp
  compileUtils.cpp

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
)
