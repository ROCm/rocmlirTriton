add_rocmlir_dialect_library(MLIRRocMLIRSupport
  DetachReattach.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
