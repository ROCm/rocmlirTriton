add_rocmlir_dialect_library(MLIRHIPDialect
  IR/HipCausalConvCanonicalize.cpp
  IR/HipDialect.cpp
  IR/HipDpsOpInterface.cpp
  IR/HipOpStateInterface.cpp
  IR/HipReifyResultShapesImpl.cpp
  IR/HipShapeUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/hip/Dialect/IR

  DEPENDS
  HipDialectIncGen
  HipTypesIncGen
  HipDpsOpInterfaceIncGen
  HipOpStateInterfaceIncGen
  HipOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRTensorDialect
  MLIRMemRefDialect
  # HipOpStateInterface.cpp's emitOpStateConstruct helper emits LLVM-dialect
  # calls for the per-op state constructors.
  MLIRLLVMDialect
)
