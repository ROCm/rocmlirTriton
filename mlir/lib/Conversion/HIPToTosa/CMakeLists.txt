add_rocmlir_conversion_library(MLIRHIPToTosa
  HIPToTosa.cpp
  HIPToTosaPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRTosaTransforms
  MLIRRockTransforms
  MLIRTransforms
  MLIRSupport
  MLIRHIPDialect
  )
