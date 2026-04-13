add_rocmlir_conversion_library(RocmlirPromoteSoftmaxPrecision
  RocmlirPromoteSoftmaxPrecision.cpp

  DEPENDS
  RocMLIRConversionPassIncGen
)

target_link_libraries(RocmlirPromoteSoftmaxPrecision
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  MLIRTransformUtils
  MLIRRockUtility
)
