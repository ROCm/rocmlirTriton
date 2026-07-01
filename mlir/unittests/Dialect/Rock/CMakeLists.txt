set(ROCK_UNITTEST_SOURCES
  AmdArchDbTests.cpp
  BuilderUtilsTests.cpp
  GetRockInfoTests.cpp
  PerfConfigOrderingGemmTests.cpp
  PerfConfigOrderingGemmGemmTests.cpp
  PerfConfigParsingTests.cpp
  RocmDeviceNameTests.cpp
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
  transformMapUtilsTests.cpp
  ParamLookupTableTests.cpp
  PointerArithExpandTests.cpp
)

add_rocmlir_unittest(MLIRRockUnitTests
  PARTIAL_SOURCES_INTENDED
  ${ROCK_UNITTEST_SOURCES}
)

target_link_libraries(MLIRRockUnitTests
  PRIVATE
  MLIRRockOps
  MLIRRockUtility
  MLIRRockTuning
  MLIRRockTransforms
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRIR
  TritonAMDGPUToLLVM
  TritonAMDUtils
)

# PointerArithExpandTests.cpp includes the (private) shared helpers header that
# lives alongside the Rock transform passes, and exercises helpers that emit
# Triton ops, so it needs both the Transforms source dir and the Triton
# includes.
target_include_directories(MLIRRockUnitTests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/Dialect/Rock/Transforms
)
rocmlir_add_triton_includes(MLIRRockUnitTests)

if(WIN32)
  target_link_libraries(MLIRRockUnitTests
    PRIVATE
    TritonAMDGPUDialectToLLVM
    TritonAMDAnalysis
    MLIRUBToLLVM
    TritonNVIDIAGPUToLLVM
  )
endif()
