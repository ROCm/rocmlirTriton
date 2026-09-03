set(ROCK_UNITTEST_SOURCES
  AmdArchDbTests.cpp
  BuilderUtilsTests.cpp
  GetRockInfoTests.cpp
  LFBOSearchTests.cpp
  LLMSearchTests.cpp
  PerfConfigOrderingGemmTests.cpp
  PerfConfigOrderingGemmGemmTests.cpp
  PerfConfigParsingTests.cpp
  RandomForestTests.cpp
  RocmDeviceNameTests.cpp
  TransformMapBuilderTests.cpp
  TuningParamAxesTests.cpp
  TuningSearchTests.cpp
  TosaUtilsTests.cpp
  transformMapUtilsTests.cpp
  ParamLookupTableTests.cpp
  PointerArithExpandTests.cpp
)

if(MLIR_ENABLE_ROCM_RUNNER)
  list(APPEND ROCK_UNITTEST_SOURCES HardwareLimitsTests.cpp)

  if(NOT TARGET hip::host OR NOT TARGET hip::amdhip64)
    rocmlir_find_hip()
  endif()
endif()

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

if(MLIR_ENABLE_ROCM_RUNNER)
  target_link_libraries(MLIRRockUnitTests
    PRIVATE
    hip::host
    hip::amdhip64
  )
endif()

# PointerArithExpandTests.cpp includes the (private) shared helpers header that
# lives alongside the Rock transform passes, and exercises helpers that emit
# Triton ops, so it needs both the Transforms source dir and the Triton
# includes. RandomForestTests.cpp likewise reaches into the Tuning source dir
# for RandomForest.h, which is internal to the LFBO search.
target_include_directories(MLIRRockUnitTests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/Dialect/Rock/Transforms
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/Dialect/Rock/Tuning
)
rocmlir_add_triton_includes(MLIRRockUnitTests)
