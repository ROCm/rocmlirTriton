set(ROCK_UNITTEST_SOURCES
  AmdArchDbTests.cpp
  BuilderUtilsTests.cpp
  PerfConfigParsingTests.cpp
  RocmDeviceNameTests.cpp
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
  transformMapUtilsTests.cpp
  ParamLookupTableTests.cpp
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
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRIR
  TritonAMDGPUToLLVM
  # LLD is strict about undefined symbols (unlike GNU ld); on Windows these
  # transitive Triton-AMD + MLIR libraries must be listed explicitly.
  TritonAMDGPUDialectToLLVM
  TritonAMDAnalysis
  MLIRUBToLLVM
  TritonAMDUtils
  # In triton-windows@main-windows, TritonInstrumentToLLVM (pulled in via the
  # AMD backend) references NVIDIA PTXBuilder helpers. With LLD's strict
  # undefined-symbol behavior we must explicitly link the NVIDIA backend lib.
  TritonNVIDIAGPUToLLVM
)
