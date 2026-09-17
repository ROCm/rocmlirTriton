set(ROCK_UNITTEST_SOURCES
  AmdArchDbTests.cpp
  BuilderUtilsTests.cpp
  GetRockInfoTests.cpp
  LoweringUtilsTests.cpp
  PerfConfigOrderingGemmTests.cpp
  PerfConfigOrderingGemmGemmTests.cpp
  PerfConfigParsingTests.cpp
  RocmDeviceNameTests.cpp
  TransformMapBuilderTests.cpp
  TosaUtilsTests.cpp
  transformMapUtilsTests.cpp
  ParamLookupTableTests.cpp
  PointerArithExpandTests.cpp
  QuickTuningProblemMapTests.cpp
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
  # TritonAMDGPUToLLVM does not stand alone: its objects reach into the
  # instrumentation and NVIDIA PTX-builder libraries, which in turn need the
  # UB-to-LLVM patterns. Linking it by itself leaves those references
  # unresolved. This is the same block every other consumer of it uses; keep it
  # in sync with mlir/test/common_utils/amd_arch_db/CMakeLists.txt, the closest
  # analogue since both are test-only targets.
  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonGPUTransforms
  TritonAMDGPUToLLVM
  TritonAMDGPUTransforms
  TritonAMDAnalysis
  TritonAMDGPUDialectToLLVM
  TritonNVIDIAGPUToLLVM
  TritonAMDUtils
  GluonTransforms
  TritonInstrumentToLLVM
  NVHopperTransforms
  ProtonToProtonGPU
  ProtonGPUToLLVM
  ProtonNVIDIAGPUToLLVM
  ProtonAMDGPUToLLVM
  ProtonGPUTransforms
  TritonLLVMIR
  NVGPUToLLVM
  TritonInstrumentTransforms
  # amd_arch_db picks this up transitively through MLIROptLib, which the unit
  # tests do not link.
  MLIRUBToLLVM
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
# includes.
target_include_directories(MLIRRockUnitTests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/Dialect/Rock/Transforms
)
rocmlir_add_triton_includes(MLIRRockUnitTests)
