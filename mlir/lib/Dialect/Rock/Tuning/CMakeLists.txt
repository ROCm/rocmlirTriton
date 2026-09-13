add_rocmlir_dialect_library(MLIRRockTuning
  ConvContext.cpp
  GridwiseGemmParams.cpp
  GridwiseGemmGemmParams.cpp
  LdsBlacklist.cpp
  OrigamiRanker.cpp
  RockTuningImpl.cpp
  ParamLookupTable.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock/Tuning

  DEPENDS
  MLIRRockOps
  MLIRSupport
  MLIRRockPassIncGen
)

target_link_libraries(MLIRRockTuning
  PRIVATE
  MLIRRockUtility
  MLIRIR
  origami
)

# RockTuningImpl.cpp includes Triton's TargetFeatures.h for the ISAFamily enum
rocmlir_add_triton_includes(MLIRRockTuning)

# add_mlir_library compiles the sources in obj.MLIRRockTuning, which does not
# inherit the link interface of MLIRRockTuning, so attach Origami's headers and
# its ORIGAMI_STATIC/ORIGAMI_NO_HIP defines there as well.
if(TARGET obj.MLIRRockTuning)
  target_include_directories(obj.MLIRRockTuning SYSTEM PRIVATE
    $<TARGET_PROPERTY:origami,INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(obj.MLIRRockTuning PRIVATE
    $<TARGET_PROPERTY:origami,INTERFACE_COMPILE_DEFINITIONS>)
endif()
