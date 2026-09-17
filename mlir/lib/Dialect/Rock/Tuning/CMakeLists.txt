add_rocmlir_dialect_library(MLIRRockTuning
  ConvContext.cpp
  GridwiseGemmParams.cpp
  GridwiseGemmGemmParams.cpp
  LdsBlacklist.cpp
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
)

# RockTuningImpl.cpp includes Triton's TargetFeatures.h for the ISAFamily enum
rocmlir_add_triton_includes(MLIRRockTuning)

# One generated .inc per <arch, kernel, data type> key, so that regenerating one
# architecture touches only its own files. Collect whatever is checked in into
# the manifest ParamLookupTable.cpp includes. CONFIGURE_DEPENDS so a newly
# generated architecture is picked up by a plain `ninja`.
set(_problem_map_dir
  ${PROJECT_SOURCE_DIR}/mlir/include/mlir/Dialect/Rock/Tuning/QuickTuningProblemMap)
file(GLOB _problem_map_shards CONFIGURE_DEPENDS
  RELATIVE ${_problem_map_dir} ${_problem_map_dir}/*.inc)
list(SORT _problem_map_shards)
set(_problem_map_inc "")
foreach(_shard ${_problem_map_shards})
  string(APPEND _problem_map_inc
    "#include \"mlir/Dialect/Rock/Tuning/QuickTuningProblemMap/${_shard}\"\n")
endforeach()
file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/QuickTuningProblemMap.inc
  CONTENT "${_problem_map_inc}")
target_include_directories(obj.MLIRRockTuning PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
