add_subdirectory(common)
if("ck" IN_LIST ROCMLIR_ENABLE_BENCHMARKS)
  add_subdirectory(ck-benchmark-driver)
endif()
if("hipblaslt" IN_LIST ROCMLIR_ENABLE_BENCHMARKS)
  add_subdirectory(hipblaslt-benchmark-driver)
endif()

set(PERFORMANCE_SCRIPTS
  attentionPerfUtils.py
  compareAttentionPerformance.py
  convertRocBlasToPerfRunner.py
  createFusionPerformanceReports.py
  createPerformanceReports.py
  filterAttentionConfigs.py
  generateAttentionPerformanceWorkbook.py
  perfCommonUtils.py
  perfRunner.py
  parameterSweeps.py
  attentionSweeps.py
  perfRegressionReport.py
  reportUtils.py
  runAttentionBranchBenchmark.py
  tuningRunner.py
  rocmlir_metrics.txt
  handleNewConfigs.py
)

set(PERFORMANCE_RESOURCES
  README-attention-perf-comparison.md
)

set(PERFORMANCE_CONFIGS
  configs/pr347-impacted-tier1-attention-configs
  configs/pr347-impacted-tier1-attention-configs.json
  configs/tier1-attention-configs
)

list(TRANSFORM PERFORMANCE_SCRIPTS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PERFORMANCE_RESOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM PERFORMANCE_CONFIGS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

execute_process(
  COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE ROCMLIR_SOURCE_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ROCMLIR_SOURCE_SHA_RESULT
)
if(NOT ROCMLIR_SOURCE_SHA_RESULT EQUAL 0)
  message(FATAL_ERROR "Unable to determine the rocmlirTriton source revision")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rocmlir-source-sha"
  "${ROCMLIR_SOURCE_SHA}\n")

# rocm-run is the bash wrapper around mlir-runner that the perf/tuning scripts
# invoke at runtime. It must sit in ${ROCMLIR_BIN_DIR} alongside them so that
# (a) the Python scripts can find it via paths.mlir_paths.rocm_run_path, and
# (b) the wrapper can self-locate the rocMLIR build dir from its own location.
set(ROCM_RUN_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/../widgets/rocm-run")

if(NOT TARGET amd_arch_db)
  message(FATAL_ERROR
    "ci-performance-scripts requires the amd_arch_db Python binding, which is "
    "currently defined under mlir/test/common_utils and is only configured "
    "when MLIR_INCLUDE_TESTS=ON.")
endif()

add_custom_target(ci-performance-scripts
  COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${ROCMLIR_BIN_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_RESOURCES} ${ROCMLIR_BIN_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ROCMLIR_BIN_DIR}/configs
  COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_CONFIGS} ${ROCMLIR_BIN_DIR}/configs
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_BINARY_DIR}/rocmlir-source-sha
          ${CMAKE_BINARY_DIR}/rocmlir-source-sha
  COMMAND ${CMAKE_COMMAND} -E copy ${ROCM_RUN_WRAPPER} ${ROCMLIR_BIN_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:amd_arch_db> ${ROCMLIR_BIN_DIR})
add_dependencies(ci-performance-scripts amd_arch_db)
