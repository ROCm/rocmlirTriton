add_subdirectory(common)
if("ck" IN_LIST ROCMLIR_ENABLE_BENCHMARKS)
  add_subdirectory(ck-benchmark-driver)
endif()
if("hipblaslt" IN_LIST ROCMLIR_ENABLE_BENCHMARKS)
  add_subdirectory(hipblaslt-benchmark-driver)
endif()

set(PERFORMANCE_SCRIPTS
  convertRocBlasToPerfRunner.py
  createFusionPerformanceReports.py
  createPerformanceReports.py
  perfCommonUtils.py
  perfRunner.py
  parameterSweeps.py
  attentionSweeps.py
  perfRegressionReport.py
  reportUtils.py
  tuningRunner.py
  rocmlir_metrics.txt
  handleNewConfigs.py
)

list(TRANSFORM PERFORMANCE_SCRIPTS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# rocm-run is the bash wrapper around mlir-runner that the perf/tuning scripts
# invoke at runtime. It must sit in ${ROCMLIR_BIN_DIR} alongside them so that
# (a) the Python scripts can find it via paths.mlir_paths.rocm_run_path, and
# (b) the wrapper can self-locate the rocMLIR build dir from its own location.
set(ROCM_RUN_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/../widgets/rocm-run")

add_custom_target(ci-performance-scripts
	COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${ROCMLIR_BIN_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${ROCM_RUN_WRAPPER} ${ROCMLIR_BIN_DIR})
