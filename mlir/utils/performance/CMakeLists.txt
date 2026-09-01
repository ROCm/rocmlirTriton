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
  crossCompile.py
  perfCommonUtils.py
  perfRunner.py
  parameterSweeps.py
  attentionSweeps.py
  perfRegressionReport.py
  reportUtils.py
  tuningArgumentUtils.py
  tuningRunner.py
  rocmlir_metrics.txt
  handleNewConfigs.py
  generateLDSBlacklist.py
)

list(TRANSFORM PERFORMANCE_SCRIPTS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# rocm-run is the bash wrapper around mlir-runner that the perf/tuning scripts
# invoke at runtime. It must sit in ${ROCMLIR_BIN_DIR} alongside them so that
# (a) the Python scripts can find it via paths.mlir_paths.rocm_run_path, and
# (b) the wrapper can self-locate the rocMLIR build dir from its own location.
set(ROCM_RUN_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/../widgets/rocm-run")

# amd_arch_db is defined under mlir/test/common_utils and so only exists when
# tests are configured. Tuning-driver-only builds reach this directory for
# benchmark-driver-utils but have no test tree, so the script bundle is skipped
# there instead of failing the configure. Whenever tests *are* configured the
# binding is a hard requirement, since lit gates on it.
if(MLIR_INCLUDE_TESTS AND NOT TARGET amd_arch_db)
  message(FATAL_ERROR
    "ci-performance-scripts requires the amd_arch_db Python binding, which is "
    "currently defined under mlir/test/common_utils and is only configured "
    "when MLIR_INCLUDE_TESTS=ON.")
endif()

if(TARGET amd_arch_db)
  add_custom_target(ci-performance-scripts
    COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${ROCMLIR_BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${ROCM_RUN_WRAPPER} ${ROCMLIR_BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:amd_arch_db> ${ROCMLIR_BIN_DIR})
  add_dependencies(ci-performance-scripts amd_arch_db)
else()
  message(STATUS
    "Not building ci-performance-scripts: amd_arch_db is unavailable "
    "(MLIR_INCLUDE_TESTS=OFF)")
endif()
