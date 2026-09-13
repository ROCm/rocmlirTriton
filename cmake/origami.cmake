# Build the vendored Origami analytical GEMM model (external/origami) as a
# static library that librockCompiler can absorb.
#
# Origami ships its own CMakeLists.txt, but it is not used here: it requires
# CMake 3.25, pulls rocm-cmake in over the network via FetchContent, and links
# hip::host. This file compiles the same sources directly instead, so the
# vendored tree stays byte-for-byte re-vendorable (see external/origami/commit.txt).
#
# ORIGAMI_NO_HIP drops Origami's device-querying entry points, leaving the
# arch-only ones. librockCompiler cross-targets GPUs it is not running on and
# must link without the HIP runtime, which the rest of the compiler does not
# depend on today.

set(ORIGAMI_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/origami")
set(ORIGAMI_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/origami")

if(NOT EXISTS "${ORIGAMI_SRC_DIR}/include/origami/origami.hpp")
  message(FATAL_ERROR
    "Vendored Origami not found at ${ORIGAMI_SRC_DIR}. "
    "See external/origami/commit.txt for how to re-vendor it.")
endif()

add_library(origami STATIC
  "${ORIGAMI_SRC_DIR}/src/origami/attention.cpp"
  "${ORIGAMI_SRC_DIR}/src/origami/gemm.cpp"
  "${ORIGAMI_SRC_DIR}/src/origami/hardware.cpp"
  "${ORIGAMI_SRC_DIR}/src/origami/heuristics.cpp"
  "${ORIGAMI_SRC_DIR}/src/origami/logger.cpp"
  "${ORIGAMI_SRC_DIR}/src/origami/origami.cpp"
  "${ORIGAMI_SRC_DIR}/src/origami/streamk.cpp"
  "${ORIGAMI_SRC_DIR}/src/origami/types.cpp"
  "${ORIGAMI_SRC_DIR}/src/simulator/tensilelite/formocast.cpp"
  "${ORIGAMI_SRC_DIR}/src/simulator/tensilelite/formocast_simulator.cpp"
)

# Origami's headers include "origami/origami_export.h", which upstream generates
# at configure time. Generate it the same way rather than checking one in, so the
# vendored tree needs no edit. STATIC_DEFINE makes the macros expand to nothing.
include(GenerateExportHeader)
generate_export_header(origami
  BASE_NAME ORIGAMI
  EXPORT_MACRO_NAME ORIGAMI_EXPORT
  EXPORT_FILE_NAME "${ORIGAMI_BINARY_DIR}/include/origami/origami_export.h"
  STATIC_DEFINE ORIGAMI_STATIC
)

# SYSTEM so rocMLIR's strict warnings do not fire inside third-party headers,
# matching how the vendored Triton headers are attached (rocmlir_add_triton_includes).
target_include_directories(origami SYSTEM PUBLIC
  "${ORIGAMI_SRC_DIR}/include"
  "${ORIGAMI_BINARY_DIR}/include"
)

target_compile_definitions(origami PUBLIC ORIGAMI_STATIC ORIGAMI_NO_HIP)
target_compile_features(origami PUBLIC cxx_std_17)

set_target_properties(origami PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  # Land beside the other rocMLIR archives so the fat-library step, which globs
  # ROCMLIR_LIB_DIR by name, can pick it up.
  ARCHIVE_OUTPUT_DIRECTORY "${ROCMLIR_LIB_DIR}"
)
