message(STATUS "Adding Triton src dependency")

#===----------------------------------------------------------------------===//
# Triton Project Configuration
# NOTE: Triton requires a pre-built LLVM/MLIR. Set MLIR_DIR before configuring.
#===----------------------------------------------------------------------===//

set(TRITON_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/triton")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/triton")

#===----------------------------------------------------------------------===//
# LLVM/MLIR Configuration
# Triton uses find_package(MLIR) - must be provided externally
#===----------------------------------------------------------------------===//

# Resolve MLIR_DIR.  Order of preference:
#   1. Explicit MLIR_DIR (from CMake cache or -D on command line) -- highest priority.
#   2. MLIR_DIR environment variable.
#   3. CMAKE_PREFIX_PATH (e.g. cget/MIGraphX install prefix).
#   4. In-tree LLVM build under external/triton/llvm-project/build/ (legacy
#      build-llvm-project.sh workflow).
# (3) MUST come before (4): when consumed via cget the source tarball can
# contain a stale external/triton/llvm-project/build/ directory whose
# MLIRConfig.cmake bakes in absolute paths from the dev tree (mlir-tblgen,
# tablegen target paths). Picking those up silently makes the build try to
# invoke a non-existent /mnt/data/.../mlir-tblgen.
if(NOT DEFINED MLIR_DIR)
  if(DEFINED ENV{MLIR_DIR})
    set(MLIR_DIR $ENV{MLIR_DIR} CACHE PATH "Path to MLIR CMake config")
  else()
    # Let CMake search CMAKE_PREFIX_PATH (cget toolchain populates this).
    find_package(MLIR QUIET CONFIG)
    if(NOT MLIR_FOUND
       AND EXISTS "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake")
      # Legacy fallback: in-tree LLVM built via build-llvm-project.sh.
      set(MLIR_DIR "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir"
          CACHE PATH "Path to MLIR CMake config")
    elseif(NOT MLIR_FOUND AND DEFINED LLVM_LIBRARY_DIR)
      set(MLIR_DIR "${LLVM_LIBRARY_DIR}/cmake/mlir"
          CACHE PATH "Path to MLIR CMake config")
    elseif(NOT MLIR_FOUND)
      message(FATAL_ERROR
        "Could not locate MLIRConfig.cmake. Set -DMLIR_DIR=<path>, add the "
        "LLVM/MLIR install prefix to CMAKE_PREFIX_PATH, or build LLVM in "
        "external/triton/llvm-project/build/ via build-llvm-project.sh.")
    endif()
  endif()
endif()

message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# Find MLIR package (this also sets up LLVM variables). If MLIR was already
# located by the QUIET probe above this is a no-op; otherwise it forces lookup
# at the path we just resolved.
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

# Set up LLD_DIR based on MLIR_DIR location
get_filename_component(_llvm_cmake_dir "${MLIR_DIR}" DIRECTORY)
set(LLD_DIR "${_llvm_cmake_dir}/lld" CACHE PATH "Path to LLD CMake config")
message(STATUS "LLD_DIR: ${LLD_DIR}")
message(STATUS "Found MLIR ${MLIR_VERSION} at ${MLIR_DIR}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

# Set up CMake module paths from found MLIR/LLVM
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include LLVM/MLIR CMake utilities
include(TableGen)
include(AddLLVM)
include(AddMLIR)

#===----------------------------------------------------------------------===//
# ROCm Configuration
#===----------------------------------------------------------------------===//

if(NOT DEFINED ROCM_PATH)
  if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
  endif()
endif()
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")

#===----------------------------------------------------------------------===//
# Triton Build Options (matching external/triton/CMakeLists.txt)
#===----------------------------------------------------------------------===//

# Disable Python module - we're using C++ API only
set(TRITON_BUILD_PYTHON_MODULE OFF CACHE BOOL "Don't build Python bindings")

# Disable Proton profiler
set(TRITON_BUILD_PROTON OFF CACHE BOOL "Don't build Proton profiler")

# Disable unit tests (can enable later)
set(TRITON_BUILD_UT OFF CACHE BOOL "Don't build Triton unit tests")

# Enable AMD backend via TRITON_CODEGEN_BACKENDS
set(TRITON_CODEGEN_BACKENDS "amd" "nvidia" CACHE STRING "Enable AMD codegen backend")

#===----------------------------------------------------------------------===//
# Include Directories
#===----------------------------------------------------------------------===//

# Triton include dirs. Used by every target that pulls in Triton headers
# Add new Triton include paths here.
list(APPEND TRITON_INCLUDE_DIRS
  ${TRITON_PROJECT_DIR}
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party
  ${TRITON_PROJECT_DIR}/third_party/amd
  ${TRITON_PROJECT_DIR}/third_party/amd/include
  ${TRITON_BINARY_DIR}/third_party/amd
  ${TRITON_BINARY_DIR}/third_party/amd/include
  ${TRITON_PROJECT_DIR}/third_party/nvidia
  ${TRITON_BINARY_DIR}/third_party/nvidia
  ${TRITON_BINARY_DIR}/third_party/nvidia/include
)

#===----------------------------------------------------------------------===//
# For lit testing configuration
#===----------------------------------------------------------------------===//

set(MLIR_CMAKE_CONFIG_DIR "${MLIR_DIR}")
set(MLIR_TABLEGEN_EXE mlir-tblgen)

#===----------------------------------------------------------------------===//
# Configure TRITON_CACHE_PATH (mimics get_triton_cache_path() logic in setup.py)
#===----------------------------------------------------------------------===//
if(NOT TRITON_CACHE_PATH)
  if(DEFINED ENV{TRITON_HOME})
    set(TRITON_CACHE_PATH "$ENV{TRITON_HOME}/.triton"
        CACHE PATH "Path to triton cache")
  elseif(DEFINED ENV{HOME})
    set(TRITON_CACHE_PATH "$ENV{HOME}/.triton"
        CACHE PATH "Path to triton cache")
  else()
    set(TRITON_CACHE_PATH "${CMAKE_BINARY_DIR}/.triton-cache"
        CACHE PATH "Path to triton cache")
  endif()
endif()
message(STATUS "TRITON_CACHE_PATH: ${TRITON_CACHE_PATH}")

#===----------------------------------------------------------------------===//
# Tell Triton where LLVM lives so it reuses our local build instead of
# downloading a prebuilt version.
#===----------------------------------------------------------------------===//
if(NOT LLVM_SYSPATH)
  if(DEFINED ENV{LLVM_SYSPATH})
    set(LLVM_SYSPATH "$ENV{LLVM_SYSPATH}" CACHE PATH "Path to LLVM install used by Triton")
  else()
    # Reuse the MLIR_DIR-derived LLVM prefix so Triton doesn't redownload prebuilts.
    # MLIR_DIR is e.g. .../llvm-project/build/lib/cmake/mlir → prefix is .../llvm-project/build
    get_filename_component(_mlir_parent "${MLIR_DIR}" DIRECTORY)   # lib/cmake
    get_filename_component(_mlir_parent "${_mlir_parent}" DIRECTORY) # lib
    get_filename_component(_llvm_prefix "${_mlir_parent}" DIRECTORY)  # build
    set(LLVM_SYSPATH "${_llvm_prefix}" CACHE PATH "Path to LLVM install used by Triton")
  endif()
endif()
message(STATUS "LLVM_SYSPATH: ${LLVM_SYSPATH}")

#===----------------------------------------------------------------------===//
# Add Triton subdirectory
#===----------------------------------------------------------------------===//

add_subdirectory("${TRITON_PROJECT_DIR}" "external/triton" EXCLUDE_FROM_ALL)

#===----------------------------------------------------------------------===//
# Create dummy targets for MLIR tablegen dependencies
# When using pre-built MLIR, tablegen targets don't exist but headers do
#===----------------------------------------------------------------------===//

if(NOT TARGET MLIRConversionPassIncGen)
  add_custom_target(MLIRConversionPassIncGen)
endif()

#===----------------------------------------------------------------------===//
# Helper Functions for rocMLIR Libraries
#===----------------------------------------------------------------------===//

function(add_rocmlir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_DIALECT_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_dialect_library)

function(add_rocmlir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_CONVERSION_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_conversion_library)

function(add_rocmlir_test_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_TEST_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_test_library)

function(add_rocmlir_public_c_api_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_PUBLIC_C_API_LIBS ${name})
  add_mlir_library(${name}
    ${ARGN}
    EXCLUDE_FROM_LIBMLIR
    ENABLE_AGGREGATION
    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir-c
  )
  set_target_properties(obj.${name}
    PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  target_compile_definitions(obj.${name}
    PRIVATE
    -DMLIR_CAPI_BUILDING_LIBRARY=1
  )
endfunction()

function(add_rocmlir_tool name)
  set(exclude_from_all "")
  if(BUILD_FAT_LIBROCKCOMPILER)
    set(exclude_from_all "EXCLUDE_FROM_ALL")
    set(LLVM_BUILD_TOOLS OFF)
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_mlir_tool(${name} ${exclude_from_all} ${ARGN})
endfunction()

# Helper function for Rock-to-Triton libraries
function(add_rocmlir_triton_library name)
  set_property(GLOBAL APPEND PROPERTY ROCMLIR_TRITON_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_rocmlir_triton_library)

# Attach the Triton include directories (TRITON_INCLUDE_DIRS) to <target> and
# its obj.* variant when one exists (as it does for libraries created via
# add_mlir_library). Marked SYSTEM so our strict warnings (-Wshadow, -Wundef,
# ...) don't fire inside Triton headers.
function(rocmlir_add_triton_includes target)
  target_include_directories(${target} SYSTEM PRIVATE ${TRITON_INCLUDE_DIRS})
  if(TARGET obj.${target})
    target_include_directories(obj.${target} SYSTEM PRIVATE ${TRITON_INCLUDE_DIRS})
  endif()
endfunction(rocmlir_add_triton_includes)

