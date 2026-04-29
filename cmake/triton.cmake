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

# User must provide MLIR_DIR (e.g., from a built LLVM or system installation)
if(NOT DEFINED MLIR_DIR)
  # Try common locations
  if(DEFINED ENV{MLIR_DIR})
    set(MLIR_DIR $ENV{MLIR_DIR} CACHE PATH "Path to MLIR CMake config")
  elseif(EXISTS "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake")
    # Default: Use LLVM built by Triton's build-llvm-project.sh script
    set(MLIR_DIR "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR CMake config")
  elseif(DEFINED LLVM_LIBRARY_DIR)
    set(MLIR_DIR "${LLVM_LIBRARY_DIR}/cmake/mlir" CACHE PATH "Path to MLIR CMake config")
  else()
    # LLVM/MLIR not found — automatically build it via our wrapper script
    set(_build_llvm_script "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-llvm.sh")
    if(EXISTS "${_build_llvm_script}")
      message(STATUS "LLVM/MLIR not found. Running ${_build_llvm_script} to build it...")
      execute_process(
        COMMAND bash "${_build_llvm_script}"
        RESULT_VARIABLE _build_llvm_result
      )
      if(NOT _build_llvm_result EQUAL 0)
        message(FATAL_ERROR "scripts/build-llvm.sh failed (exit code: ${_build_llvm_result})")
      endif()
      # After building, the MLIR config should now exist
      if(EXISTS "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake")
        set(MLIR_DIR "${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir" CACHE PATH "Path to MLIR CMake config")
      else()
        message(FATAL_ERROR
          "scripts/build-llvm.sh completed but MLIRConfig.cmake was not found at\n"
          "  ${TRITON_PROJECT_DIR}/llvm-project/build/lib/cmake/mlir/\n"
          "Check the build output above for errors.")
      endif()
    else()
      message(FATAL_ERROR 
        "MLIR_DIR must be set to the path containing MLIRConfig.cmake\n"
        "Example: cmake -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir ..\n"
        "You can build LLVM/MLIR using: bash scripts/build-llvm.sh")
    endif()
  endif()
endif()

message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# Find MLIR package (this also sets up LLVM variables)
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
# ROCm / HIP SDK Configuration
#
# Accepted inputs (first hit wins):
#   -DROCM_PATH=<dir>         (cache/command-line)
#   ENV{ROCM_PATH}=<dir>
#   ENV{HIP_PATH}=<dir>       (the HIP SDK installer sets this on Windows)
#   platform default          ("C:/opt/rocm" on Windows, "/opt/rocm" elsewhere)
#
# The Linux ROCm layout has HIP CMake files under   ${ROCM_PATH}/hip/cmake
# The Windows HIP SDK layout has them under        ${ROCM_PATH}/lib/cmake/hip
# find_package(hip) / find_package(hiprtc) pick up either location once
# ${ROCM_PATH} is on CMAKE_PREFIX_PATH.
#===----------------------------------------------------------------------===//

if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm / HIP SDK installation")
  elseif(DEFINED ENV{HIP_PATH})
    set(ROCM_PATH "$ENV{HIP_PATH}" CACHE PATH "Path to ROCm / HIP SDK installation")
  elseif(WIN32)
    set(ROCM_PATH "C:/opt/rocm" CACHE PATH "Path to ROCm / HIP SDK installation")
  else()
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm / HIP SDK installation")
  endif()
endif()
file(TO_CMAKE_PATH "${ROCM_PATH}" ROCM_PATH)
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

# Expose ROCM_PATH to subdirectories (notably rocmlir-tuning-driver) via
# CMAKE_PREFIX_PATH so find_package(hip) / find_package(hiprtc) succeed on
# both Linux and the Windows HIP SDK without hard-coded per-OS path hacks.
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# Legacy HIP CMake module location (Linux); the Windows HIP SDK exposes these
# via the standard lib/cmake/<pkg> layout, discovered through CMAKE_PREFIX_PATH.
if(NOT WIN32)
  list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")
endif()

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

# Triton's CMakeLists.txt now FATAL_ERRORs at configure time if
# TRITON_CACHE_PATH is empty (see external/triton/CMakeLists.txt). Normally
# Triton's setup.py supplies this from $HOME/.triton/cache; since rocmlirTriton
# integrates Triton via add_subdirectory and doesn't ship a Python kernel
# runtime, we default to a build-tree-local directory. Users can still
# override on the cmake command line.
if(NOT TRITON_CACHE_PATH)
  set(TRITON_CACHE_PATH "${CMAKE_BINARY_DIR}/triton-cache" CACHE PATH "Path to triton cache")
  file(MAKE_DIRECTORY "${TRITON_CACHE_PATH}")
endif()

# LLVM_SYSPATH tells Triton where to find a pre-built LLVM. If empty,
# Triton's build_helpers.py downloads a ~1.8 GB prebuilt tarball from the
# triton-windows release bucket. We always have a locally-built LLVM
# at the path MLIR_DIR was derived from (see above), so point Triton at
# it to skip the download. Layout expected by Triton:
#   ${LLVM_SYSPATH}/include   (LLVM headers, populated by find_package(MLIR))
#   ${LLVM_SYSPATH}/lib       (LLVM libraries + lib/cmake/{mlir,lld})
#   ${LLVM_SYSPATH}/bin       (FileCheck, llvm-tblgen, mlir-tblgen, etc.)
# MLIR_DIR is .../build/lib/cmake/mlir, so LLVM_SYSPATH is the .../build
# directory two levels up.
if(NOT LLVM_SYSPATH)
  get_filename_component(_llvm_lib_cmake_dir "${MLIR_DIR}" DIRECTORY)
  get_filename_component(_llvm_lib_dir       "${_llvm_lib_cmake_dir}" DIRECTORY)
  get_filename_component(_llvm_syspath_guess "${_llvm_lib_dir}" DIRECTORY)
  if(EXISTS "${_llvm_syspath_guess}/bin")
    set(LLVM_SYSPATH "${_llvm_syspath_guess}" CACHE PATH "Path to system LLVM installation")
  endif()
endif()
if(LLVM_SYSPATH)
  message(STATUS "LLVM_SYSPATH (skips Triton's prebuilt LLVM download): ${LLVM_SYSPATH}")
endif()

#===----------------------------------------------------------------------===//
# Include Directories
#===----------------------------------------------------------------------===//

# Triton includes
list(APPEND TRITON_INCLUDE_DIRS
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party
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