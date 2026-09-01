cmake_minimum_required(VERSION 3.20)

# Allow VERSION for projects, as expected in CMake 3.0+
cmake_policy(SET CMP0048 NEW)

# Resolve the ROCm/HIP root once before project(). The Windows toolchain setup
# below needs it this early, and cmake/triton.cmake reuses the same value for
# package discovery. Precedence is:
#   -DROCM_PATH -> ENV{ROCM_PATH} -> ENV{HIP_PATH} -> platform default
if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm / HIP SDK installation")
  elseif(CMAKE_HOST_WIN32 AND DEFINED ENV{HIP_PATH})
    set(ROCM_PATH "$ENV{HIP_PATH}" CACHE PATH "Path to ROCm / HIP SDK installation")
  elseif(CMAKE_HOST_WIN32)
    set(ROCM_PATH "C:/opt/rocm" CACHE PATH "Path to ROCm / HIP SDK installation")
  else()
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm / HIP SDK installation")
  endif()
endif()
file(TO_CMAKE_PATH "${ROCM_PATH}" ROCM_PATH)

#===----------------------------------------------------------------------===//
# Windows toolchain auto-detection
# NOTE: This MUST happen before project() so CMake doesn't detect MSVC first.
#===----------------------------------------------------------------------===//

if(CMAKE_HOST_WIN32)
  if(NOT CMAKE_C_COMPILER AND NOT CMAKE_CXX_COMPILER)
    # clang-cl lives under lib/llvm/bin in TheRock and under bin in the HIP SDK.
    if(EXISTS "${ROCM_PATH}/lib/llvm/bin/clang-cl.exe")
      set(_rocmlir_compiler_dir "${ROCM_PATH}/lib/llvm/bin")
    elseif(EXISTS "${ROCM_PATH}/bin/clang-cl.exe")
      set(_rocmlir_compiler_dir "${ROCM_PATH}/bin")
    else()
      message(FATAL_ERROR
        "clang-cl not found under ${ROCM_PATH} "
        "(checked lib/llvm/bin and bin). Set ROCM_PATH/HIP_PATH or -DROCM_PATH to "
        "your ROCm/HIPSDK install (or pass -DCMAKE_C_COMPILER/-DCMAKE_CXX_COMPILER).")
    endif()
    set(CMAKE_C_COMPILER
        "${_rocmlir_compiler_dir}/clang-cl.exe" CACHE FILEPATH "")
    set(CMAKE_CXX_COMPILER
        "${_rocmlir_compiler_dir}/clang-cl.exe" CACHE FILEPATH "")
  else()
    if(NOT EXISTS "${CMAKE_C_COMPILER}" OR
       NOT EXISTS "${CMAKE_CXX_COMPILER}")
      message(FATAL_ERROR
        "Set CMAKE_C_COMPILER and CMAKE_CXX_COMPILER to existing clang-cl executables.")
    endif()
    get_filename_component(
      _rocmlir_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
  endif()
  message(STATUS "Using ROCm clang-cl at ${CMAKE_CXX_COMPILER}")

  if(NOT CMAKE_LINKER)
    set(CMAKE_LINKER
        "${_rocmlir_compiler_dir}/lld-link.exe"
        CACHE FILEPATH "Path to lld-link")
  endif()
  if(NOT EXISTS "${CMAKE_LINKER}" OR
     NOT CMAKE_LINKER MATCHES "lld-link(\\.exe)?$")
    message(FATAL_ERROR
      "Set CMAKE_LINKER to an existing lld-link executable.")
  endif()
  message(STATUS "Using lld-link at ${CMAKE_LINKER}")
endif()

# clang-cl with Ninja defaults an unset build type to Debug during project().
# Select the Windows default before that initialization occurs.
if(CMAKE_HOST_WIN32 AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

project(rocMLIR VERSION 2.0.0 LANGUAGES CXX C)

if(WIN32 AND
   (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
  message(FATAL_ERROR
    "Windows builds require clang-cl; configured compiler is "
    "${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}).")
endif()

# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "")

#===----------------------------------------------------------------------===//
# Windows / MSVC-compatible compiler adjustments
#===----------------------------------------------------------------------===//

if(WIN32)
  # clang-cl/MSVC default to delayed template parsing, which breaks the standard
  # two-phase name lookup Triton/MLIR rely on (multiple `detail` namespaces
  # found at instantiation time); disable it so clang-cl matches clang on Linux.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-delayed-template-parsing>")

  # Triton passes cl.exe-only /Zc flags plus /WX; clang-cl warns
  # -Wunused-command-line-argument on them and /WX makes it fatal. Suppress it
  # for C/C++ only (ml64 misparses a leading -W as its /W warning level).
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-command-line-argument>")

  # Suppress MSVC STL noise: MSVC's <complex> header deprecates std::complex<T>
  # for non-floating T; MLIR's APFloat-based code legitimately uses it.
  add_compile_definitions(
    _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING
    # MSVC's getenv/etc. deprecation warnings are not actionable here.
    _CRT_SECURE_NO_WARNINGS
    _CRT_NONSTDC_NO_WARNINGS
  )
endif()

# Adapted from https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set policy CMP0057 to support IN_LIST operators
cmake_policy(SET CMP0057 NEW)

#===----------------------------------------------------------------------===//
# Parallel build configuration
#===----------------------------------------------------------------------===//

set(ROCMLIR_PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(ROCMLIR_PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${ROCMLIR_PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
  endif()
elseif(ROCMLIR_PARALLEL_LINK_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()

set(ROCMLIR_PARALLEL_COMPILE_JOBS "" CACHE STRING
  "Define the maximum number of concurrent compile jobs (Ninja only).")
if(CMAKE_GENERATOR MATCHES "Ninja")
  if(ROCMLIR_PARALLEL_COMPILE_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=${ROCMLIR_PARALLEL_COMPILE_JOBS})
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
  endif()
elseif(ROCMLIR_PARALLEL_COMPILE_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()

#===----------------------------------------------------------------------===//
# rocMLIR Options
#===----------------------------------------------------------------------===//

# MLIR_INCLUDE_TESTS is deliberately left alone: MLIR defaults it to
# LLVM_INCLUDE_TESTS, so standalone builds get check-rocmlir-build-only while
# consumers that pass -DLLVM_INCLUDE_TESTS=Off (e.g. MIGraphX) get no test tree.
# Disable E2E tests by default since they require JIT runner
set(ROCMLIR_DRIVER_E2E_TEST_ENABLED OFF CACHE BOOL "Enable build E2E tests for Rock driver")
set(ROCMLIR_DRIVER_RANDOM_DATA_SEED "none" CACHE STRING "Enable E2E tests using random data")
set(ROCMLIR_GEN_FLAGS "" CACHE BOOL "Set feature flag for rocmlir-gen")
# Disable E2E tests by default since they require JIT runner
set(ROCK_E2E_TEST_ENABLED OFF CACHE BOOL "Enable build rock E2E tests")
set(ROCMLIR_ENABLE_BENCHMARKS "" CACHE STRING "List of enabled benchmarks")
option(ROCMLIR_BUILD_TUNING_DRIVER
  "Build rocmlir-tuning-driver (default ON when BUILD_FAT_LIBROCKCOMPILER)" OFF)

set(ROCMLIR_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin" CACHE PATH "")
set(ROCMLIR_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "")
message(STATUS "ROCMLIR_BIN_DIR: ${ROCMLIR_BIN_DIR}")

if(ROCMLIR_ENABLE_BENCHMARKS STREQUAL "all")
  set(ROCMLIR_ENABLE_BENCHMARKS CACHE STRING "ck;hipblaslt" FORCE)
endif()

#===----------------------------------------------------------------------===//
# Library type and linkage settings
#===----------------------------------------------------------------------===//

if(BUILD_MIXR_TARGET)
  set(BUILD_FAT_LIBROCKCOMPILER ON CACHE BOOL "Build static rockCompiler library")
endif()

if(NOT DEFINED BUILD_FAT_LIBROCKCOMPILER)
  set(BUILD_FAT_LIBROCKCOMPILER OFF CACHE BOOL "Build fat rockCompiler library to link into Rock driver")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(ROCMLIR_DRIVER_PR_E2E_TEST_ENABLED OFF CACHE BOOL "Enable build PR-triggered E2E tests for Rock driver")

# The ROCm runner JITs host code, which a static fat-library build cannot
# support, so it is off there and rocmlir-tuning-driver is built through its own
# switch instead. This must precede cmake/triton.cmake, which decides
# LLVM_TARGETS_TO_BUILD from MLIR_ENABLE_ROCM_RUNNER.
if(BUILD_FAT_LIBROCKCOMPILER)
  set(MLIR_ENABLE_ROCM_RUNNER 0 CACHE BOOL "")
  set(ROCMLIR_BUILD_TUNING_DRIVER ON CACHE BOOL "" FORCE)
else()
  set(MLIR_ENABLE_ROCM_RUNNER 1 CACHE BOOL "")
endif()

if(BUILD_FAT_LIBROCKCOMPILER AND NOT WIN32)
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING "")
  list(APPEND CMAKE_IGNORE_PATH /opt/conda)
endif()

message(BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}")

#===----------------------------------------------------------------------===//
# Dual-LLVM coexistence: -Bsymbolic-functions
#===----------------------------------------------------------------------===//
#
# Bind each module's own LLVM *function* references internally so a second LLVM
# in the same process (e.g. ROCm Comgr's libLLVM.so, reached via the HIP
# runtime's DT_NEEDED chain) cannot interpose them. Applied here -- before the
# Triton and mlir subdirectories are added -- so it covers every target in this
# build regardless of how CMake is invoked, rather than relying on cmake.sh.
#
# Must be -Bsymbolic-functions, NOT full -Bsymbolic: binding *data* symbols
# locally breaks MLIR TypeID identity (http://llvm.org/pr51420), which is the
# exact cross-module symbol-identity problem we are trying to avoid.
if(NOT WIN32)
  add_link_options($<$<PLATFORM_ID:Linux>:LINKER:-Bsymbolic-functions>)
endif()

#===----------------------------------------------------------------------===//
# External dependency paths for RPATH and in-tree LLVM discovery.
# These must be set BEFORE include(cmake/triton.cmake) because the in-tree
# LLVM path uses LLVM_EXTERNAL_BUILD_DIR to populate MLIR_INCLUDE_DIRS etc.
#===----------------------------------------------------------------------===//

set(LLVM_EXTERNAL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm-project")
set(LLVM_EXTERNAL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project")
set(LLVM_EXTERNAL_BIN_DIR "${LLVM_EXTERNAL_BUILD_DIR}/llvm/bin")
set(LLVM_EXTERNAL_LIB_DIR "${LLVM_EXTERNAL_BUILD_DIR}/llvm/lib")

set(TRITON_EXTERNAL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/triton")
set(TRITON_EXTERNAL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/triton")
set(TRITON_EXTERNAL_BIN_DIR "${TRITON_EXTERNAL_BUILD_DIR}/bin")
set(TRITON_EXTERNAL_LIB_DIR "${TRITON_EXTERNAL_BUILD_DIR}/lib")

message(STATUS "LLVM_EXTERNAL_BIN_DIR: ${LLVM_EXTERNAL_BIN_DIR}")
message(STATUS "LLVM_EXTERNAL_LIB_DIR: ${LLVM_EXTERNAL_LIB_DIR}")
message(STATUS "TRITON_EXTERNAL_BIN_DIR: ${TRITON_EXTERNAL_BIN_DIR}")
message(STATUS "TRITON_EXTERNAL_LIB_DIR: ${TRITON_EXTERNAL_LIB_DIR}")

#===----------------------------------------------------------------------===//
# Address sanitizer workaround
#===----------------------------------------------------------------------===//

if(DEFINED ENV{ADDRESS_SANITIZER})
  if($ENV{ADDRESS_SANITIZER} MATCHES "ON")
    message("Address sanitizer workarounds")
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
      OUTPUT_VARIABLE clang_asan_lib_file
      ERROR_VARIABLE clang_stderr
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE clang_exit_code)
    if(NOT "${clang_exit_code}" STREQUAL "0")
      message(FATAL_ERROR
      "Unable to invoke clang to find asan lib dir: ${clang_stderr}")
    endif()
    file(TO_CMAKE_PATH "${clang_asan_lib_file}" clang_asan_lib_file_cmake)
    get_filename_component(clang_asan_lib_dir "${clang_asan_lib_file_cmake}" DIRECTORY)
    message(STATUS "Asan lib directory ${clang_asan_lib_dir}")
    list(APPEND CMAKE_BUILD_RPATH "${clang_asan_lib_dir}")
  endif()
endif()

#===----------------------------------------------------------------------===//
# Set up LLVM/MLIR and Triton. Both are vendored under external/ and built
# in-tree as part of this build (see cmake/triton.cmake).
#===----------------------------------------------------------------------===//

include(cmake/triton.cmake)
include(cmake/rocm.cmake)

# TODO: Re-enable mlir-hal when updated for newer LLVM (22+)
# mlir-hal has API incompatibilities with LLVM 22 (getArgAttrsAttr, etc.)
# include(cmake/mlir-hal.cmake)

#===----------------------------------------------------------------------===//
# RPATH configuration
#===----------------------------------------------------------------------===//

if(NOT WIN32)
  # Add library paths to RPATH
  list(PREPEND CMAKE_BUILD_RPATH "${TRITON_EXTERNAL_LIB_DIR}")
    list(PREPEND CMAKE_BUILD_RPATH "${ROCMLIR_LIB_DIR}")
  # Also add LLVM library dir if available
  if(DEFINED LLVM_LIBRARY_DIR)
    list(PREPEND CMAKE_BUILD_RPATH "${LLVM_LIBRARY_DIR}")
  endif()
    message(STATUS "Final CMAKE_BUILD_RPATH: ${CMAKE_BUILD_RPATH}")
endif()

#===----------------------------------------------------------------------===//
# Build rocMLIR dialects
#===----------------------------------------------------------------------===//

add_subdirectory(mlir)
