message(STATUS "Adding Triton src dependency")

#===----------------------------------------------------------------------===//
# Paths for building rocmlirTriton
#===----------------------------------------------------------------------===//

set(TRITON_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/triton")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/triton")
set(ROCMLIR_LLVM_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project")

# external/llvm-project and external/triton are vendored directly in this repo
# (imported via git subtree) with our downstream patches committed on top, so
# there are no submodules to initialize and nothing to patch at configure time.
# The patch files are kept under llvm-patches/ and triton-patches/ for
# provenance and to ease the next upstream bump.

#===----------------------------------------------------------------------===//
# LLVM/MLIR (in-tree)
#
# LLVM/MLIR is always built from the vendored external/llvm-project tree as
# part of this same CMake build via add_subdirectory.
#
# We manually populate the MLIR_*/LLVM_* discovery variables that
# find_package(MLIR) would normally set, so that Triton's CMake (which we
# patch to skip find_package when MLIRSupport already exists) and our own
# code keep working unchanged.
#===----------------------------------------------------------------------===//

if(NOT EXISTS "${ROCMLIR_LLVM_PROJECT_DIR}/llvm/CMakeLists.txt")
  message(FATAL_ERROR
    "external/llvm-project/llvm/CMakeLists.txt is missing.\n"
    "\n"
    "external/llvm-project is vendored via git subtree and should always be "
    "present in a normal checkout; if it is missing the working tree is "
    "incomplete.\n")
endif()

message(STATUS "Adding Triton-pinned LLVM/MLIR (external/llvm-project) src dependency")

# LLVM/MLIR build flags. Must be set BEFORE add_subdirectory so they take
# effect when LLVM processes its own CMakeLists.txt.
set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE STRING "List of LLVM sub-projects")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

# In-tree dev builds do not install MLIR; consumers (us, Triton) use the
# build tree directly. Skipping install(EXPORT MLIRTargets) also avoids
# CMake errors about Triton's first-class targets (TritonGPUTransforms,
# etc.) not being in MLIR's export set when our Rock libraries link them.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")

# Disable LLVM's PCH reuse machinery. When LLVM is in-tree, LLVM's
# CMake (see llvm/lib/Support/CMakeLists.txt and add_llvm_library
# PRECOMPILE_HEADERS) wires every library that links LLVMSupport to
# reuse LLVMSupport's PCH (`target_precompile_headers(... REUSE_FROM
# LLVMSupport)`), which propagates transitively to our Rock libraries
# and to mlir/lib/ExecutionEngine/conv-validation-wrappers.cpp. The PCH
# is compiled with LLVM's own flags (-std=c++17, no GNU extensions);
# if our targets compile with -std=gnu++17 (the CMake default for
# GCC/Clang when CMAKE_CXX_EXTENSIONS is left at its ON default),
# Clang refuses to load the PCH with "GNU extensions was disabled in
# AST file ... but is currently enabled".
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON CACHE BOOL "")

if(MLIR_ENABLE_ROCM_RUNNER)
  set(LLVM_TARGETS_TO_BUILD "X86;AMDGPU" CACHE STRING "")
else()
  set(LLVM_TARGETS_TO_BUILD "AMDGPU" CACHE STRING "")
endif()

# Matches rocMLIR approach in cmake/llvm-project.cmake.
# This is needed to compile rocmlirTriton and LLVM together.
# Before, we used to compile each project in a different build
# directory using a separate bash script. By doing a monolithic build,
# we avoid the bash script to handle the separate build process.
add_subdirectory("${ROCMLIR_LLVM_PROJECT_DIR}/llvm"
                 "external/llvm-project/llvm"
                 EXCLUDE_FROM_ALL)

# MLIR writes its build-tree config files to ${CMAKE_BINARY_DIR} (top-level),
# not to its own per-directory binary dir. See
# external/llvm-project/mlir/cmake/modules/CMakeLists.txt (the
# `mlir_cmake_builddir` variable).
#
# We point the package dirs at the build tree. LLVM_DIR must be cached because
# MLIRConfig.cmake's find_dependency(LLVM) consults the cache; without this it
# can discover ROCm SDK LLVM through CMAKE_PREFIX_PATH and mix incompatible LLVM
# headers with the in-tree MLIR headers.
set(MLIR_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")
# Cache MLIR_DIR so Triton's find_package(MLIR) uses the in-tree package on a
# clean configure. Mirrors the LLVM_DIR caching below.
set(MLIR_DIR "${MLIR_CMAKE_DIR}" CACHE PATH "Path to in-tree MLIR CMake package")

set(LLVM_LIBRARY_DIR "${LLVM_EXTERNAL_BUILD_DIR}/llvm/lib${LLVM_LIBDIR_SUFFIX}")
set(LLVM_CMAKE_DIR "${LLVM_LIBRARY_DIR}/cmake/llvm")
set(LLVM_DIR "${LLVM_CMAKE_DIR}" CACHE PATH "Path to in-tree LLVM CMake package" FORCE)
set(LLD_DIR "${LLVM_LIBRARY_DIR}/cmake/lld")

# find_package(MLIR) normally sets LLVM_TOOLS_BINARY_DIR. In the in-tree
# flow it isn't, so derive it from LLVM_EXTERNAL_BIN_DIR. Downstream code
# (e.g. mlir/test/CMakeLists.txt computing LLVM_LIT_TOOLS_DIR) depends on it.
set(LLVM_TOOLS_BINARY_DIR "${LLVM_EXTERNAL_BIN_DIR}")

list(APPEND MLIR_INCLUDE_DIRS
  "${ROCMLIR_LLVM_PROJECT_DIR}/mlir/include"
  "${LLVM_EXTERNAL_BUILD_DIR}/llvm/tools/mlir/include")
list(APPEND LLVM_INCLUDE_DIRS
  "${ROCMLIR_LLVM_PROJECT_DIR}/llvm/include"
  "${LLVM_EXTERNAL_BUILD_DIR}/llvm/include")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# NOTE: Upstream Triton's build_helpers.py would otherwise FORCE-overwrite
# MLIR_DIR to a ${LLVM_SYSPATH}-relative prebuilt-LLVM layout, which doesn't
# exist in our in-tree monolithic build. We sidestep that entirely by skipping
# build_helpers for in-tree builds (see the `if(NOT TARGET MLIRSupport)` guard
# in external/triton/CMakeLists.txt), so MLIR_DIR set above is never clobbered
# and Triton's find_package(MLIR CONFIG PATHS ${MLIR_DIR}) resolves directly.

message(STATUS "MLIR_DIR: ${MLIR_DIR}")
message(STATUS "LLD_DIR: ${LLD_DIR}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

#===----------------------------------------------------------------------===//
# ROCm Configuration
#
# Accepted inputs (first hit wins):
#   -DROCM_PATH=<dir>         (cache/command-line)
#   ENV{ROCM_PATH}=<dir>
#   ENV{HIP_PATH}=<dir>       (the HIP SDK installer sets this on Windows)
#   platform default          ("C:/opt/rocm" on Windows, "/opt/rocm" elsewhere)
#
# HIP package files may use the legacy hip/cmake module path or the standard
# lib/cmake/hip package-prefix layout used by the HIP SDK and TheRock.
#===----------------------------------------------------------------------===//

if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm / HIP SDK installation")
  elseif(WIN32 AND DEFINED ENV{HIP_PATH})
    set(ROCM_PATH "$ENV{HIP_PATH}" CACHE PATH "Path to ROCm / HIP SDK installation")
  elseif(WIN32)
    set(ROCM_PATH "C:/opt/rocm" CACHE PATH "Path to ROCm / HIP SDK installation")
  else()
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm / HIP SDK installation")
  endif()
endif()
file(TO_CMAKE_PATH "${ROCM_PATH}" ROCM_PATH)
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

# Preserve the legacy module path where it is supported.
if(NOT WIN32)
  list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")
endif()

#===----------------------------------------------------------------------===//
# Triton Build Options (matching external/triton/CMakeLists.txt)
#===----------------------------------------------------------------------===//

set(TRITON_BUILD_PYTHON_MODULE OFF CACHE BOOL "Don't build Python bindings")
set(TRITON_BUILD_PROTON OFF CACHE BOOL "Don't build Proton profiler")
set(TRITON_BUILD_UT OFF CACHE BOOL "Don't build Triton unit tests")
set(TRITON_CODEGEN_BACKENDS "amd" "nvidia"
    CACHE STRING "Triton codegen backends to enable")

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
# LLVM_SYSPATH
# Triton uses our in-tree LLVM targets directly, so LLVM_SYSPATH is unused but
# harmless to set; point it at the in-tree LLVM build dir.
#===----------------------------------------------------------------------===//

if(NOT LLVM_SYSPATH)
  if(DEFINED ENV{LLVM_SYSPATH})
    set(LLVM_SYSPATH "$ENV{LLVM_SYSPATH}"
        CACHE PATH "Path to LLVM install used by Triton")
  else()
    set(LLVM_SYSPATH "${LLVM_EXTERNAL_BUILD_DIR}/llvm"
        CACHE PATH "Path to LLVM install used by Triton")
  endif()
endif()
message(STATUS "LLVM_SYSPATH: ${LLVM_SYSPATH}")

#===----------------------------------------------------------------------===//
# Add Triton subdirectory
#===----------------------------------------------------------------------===//

add_subdirectory("${TRITON_PROJECT_DIR}" "external/triton" EXCLUDE_FROM_ALL)

# Upstream Triton's bin targets (triton-opt, triton-reduce, triton-lsp,
# triton-tensor-layout) unconditionally link the TritonTest* pass libraries and
# register their passes (bin/RegisterTritonDialects.h). Those libraries are in
# external/triton/test/lib, which upstream only adds under TRITON_BUILD_UT.
# Turning UT on would trigger FetchContent() googletest for the C++ unit tests.
# We keep UT OFF and instead build just test/lib here.
if(TRITON_BUILD_BINARY AND NOT TRITON_BUILD_UT)
  if(NOT MSVC)
    set(TRITON_DISABLE_EH_RTTI_FLAGS "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
  endif()
  include_directories(
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${TRITON_INCLUDE_DIRS}
  )
  add_subdirectory("${TRITON_PROJECT_DIR}/test/lib"
                   "external/triton/rocmlir-test-lib" EXCLUDE_FROM_ALL)
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

