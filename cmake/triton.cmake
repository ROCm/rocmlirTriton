include(cmake/submodules.cmake)
include(cmake/patches.cmake)

message(STATUS "Adding Triton src dependency")

#===----------------------------------------------------------------------===//
# Paths for building rocmlirTriton
#===----------------------------------------------------------------------===//

set(TRITON_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/triton")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/triton")
set(ROCMLIRTRITON_LLVM_PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llvm-project")

option(ROCMLIRTRITON_SKIP_SUBMODULE_UPDATE "Skip 'git submodule update'" OFF)

# Ensure submodules are available and up-to-date.
rocmlir_ensure_submodules()
# Apply the downstream patches.
rocmlir_apply_patches()

#===----------------------------------------------------------------------===//
# LLVM/MLIR discovery
#
# Two paths are supported, tried in this order:
#
#   (imported)  An externally-built LLVM is provided via MLIR_DIR (cache or
#               -D), the MLIR_DIR env var, CMAKE_PREFIX_PATH (used by
#               MIGraphX flow).We import it via find_package(MLIR CONFIG).
#
#   (in-tree)   No external LLVM was found. We add_subdirectory the vendored
#               external/llvm-project/llvm/, making LLVM part
#               of the same CMake build. This is the standalone build
#               flow.
#
# In the in-tree flow we manually populate the MLIR_*/LLVM_* discovery
# variables that find_package(MLIR) would normally set, so that Triton's
# CMake (which we patch to skip find_package when MLIRSupport already
# exists) and our own code keep working unchanged.
#===----------------------------------------------------------------------===//

# find_package(MLIR) caches MLIR_DIR=MLIR_DIR-NOTFOUND on failure; a plain
# CACHE set() does not override that, and later configures skip discovery
# because MLIR_DIR is already DEFINED. Clear stale -NOTFOUND so re-configure
# can re-discover.
if(MLIR_DIR MATCHES "-NOTFOUND$"
   OR (MLIR_DIR AND NOT EXISTS "${MLIR_DIR}/MLIRConfig.cmake"))
  unset(MLIR_DIR CACHE)
endif()

set(_rocmlir_legacy_in_tree_mlir_dir
    "${ROCMLIRTRITON_LLVM_PROJECT_DIR}/build/lib/cmake/mlir")

set(_rocmlir_have_external_mlir FALSE)
if(MLIR_DIR)
  set(_rocmlir_have_external_mlir TRUE)
elseif(DEFINED ENV{MLIR_DIR})
  set(MLIR_DIR "$ENV{MLIR_DIR}" CACHE PATH "Path to MLIR CMake config" FORCE)
  set(_rocmlir_have_external_mlir TRUE)
else()
  find_package(MLIR QUIET CONFIG)
  if(MLIR_FOUND)
    set(MLIR_DIR "${MLIR_DIR}" CACHE PATH "Path to MLIR CMake config" FORCE)
    set(_rocmlir_have_external_mlir TRUE)
  elseif(EXISTS "${_rocmlir_legacy_in_tree_mlir_dir}/MLIRConfig.cmake")
    set(MLIR_DIR "${_rocmlir_legacy_in_tree_mlir_dir}"
        CACHE PATH "Path to MLIR CMake config" FORCE)
    set(_rocmlir_have_external_mlir TRUE)
  endif()
endif()

if(_rocmlir_have_external_mlir)
  #===--------------------------------------------------------------------===//
  # Imported MLIR (MIGraphX flow)
  #===--------------------------------------------------------------------===//
  message(STATUS "Using externally-built MLIR (MLIR_DIR=${MLIR_DIR})")
  find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

  get_filename_component(_llvm_cmake_dir "${MLIR_DIR}" DIRECTORY)
  set(LLD_DIR "${_llvm_cmake_dir}/lld" CACHE PATH "Path to LLD CMake config")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
else()
  #===--------------------------------------------------------------------===//
  # In-tree (rocmlirTriton standalone flow)
  #===--------------------------------------------------------------------===//
  if(NOT EXISTS "${ROCMLIRTRITON_LLVM_PROJECT_DIR}/llvm/CMakeLists.txt")
    message(FATAL_ERROR
      "external/llvm-project/llvm/CMakeLists.txt is missing and no "
      "externally-built MLIR was found.\n"
      "\n"
      "Initialize the LLVM submodule:\n"
      "  git submodule update --init external/llvm-project\n"
      "\n"
      "Or, to point at an existing LLVM/MLIR install, set one of:\n"
      "  -DMLIR_DIR=/path/to/lib/cmake/mlir\n"
      "  -DCMAKE_PREFIX_PATH=/path/to/llvm-install\n"
      "  export MLIR_DIR=/path/to/lib/cmake/mlir\n"
      "\n")
  endif()

  message(STATUS "Adding LLVM/MLIR git-submodule src dependency")

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
  add_subdirectory("${ROCMLIRTRITON_LLVM_PROJECT_DIR}/llvm"
                   "external/llvm-project/llvm"
                   EXCLUDE_FROM_ALL)

  # MLIR writes its build-tree config files to ${CMAKE_BINARY_DIR} (top-level),
  # not to its own per-directory binary dir. See
  # external/llvm-project/mlir/cmake/modules/CMakeLists.txt (the
  # `mlir_cmake_builddir` variable).
  #
  # Intentionally NOT caching MLIR_DIR / LLD_DIR here. Caching them would
  # make subsequent configures see "MLIR_DIR is set" and take the imported
  # branch above, which would call find_package against the in-tree
  # MLIRConfig.cmake instead of re-running add_subdirectory(llvm), and LLVM
  # would silently drop out of the build graph after the first configure.
  set(MLIR_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")
  set(MLIR_DIR "${MLIR_CMAKE_DIR}")

  set(LLVM_LIBRARY_DIR "${LLVM_EXTERNAL_BUILD_DIR}/llvm/lib${LLVM_LIBDIR_SUFFIX}")
  set(LLVM_CMAKE_DIR "${LLVM_LIBRARY_DIR}/cmake/llvm")
  set(LLD_DIR "${LLVM_LIBRARY_DIR}/cmake/lld")

  # find_package(MLIR) normally sets LLVM_TOOLS_BINARY_DIR. In the in-tree
  # flow it isn't, so derive it from LLVM_EXTERNAL_BIN_DIR. Downstream code
  # (e.g. mlir/test/CMakeLists.txt computing LLVM_LIT_TOOLS_DIR) depends on it.
  set(LLVM_TOOLS_BINARY_DIR "${LLVM_EXTERNAL_BIN_DIR}")

  list(APPEND MLIR_INCLUDE_DIRS
    "${ROCMLIRTRITON_LLVM_PROJECT_DIR}/mlir/include"
    "${LLVM_EXTERNAL_BUILD_DIR}/llvm/tools/mlir/include")
  list(APPEND LLVM_INCLUDE_DIRS
    "${ROCMLIRTRITON_LLVM_PROJECT_DIR}/llvm/include"
    "${LLVM_EXTERNAL_BUILD_DIR}/llvm/include")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

message(STATUS "MLIR_DIR: ${MLIR_DIR}")
message(STATUS "LLD_DIR: ${LLD_DIR}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

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
# In the imported flow, this tells Triton where its LLVM lives so it does not
# try to download a prebuilt LLVM tarball. In the in-tree flow Triton uses
# our in-tree targets directly; LLVM_SYSPATH is unused but harmless to set.
#===----------------------------------------------------------------------===//

if(NOT LLVM_SYSPATH)
  if(DEFINED ENV{LLVM_SYSPATH})
    set(LLVM_SYSPATH "$ENV{LLVM_SYSPATH}"
        CACHE PATH "Path to LLVM install used by Triton")
  elseif(_rocmlir_have_external_mlir)
    # MLIR_DIR is .../llvm-install/lib/cmake/mlir → prefix is .../llvm-install
    get_filename_component(_mlir_parent "${MLIR_DIR}" DIRECTORY)
    get_filename_component(_mlir_parent "${_mlir_parent}" DIRECTORY)
    get_filename_component(_llvm_prefix "${_mlir_parent}" DIRECTORY)
    set(LLVM_SYSPATH "${_llvm_prefix}"
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

