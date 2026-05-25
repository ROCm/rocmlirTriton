message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_BUILD_LIBRARY_DIR: ${LLVM_BUILD_LIBRARY_DIR}")
message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Strict warnings for our code only. These are directory-scoped so they
# don't leak into the Triton subdirectory; LLVM/MLIR headers are consumed 
# via find_package and aren't built here, so they're unaffected as well.
#
# -Werror is wired through LLVM's HandleLLVMOptions, so it must be set BEFORE
# the include() below.
set(LLVM_ENABLE_WERROR ON)
include(HandleLLVMOptions)

# HandleLLVMOptions appends -Wl,-z,defs to CMAKE_SHARED_LINKER_FLAGS when
# building shared libs on ELF (the default standalone configuration). That
# is great hygiene but exposes pre-existing latent missing-LINK_LIBS in our
# graph -- e.g. MLIRRockOps invoking rock::* utilities from MLIRRockUtility
# without declaring the dependency, MLIRRockOps using
# triton::AMD::TargetInfo without linking TritonAMDGPUToLLVM, etc. The
# develop branch never hit this because the legacy build flow consumed an
# already-installed MLIR via find_package, which does not run
# HandleLLVMOptions on a built tree and so does not set -Wl,-z,defs.
#
# Strip the flag so the in-tree monolithic build matches the develop-branch
# behaviour. The latent dep bugs should be fixed properly (either by
# declaring the missing LINK_LIBS, accepting the resulting CMake cycle,
# or by lifting the utility helpers used by MLIRRockOps into MLIRRockOps
# itself). Tracked separately; see docs/rocmlirtriton_build.md.
string(REPLACE " -Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
string(REPLACE " -Wl,-z,defs" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)

# Match Triton: -Wcovered-switch-default (on by default in Clang/LLVM) is
# noisy on MLIR-style enums where a default: is intentionally present.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-covered-switch-default>)

# Extras on top of -Wall/-Wextra and the LLVM warning set.
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
  $<$<COMPILE_LANGUAGE:CXX>:-Wnull-dereference>
  $<$<COMPILE_LANGUAGE:CXX>:-Wformat=2>
  $<$<COMPILE_LANGUAGE:CXX>:-Wundef>
  $<$<COMPILE_LANGUAGE:CXX>:-Wmissing-declarations>
)

# Refer to https://reviews.llvm.org/D122088
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Treat upstream LLVM/MLIR headers as system headers so our strict warnings
# (-Wshadow, -Wnull-dereference, etc.) don't fire on them.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
# Generated TableGen output (*.h.inc, *.cpp.inc) goes under the build tree.
# Treat it as system so the auto-generated code doesn't fail our lint.
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/mlir/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR})
set(LLVM_VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR})
set(LLVM_VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH})
set(ARG_PRODUCT_NAME ${CMAKE_PROJECT_NAME} CACHE STRING "")
set(ARG_VERSION_STRING ${CMAKE_PROJECT_VERSION} CACHE STRING "")

add_subdirectory(include/mlir)
add_subdirectory(lib)
# C API needs all dialects for registration, but should be built before tests.
add_subdirectory(lib/CAPI)

if (MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_custom_target(RocMLIRUnitTests)

  # If LLVM_THIRD_PARTY_DIR is not set or doesn't contain gtest, try Triton's LLVM
  if (NOT EXISTS "${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h")
    set(TRITON_LLVM_THIRD_PARTY "${PROJECT_SOURCE_DIR}/external/llvm-project/third-party")
    if (EXISTS "${TRITON_LLVM_THIRD_PARTY}/unittest/googletest/include/gtest/gtest.h")
      set(LLVM_THIRD_PARTY_DIR "${TRITON_LLVM_THIRD_PARTY}")
      message(STATUS "Using gtest from Triton LLVM: ${LLVM_THIRD_PARTY_DIR}")
    endif()
  endif()

  if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
  else()
    message(WARNING "gtest not found, unittests will not be available LLVM_THIRD_PARTY_DIR=${LLVM_THIRD_PARTY_DIR}")
  endif()
  add_subdirectory(test)
endif()
add_subdirectory(tools)
add_subdirectory(utils)
