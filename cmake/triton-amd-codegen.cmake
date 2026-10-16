#===----------------------------------------------------------------------===//
# Standalone Triton AMD code-generation library (triton_amd_codegen)
#
# Triton's AMD backend ships a small shared library, triton_amd_codegen, that is
# loaded at runtime via ctypes/dlopen. It is deliberately built
# against its OWN, independently pinned LLVM -- different from the Triton-core
# LLVM built in cmake/triton.cmake -- so the AMDGPU code generator's cl::opt
# registry stays isolated from the LLVM linked into librockCompiler.
#
# Here we build with two ExternalProject_Add steps that run as
# nested builds:
#
#   1. amd_codegen_llvm   -- configure/build a minimal AMDGPU-only LLVM.
#   2. triton_amd_codegen -- configure/build the standalone library
#                            (third_party/amd/backend/codegen) against (1).
#===----------------------------------------------------------------------===//

include(ExternalProject)

set(ROCMLIR_TRITON_AMD_CODEGEN_LIBRARY "" CACHE FILEPATH
	"Path to the built triton_amd_codegen library")

if(ROCMLIR_TRITON_AMD_CODEGEN_LIBRARY)
  return()
endif()

# AMD code-generation LLVM from Triton cmake/amd-llvm-build-info.json
set(ROCMLIR_AMD_CODEGEN_LLVM_GIT_REPOSITORY "https://github.com/ROCm/llvm-project"
    CACHE STRING "Git repository for the standalone AMD code-generation LLVM")
set(ROCMLIR_AMD_CODEGEN_LLVM_GIT_TAG "rmt-amd"
    CACHE STRING "Git branch/tag/commit for the standalone AMD code-generation LLVM")
set(ROCMLIR_AMD_CODEGEN_LLVM_REVISION "${ROCMLIR_AMD_CODEGEN_LLVM_GIT_TAG}-1"
    CACHE STRING
    "Revision string baked into triton_amd_codegen (triton_amdgpu_revision())")

message(STATUS "Fetching AMD Codegen LLVM/MLIR: ${ROCMLIR_AMD_CODEGEN_LLVM_GIT_REPOSITORY}@${ROCMLIR_AMD_CODEGEN_LLVM_GIT_TAG}")

set(amd_codegen_root "${CMAKE_CURRENT_BINARY_DIR}/amd-codegen")

foreach(_var
	CMAKE_C_COMPILER CMAKE_CXX_COMPILER
    CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER)
  if(${_var})
    list(APPEND _amd_codegen_tool_args "-D${_var}=${${_var}}")
  endif()
endforeach()

if(WIN32)
  set(triton_amd_codegen_libname "triton_amd_codegen.dll")
else()
  set(triton_amd_codegen_libname "libtriton_amd_codegen.so")
endif()

# Build the AMD code-generation LLVM (AMDGPU target only, static libs).
ExternalProject_Add(amd_codegen_llvm
  GIT_REPOSITORY "${ROCMLIR_AMD_CODEGEN_LLVM_GIT_REPOSITORY}"
  GIT_TAG "${ROCMLIR_AMD_CODEGEN_LLVM_GIT_TAG}"
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  PREFIX "${amd_codegen_root}/llvm"
  SOURCE_SUBDIR llvm
  CMAKE_ARGS
    ${_amd_codegen_tool_args}
	"-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    -DLLVM_ENABLE_PROJECTS=
    -DLLVM_TARGETS_TO_BUILD=AMDGPU
    -DLLVM_BUILD_UTILS=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DLLVM_ENABLE_ZSTD=OFF
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "<BINARY_DIR>/lib/cmake/llvm/LLVMConfig.cmake"
  USES_TERMINAL_DOWNLOAD TRUE
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE)

ExternalProject_Get_Property(amd_codegen_llvm BINARY_DIR)

# Build the standalone triton_amd_codegen library against the LLVM
ExternalProject_Add(triton_amd_codegen
  DEPENDS amd_codegen_llvm
  DOWNLOAD_COMMAND ""
  SOURCE_DIR "${TRITON_PROJECT_DIR}/third_party/amd/backend/codegen"
  PREFIX "${amd_codegen_root}/codegen"
  CMAKE_ARGS
    ${_amd_codegen_tool_args}
	"-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
	"-DLLVM_DIR=${BINARY_DIR}/lib/cmake/llvm"
    "-DTRITON_AMD_LLVM_REVISION=${ROCMLIR_AMD_CODEGEN_LLVM_REVISION}"
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "<BINARY_DIR>/${triton_amd_codegen_libname}"
  USES_TERMINAL_CONFIGURE TRUE
  USES_TERMINAL_BUILD TRUE)

ExternalProject_Get_Property(triton_amd_codegen BINARY_DIR)
set(ROCMLIR_TRITON_AMD_CODEGEN_LIBRARY "${BINARY_DIR}/${triton_amd_codegen_libname}")

message(STATUS "triton_amd_codegen: ${ROCMLIR_TRITON_AMD_CODEGEN_LIBRARY}")
