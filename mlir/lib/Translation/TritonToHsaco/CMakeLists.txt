if(TARGET lldELF)
  set(LLD_FOUND TRUE)
else()
  find_package(LLD CONFIG QUIET)
  if(NOT LLD_FOUND)
    message(SEND_ERROR "LLD is not found, please ensure LLD was built with LLVM")
  endif()
endif()

if(WIN32)
  include(FetchContent)
  FetchContent_Declare(
    dlfcn_win32
    GIT_REPOSITORY https://github.com/dlfcn-win32/dlfcn-win32
    GIT_TAG v1.4.2
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(dlfcn_win32)

  set(ROCMLIR_DLFCN_TARGET "")
  if(TARGET dlfcn-win32::dl)
    set(ROCMLIR_DLFCN_TARGET dlfcn-win32::dl)
  elseif(TARGET dlfcn_win32::dl)
    set(ROCMLIR_DLFCN_TARGET dlfcn_win32::dl)
  elseif(TARGET dl)
    set(ROCMLIR_DLFCN_TARGET dl)
  elseif(TARGET dlfcn-win32)
    set(ROCMLIR_DLFCN_TARGET dlfcn-win32)
  endif()

  if(NOT ROCMLIR_DLFCN_TARGET)
    message(FATAL_ERROR "Unable to find dlfcn-win32 CMake target after FetchContent")
  endif()

  # This in-tree FetchContent build inherits LLVM's /WX. dlfcn-win32 1.4.2
  # compares a signed index with unsigned directory counts in dlfcn.c, and
  # MSVC then rejects the file. The warning is in the third-party source.
  if(MSVC AND TARGET dl)
    target_compile_options(dl PRIVATE /WX-)
  endif()
endif()

set(TRITON_AMD_DEVICE_LIB_DIR
  "${TRITON_PROJECT_DIR}/third_party/amd/backend/lib")
set(TRITON_AMD_DEVICE_LIB_NAMES ocml ockl)
set(TRITON_AMD_DEVICE_LIB_FILES "")
foreach(_lib IN LISTS TRITON_AMD_DEVICE_LIB_NAMES)
  set(_lib_file "${TRITON_AMD_DEVICE_LIB_DIR}/${_lib}.bc")
  if(NOT EXISTS "${_lib_file}")
    message(FATAL_ERROR "Triton AMD device bitcode not found: ${_lib_file}")
  endif()
  list(APPEND TRITON_AMD_DEVICE_LIB_FILES "${_lib_file}")
endforeach()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(TRITON_AMD_DEVICE_LIBS_INC
  "${CMAKE_CURRENT_BINARY_DIR}/TritonAmdDeviceLibs.cpp.inc")
add_custom_command(
  OUTPUT "${TRITON_AMD_DEVICE_LIBS_INC}"
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/TritonAmdDeviceLibsIncGen.py"
    "${TRITON_AMD_DEVICE_LIBS_INC}"
    "${TRITON_AMD_DEVICE_LIB_DIR}"
    ${TRITON_AMD_DEVICE_LIB_NAMES}
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/TritonAmdDeviceLibsIncGen.py"
    ${TRITON_AMD_DEVICE_LIB_FILES}
  COMMENT "Embedding Triton AMD device libraries"
  VERBATIM
)
add_custom_target(TritonAmdDeviceLibsIncGen
  DEPENDS "${TRITON_AMD_DEVICE_LIBS_INC}")

add_mlir_translation_library(TritonToHsacoTranslation
  TritonToHsaco.cpp

  DEPENDS
  MLIRRockPassIncGen
  TritonAmdDeviceLibsIncGen
  triton_amd_codegen_lib

  LINK_COMPONENTS
  Core
  IRReader
  Linker
  Passes
  Target
  MC
  MCParser
  AsmParser
  AMDGPUCodeGen
  AMDGPUAsmParser
  AMDGPUDesc
  AMDGPUInfo
  TransformUtils
  Instrumentation
  # X86 targets needed for InitializeAllTargets()
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRPass
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRIR
  MLIRROCDLDialect
  MLIRRockOps
  MLIRRockUtility

  # Triton libraries
  TritonAMDGPUToLLVM
  TritonLLVMIR  # For BreakStructPhiNodesPass
)

# Link LLD libraries for AMDGPU linking
target_link_libraries(TritonToHsacoTranslation
  PRIVATE
  lldELF
  lldCommon
)

if(WIN32)
  target_link_libraries(TritonToHsacoTranslation
    PRIVATE
    ${ROCMLIR_DLFCN_TARGET}
  )
endif()

rocmlir_add_triton_includes(TritonToHsacoTranslation)

target_include_directories(obj.TritonToHsacoTranslation PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}")

if(WIN32)
  get_target_property(_rocmlir_dlfcn_includes ${ROCMLIR_DLFCN_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  if(_rocmlir_dlfcn_includes)
    target_include_directories(obj.TritonToHsacoTranslation PRIVATE ${_rocmlir_dlfcn_includes})
  endif()
endif()

# Add LLD include directory for compilation (header access). Look in both the
# source-tree layout (llvm-project/lld/include, used when LLVM is built in-tree)
# and the install-tree layout (<prefix>/include/lld, used when LLVM is consumed
# via find_package from an installed package).
set(LLD_INCLUDE_CANDIDATES "")
# rocmlirTriton in-tree build: lld is vendored next to llvm/ in
# external/llvm-project, so its headers live at <src>/lld/include. Prefer
# this explicit source path. We cannot rely on LLVM_INCLUDE_DIRS[0] pointing
# at the source tree here: Triton's own CMake runs build_helpers.py and
# FORCE-overwrites LLVM_INCLUDE_DIRS with the build-tree include dir only,
# so the LLVM_INCLUDE_DIRS-based derivation below would miss lld/include.
if(DEFINED LLVM_EXTERNAL_SRC_DIR)
  list(APPEND LLD_INCLUDE_CANDIDATES "${LLVM_EXTERNAL_SRC_DIR}/lld/include")
endif()
list(GET LLVM_INCLUDE_DIRS 0 LLVM_FIRST_INCLUDE_DIR)
get_filename_component(LLVM_SRC_DIR "${LLVM_FIRST_INCLUDE_DIR}" DIRECTORY)
get_filename_component(LLVM_PROJECT_DIR "${LLVM_SRC_DIR}" DIRECTORY)
list(APPEND LLD_INCLUDE_CANDIDATES
  "${LLVM_PROJECT_DIR}/lld/include"
  "${LLVM_FIRST_INCLUDE_DIR}/lld"
  "${LLVM_FIRST_INCLUDE_DIR}"
)
set(LLD_INCLUDE_DIR_TRITON "")
foreach(_cand IN LISTS LLD_INCLUDE_CANDIDATES)
  if(EXISTS "${_cand}/lld/Common/Driver.h" OR EXISTS "${_cand}/Common/Driver.h")
    set(LLD_INCLUDE_DIR_TRITON "${_cand}")
    break()
  endif()
endforeach()

if(LLD_INCLUDE_DIR_TRITON)
  message(STATUS "TritonToHsaco: Found LLD include dir at ${LLD_INCLUDE_DIR_TRITON}")
  target_include_directories(obj.TritonToHsacoTranslation PRIVATE ${LLD_INCLUDE_DIR_TRITON})
else()
  message(WARNING "TritonToHsaco: LLD include dir not found in ${LLD_INCLUDE_CANDIDATES}")
endif()
