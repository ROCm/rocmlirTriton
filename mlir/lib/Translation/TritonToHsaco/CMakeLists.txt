llvm_canonicalize_cmake_booleans(
    LLVM_HAS_AMDGPU_TARGET
)

# Find LLD package for linking AMDGPU objects. When LLD is added in-tree via
# add_subdirectory(llvm-project/llvm) (the standalone monolithic build),
# find_package would collide on imported targets, so detect the in-tree case
# via TARGET lldELF instead.
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  if(TARGET lldELF)
    set(LLD_FOUND TRUE)
  else()
    find_package(LLD CONFIG QUIET)
    if(NOT LLD_FOUND)
      message(SEND_ERROR "LLD is not found, please ensure LLD was built with LLVM")
    endif()
  endif()
endif()

add_mlir_translation_library(TritonToHsacoTranslation
  TritonToHsaco.cpp

  DEPENDS
  MLIRRockPassIncGen

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
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD AND LLD_FOUND)
  target_link_libraries(TritonToHsacoTranslation
    PRIVATE
    lldELF
    lldCommon
  )
endif()

rocmlir_add_triton_includes(TritonToHsacoTranslation)

# Define path to Triton's bundled device libraries (ocml.bc, ockl.bc)
target_compile_definitions(obj.TritonToHsacoTranslation PRIVATE
  TRITON_AMD_BACKEND_LIB_DIR="${TRITON_PROJECT_DIR}/third_party/amd/backend/lib"
)

# Add LLD include directory for compilation (header access). Look in both the
# source-tree layout (llvm-project/lld/include, used when LLVM is built in-tree)
# and the install-tree layout (<prefix>/include/lld, used when LLVM is consumed
# via find_package from an installed package).
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
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
endif()
