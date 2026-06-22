# Test-only binding. Never linked into librockCompiler.
pybind11_add_module(amd_arch_db MODULE binding.cpp)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

target_link_libraries(amd_arch_db PRIVATE
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  ${rocmlir_test_libs}
  MLIROptLib
  MLIRAnalysis
  MLIRFuncInlinerExtension
  MLIRVectorToLLVM
  MLIRVectorToLLVMPass
  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonGPUTransforms
  TritonAMDGPUToLLVM
  TritonAMDGPUTransforms
  TritonAMDAnalysis
  TritonAMDGPUDialectToLLVM
  TritonAMDUtils
  GluonTransforms
  TritonInstrumentToLLVM
  NVHopperTransforms
  ProtonToProtonGPU
  ProtonGPUToLLVM
  ProtonNVIDIAGPUToLLVM
  ProtonAMDGPUToLLVM
  ProtonGPUTransforms
  TritonLLVMIR
  NVGPUToLLVM
  TritonInstrumentTransforms)

rocmlir_add_triton_includes(amd_arch_db)

# This module statically links LLVM/MLIR (from Triton's llvm-project), but it
# also transitively pulls in the ROCm runtime (libamd_comgr -> libLLVM.so /
# libclang-cpp.so) via MLIRRockUtility's HIP dependency. Both ship the same
# LLVM soname, so if this module exported its statically-linked llvm::cl::*
# symbols, the ROCm libclang-cpp static initializers would bind to them and
# trip an `llvm::cl` SmallPtrSet assertion at import time. Hide all symbols
# coming from static archives (pybind still exports the PyInit_ entry point)
# so libclang-cpp resolves against its own libLLVM.so instead.
if (NOT WIN32)
  target_link_options(amd_arch_db PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# Drop the .so next to common.py so `import amd_arch_db` just works.
set_target_properties(amd_arch_db PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/..")
