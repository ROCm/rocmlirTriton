# Test-only binding. Never linked into librockCompiler.
pybind11_add_module(amd_arch_db MODULE binding.cpp)

# pybind11's PYBIND11_MODULE macro uses try/catch and typeid. clang-cl (the
# Windows toolchain) defaults to exceptions and RTTI disabled (inherited from
# the LLVM/MLIR build flags), so re-enable both for this binding. On Linux
# clang already enables exceptions by default, so this is Windows-only.
if(WIN32)
  target_compile_options(amd_arch_db PRIVATE /EHsc /GR)
endif()

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

# MLIRRockDeviceInfo is deliberately not a dialect library (it links HIP and
# must stay out of librockCompiler), so it has to be listed explicitly.
rocmlir_find_hip()

target_link_libraries(amd_arch_db PRIVATE
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  ${rocmlir_test_libs}
  MLIRRockDeviceInfo
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

# This module statically links LLVM/MLIR (from Triton's llvm-project). Hide
# those statically-linked symbols (the ~1300 default-visibility llvm::cl::* /
# pass-registry / ManagedStatic data symbols) instead of re-exporting them from
# the Python module. If the importing process also has a second LLVM in it (for
# example ROCm's libLLVM.so / libclang-cpp.so, reached through some other
# extension's libamd_comgr), two same-soname LLVMs would otherwise share these
# cl::opt singletons and trip an `llvm::cl` SmallPtrSet assertion at import
# time. --exclude-libs,ALL localizes the static-archive symbols while pybind
# still exports the PyInit_ entry point. It is a GNU-ld/lld option (invalid on
# Apple ld), so gate it on Linux to match conv-validation-wrappers, the global
# -Bsymbolic-functions option, and the mlir-rocm-runtime llvm-patch.
target_link_options(amd_arch_db PRIVATE
  $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>)

# Drop the .so next to common.py so `import amd_arch_db` just works.
set_target_properties(amd_arch_db PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/..")
