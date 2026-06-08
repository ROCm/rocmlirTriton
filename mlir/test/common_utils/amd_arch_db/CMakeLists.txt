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

# Drop the .so next to common.py so `import amd_arch_db` just works.
set_target_properties(amd_arch_db PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/..")
