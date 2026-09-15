set(LLVM_LINK_COMPONENTS
  Support
  )

rocmlir_find_hip()

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  MLIRAnalysis
  MLIRFuncInlinerExtension

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
  TritonInstrumentTransforms
  )

add_rocmlir_tool(rocmlir-gen
  PARTIAL_SOURCES_INTENDED

  rocmlir-gen.cpp

  DEPENDS
  ${LIBS}
)


target_link_libraries(rocmlir-gen PRIVATE ${LIBS} MLIRRockDeviceInfo)
llvm_update_compile_flags(rocmlir-gen)

rocmlir_add_triton_includes(rocmlir-gen)

mlir_check_link_libraries(rocmlir-gen)
