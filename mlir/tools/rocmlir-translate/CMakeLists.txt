set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_rocmlir_tool(rocmlir-translate
  rocmlir-translate.cpp
  )
target_link_libraries(rocmlir-translate
  PRIVATE
  ${translation_libs}
  ${test_libs}
  MLIRROCDLToLLVMIRTranslation
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

llvm_update_compile_flags(rocmlir-translate)

rocmlir_add_triton_includes(rocmlir-translate)

mlir_check_link_libraries(rocmlir-translate)
