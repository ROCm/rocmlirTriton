set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)
get_property(rocmlir_test_libs GLOBAL PROPERTY ROCMLIR_TEST_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
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
  TritonInstrumentTransforms
  )

# Only link MLIRTestIR when the test libraries actually exist as imported
# targets. MLIR_INCLUDE_TESTS=ON only means "the upstream MLIR build included
# tests"; it does *not* guarantee those targets are exported -- and they
# aren't, when MLIR is consumed via an installed package (e.g. cget).
if(MLIR_INCLUDE_TESTS AND TARGET MLIRTestIR)
  list(APPEND LIBS MLIRTestIR)
endif()

set(disable_install "")
if (BUILD_FAT_LIBROCKCOMPILER)
  set(disable_install "DISABLE_INSTALL")
endif()

add_mlir_library(RocMLIROptMain
  rocmlir-opt.cpp

  EXCLUDE_FROM_LIBMLIR
  ${disable_install}

  LINK_LIBS PRIVATE
  ${LIBS}
  )
set_target_properties(RocMLIROptMain PROPERTIES EXCLUDE_FROM_ALL ${BUILD_FAT_LIBROCKCOMPILER})

add_rocmlir_tool(rocmlir-opt
  rocmlir-opt.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(rocmlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(rocmlir-opt)

rocmlir_add_triton_includes(rocmlir-opt)

mlir_check_all_link_libraries(rocmlir-opt)
