set(LLVM_OPTIONAL_SOURCES
  rocmlir-tuning-driver.cpp
  )

if(MLIR_ENABLE_ROCM_RUNNER)

get_property(rocmlir_dialect_libs GLOBAL PROPERTY ROCMLIR_DIALECT_LIBS)
get_property(rocmlir_conversion_libs GLOBAL PROPERTY ROCMLIR_CONVERSION_LIBS)

set(LIBS
  ${rocmlir_dialect_libs}
  ${rocmlir_conversion_libs}
  MLIRParser
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

add_rocmlir_tool(rocmlir-tuning-driver
  rocmlir-tuning-driver.cpp
  ArtifactIO.cpp
  CacheFlush.cpp
)

# Grab HIP again, since we'll be using it directly.
rocmlir_find_hip(HIPRTC)

# Supress compiler warnings from HIP headers
# Only set compile options on mlir_rocm_runtime if it exists and is not an IMPORTED target
# (i.e., when building LLVM/MLIR from source rather than using a pre-built version)
if(TARGET mlir_rocm_runtime)
  get_target_property(_mlir_rocm_runtime_imported mlir_rocm_runtime IMPORTED)
endif()
if(TARGET mlir_rocm_runtime AND NOT _mlir_rocm_runtime_imported)
  check_cxx_compiler_flag(-Wno-c++98-compat-extra-semi
    CXX_SUPPORTS_NO_CXX98_COMPAT_EXTRA_SEMI_FLAG)
  if (CXX_SUPPORTS_NO_CXX98_COMPAT_EXTRA_SEMI_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
      "-Wno-c++98-compat-extra-semi")
  endif()
  check_cxx_compiler_flag(-Wno-return-type-c-linkage
      CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
  if (CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
      "-Wno-return-type-c-linkage")
  endif()
  check_cxx_compiler_flag(-Wno-nested-anon-types
    CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
  if (CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
      "-Wno-nested-anon-types")
  endif()
  check_cxx_compiler_flag(-Wno-gnu-anonymous-struct
    CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
  if (CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
    target_compile_options(mlir_rocm_runtime PRIVATE
      "-Wno-gnu-anonymous-struct")
  endif()
endif()

set_property(TARGET rocmlir-tuning-driver
  PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

# MLIRRockDeviceInfo is deliberately not a dialect library, so ${LIBS} does not
# cover it.
target_link_libraries(rocmlir-tuning-driver PRIVATE ${LIBS} MLIRRockDeviceInfo)
target_link_libraries(rocmlir-tuning-driver PUBLIC benchmark-driver-utils)
target_link_libraries(rocmlir-tuning-driver PUBLIC hip::host hip::amdhip64 hiprtc::hiprtc)
llvm_update_compile_flags(rocmlir-tuning-driver)

rocmlir_add_triton_includes(rocmlir-tuning-driver)

endif()
