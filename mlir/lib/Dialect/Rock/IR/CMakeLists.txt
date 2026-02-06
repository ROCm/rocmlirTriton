
add_rocmlir_dialect_library(MLIRRockOps
  AmdArchDb.cpp
  GetRockInfo.cpp
  TransformMapBuilder.cpp
  RockDialect.cpp
  RockGemmWrapperInterface.cpp
  RockGemmGemmWrapperInterface.cpp
  RockConvInterface.cpp
  RockTuningParamAttrInterface.cpp
  RockAcceptingViewOpInterface.cpp
  RockWriterOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rock

  DEPENDS
  MLIRRockAttrDefsIncGen
  MLIRRockGemmWrapperInterfaceIncGen
  MLIRRockGemmGemmWrapperInterfaceIncGen
  MLIRRockTuningParamAttrInterfaceIncGen
  MLIRRockAcceptingViewOpInterfaceIncGen
  MLIRRockWriterOpInterfaceIncGen
  MLIRRockOpsIncGen
  )

# Add Triton AMD include paths for matrix acceleration detection APIs
# and for lib/TritonAMDGPUToLLVM/TargetInfo.h (AMD::TargetInfo)
target_include_directories(MLIRRockOps PRIVATE
  ${CMAKE_SOURCE_DIR}/external/triton/third_party/amd/include
  ${TRITON_PROJECT_DIR}/third_party/amd
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
)
target_include_directories(obj.MLIRRockOps PRIVATE
  ${CMAKE_SOURCE_DIR}/external/triton/third_party/amd/include
  ${TRITON_PROJECT_DIR}/third_party/amd
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
)

target_link_libraries(MLIRRockOps
  PUBLIC
  MLIRAffineDialect
  MLIRAMDGPUDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRDialect
  MLIRAnalysis
  MLIRParser
  MLIRROCDLDialect
  MLIRSupport
  MLIRLinalgTransformOps
  PRIVATE
  TritonAMDGPUTransforms
  )

# ROCm on Windows does not support HSA.
# Also HIP is not supported on our CI,
# so we cannot use either of them.
if (NOT WIN32)
  # From: https://github.com/ROCm/rocMLIR/blob/b6c726a324d2807494f770ed915af0e54cc6cb3f/external/llvm-project/mlir/lib/ExecutionEngine/CMakeLists.txt#L414
  find_package(hip REQUIRED PATHS ${ROCM_PATH})
  # We do need both of these. One is for linking, to get the HIP
  # functions.  The other is for compiling, to get -D__HIP_PLATFORM_AMD__
  target_link_libraries(MLIRRockOps PUBLIC
    hip::host hip::amdhip64
  )
  target_link_libraries(obj.MLIRRockOps PUBLIC
    hip::host hip::amdhip64
  )

  # From: https://github.com/ROCm/rocm-systems/blob/c53bdb9643afc2ef943fd25113a1c704d39aa6b4/projects/rocminfo/CMakeLists.txt
  find_package(hsa-runtime64 1.0 REQUIRED)
  target_link_libraries(MLIRRockOps PUBLIC hsa-runtime64::hsa-runtime64)
endif()
