add_rocmlir_dialect_library(MLIRRockPipeline
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRAMDGPUTransforms
  MLIRCpuTransforms
  MLIRGPUToROCDLTransforms
  MLIRArithToAMDGPU
  MLIRComplexToLLVM
  MLIRIndexToLLVM
  MLIRMathToLibm
  MLIRMathToLLVM
  MLIRMathTransforms
  MLIRTensorToLinalg
  MLIRTosaToRock
  MLIRTosaToArith
  MLIRTosaToLinalg
  MLIRTosaToSCF
  MLIRTosaToTensor
  MLIRRockOps
  MLIRRockTransforms
  MLIRRockUtility
  MLIRUBToLLVM
  RocmlirCustomTosaDecompose
  RocmlirCustomTosaToLinalg
  RocmlirPromoteSoftmaxPrecision
  RocmlirEmulateFp8ExtTrunc

  # Upstream dialect libraries we need to load that we don't directly depend on
  MLIRBufferizationTransformOps
  MLIRControlFlowTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  
  TritonGPUTransforms
  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonAMDGPUToLLVM
  TritonAMDGPUTransforms
  TritonAMDAnalysis
  TritonAMDGPUDialectToLLVM
  TritonAMDUtils

  # Translation for HSACO generation
  TritonToHsacoTranslation
)

# Add Triton include directories for compilation
target_include_directories(MLIRRockPipeline PRIVATE
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
  ${TRITON_PROJECT_DIR}/third_party/amd
  ${TRITON_BINARY_DIR}/third_party/amd
  ${TRITON_PROJECT_DIR}/third_party/nvidia
  ${TRITON_BINARY_DIR}/third_party/nvidia
  ${TRITON_PROJECT_DIR}/third_party
  ${TRITON_PROJECT_DIR}
  ${TRITON_BINARY_DIR}/third_party
  ${TRITON_BINARY_DIR}/third_party/amd/include
)
