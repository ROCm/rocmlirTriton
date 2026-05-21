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

rocmlir_add_triton_includes(MLIRRockPipeline)
