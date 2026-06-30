add_rocmlir_dialect_library(MLIRRockTransforms
  AddTritonMetadata.cpp
  AffixTuningParameters.cpp
  AnalyzeMemoryUse.cpp
  ConvToGemm.cpp
  SetMatmulOutputTranspose.cpp
  DetectFlashDecoding.cpp
  FoldBroadcast.cpp
  GemmToGridwise.cpp
  AttnToGridwise.cpp
  GridwiseGemmToBlockwise.cpp
  GridwiseAttnToBlockwise.cpp
  DecomposeNonPow2Tiles.cpp
  GridLayoutEmitter.cpp
  ViewToTransform.cpp
  FusionSplitkRegularizationPass.cpp
  LegalizeFloatTypes.cpp
  LowerBlockwiseToPtr.cpp
  ConvertNarrowTypeSignatures.cpp
  EmulateNarrowTypes.cpp
  PreserveMaskedLoadSemantics.cpp
  CollapseContiguousMerges.cpp
  TransformsInvariantCodeMotion.cpp
  TransformsToPointerArith.cpp
  RockToTTIR.cpp
  SerializeHostFuncs.cpp
  TensorToTritonPtr.cpp
  ResolveKernelLaunchParams.cpp
  RockEmitGpuBinary.cpp
  SortDimensionsMemoryLayout.cpp
  InsertOutputStores.cpp
  InsertOutputFusionLoads.cpp
  RegularizeInput.cpp
  LowerLoads.cpp
  LowerReduce.cpp
  LowerStores.cpp
  RegularizeInterGemmFusion.cpp
  RegularizeOutput.cpp
  RockFlattenTosaFuncArgs.cpp
  RockPrepareLLVM.cpp
  RockTosaToElementwise.cpp
  RockAllowFastMathFlags.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  DEPENDS
  MLIRRockOpsIncGen
  MLIRRockPassIncGen
  MLIRConversionPassIncGen
  RocMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRLinalgUtils
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMathToLLVM
  MLIRMemRefTransforms
  MLIRAffineToStandard
  MLIRReconcileUnrealizedCasts
  MLIRRockOps
  MLIRRockTuning
  MLIRRockUtility
  MLIRExecutionEngineUtils
  MLIRSCFTransforms
  MLIRAMDGPUUtils
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTosaDialect
  MLIRUBDialect
  MLIRVectorTransforms

  # Triton dependencies
  # TritonIR depends on TritonGPUIR because Traits.cpp uses MemDescType
  # TritonGPUIR depends on TritonNvidiaGPUIR because it references NVIDIA types/ops
  # TritonAnalysis provides supportMMA, getReshapeDecomposition
  # TritonGPUTransforms provides inferDstEncoding, verifyBarrierType
  TritonGPUTransforms
  TritonIR
  TritonGPUIR
  TritonAnalysis
)

rocmlir_add_triton_includes(MLIRRockTransforms)

# Fix build order: TritonIR and TritonGPUIR include headers that need generated files
# This ensures tablegen runs before these libraries build
if(TARGET TritonIR)
  foreach(target IN ITEMS
      TritonGPUTableGen
      TritonGPUAttrDefsIncGen
      TritonGPUCGAAttrIncGen
      TritonGPUTypeInterfacesIncGen
      TritonGPUOpInterfacesIncGen)
    if(TARGET ${target})
      add_dependencies(TritonIR ${target})
    endif()
  endforeach()
endif()

# TritonGPUIR unconditionally includes NVIDIA headers, so we need NVIDIA tablegen
if(TARGET TritonGPUIR)
  foreach(target IN ITEMS
      TritonNvidiaGPUTableGen
      TritonNvidiaGPUAttrDefsIncGen
      TritonNvidiaGPUOpInterfacesIncGen)
    if(TARGET ${target})
      add_dependencies(TritonGPUIR ${target})
    endif()
  endforeach()
endif()

# TritonGPUTransforms includes NVWS headers that need generated files
# This ensures NVWS tablegen runs before TritonGPUTransforms builds
if(TARGET TritonGPUTransforms)
  foreach(target IN ITEMS
      NVWSTableGen
      NVWSAttrDefsIncGen)
    if(TARGET ${target})
      add_dependencies(TritonGPUTransforms ${target})
    endif()
  endforeach()
endif()
