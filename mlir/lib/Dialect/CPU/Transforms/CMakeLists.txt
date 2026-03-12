add_rocmlir_dialect_library(MLIRCpuTransforms
  LowerCpuVerifier.cpp
  Schedules.cpp
  Schedules/LowerToLLVMSchedule.cpp
  Schedules/UnrollSchedule.cpp
  Schedules/PrePostSchedules.cpp
  Schedules/ScheduleUtils.cpp
  Schedules/TilingSchedule.cpp
  Schedules/VectorizationSchedule.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/CPU

  DEPENDS
  MLIRCpuPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBufferizationTransformOps
  MLIRDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgTransformOps
  MLIRLinalgUtils
  MLIRLinalgTransforms
  MLIRLLVMDialect
  MLIRMathToLLVM
  MLIRMemRefTransforms
  MLIRAffineToStandard
  MLIRParser
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFTransforms
  MLIRSCFTransformOps
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTensorToLinalg
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRVectorTransforms
  MLIRVectorTransformOps
  MLIRFuncTransformOps
  MLIRMemRefTransformOps
)
