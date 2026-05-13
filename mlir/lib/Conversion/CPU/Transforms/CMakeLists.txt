add_rocmlir_dialect_library(MLIRCpuTransforms
  ConvToGemm.cpp
  LowerCpuVerifier.cpp
  Schedules.cpp
  Schedules/FusedConvToMatmulSchedule.cpp
  Schedules/LowerToLLVMSchedule.cpp
  Schedules/UnrollSchedule.cpp
  Schedules/PrePostSchedules.cpp
  Schedules/ScheduleUtils.cpp
  Schedules/TilingSchedule.cpp
  Schedules/VectorizationSchedule.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/CPU

  DEPENDS
  MLIRCpuPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRRocMLIRUtils
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
  MLIRTensorTransformOps
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRVectorTransforms
  MLIRVectorTransformOps
  MLIRFuncTransformOps
  MLIRMemRefTransformOps
)

# HIP needed transitively via tritonUtils.h (included by LowerCpuVerifier.cpp)
if (NOT WIN32)
  find_package(hip REQUIRED PATHS ${ROCM_PATH})
  target_link_libraries(MLIRCpuTransforms PUBLIC hip::host)
  target_link_libraries(obj.MLIRCpuTransforms PUBLIC hip::host)
endif()
