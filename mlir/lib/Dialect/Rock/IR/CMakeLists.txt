
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

# Triton AMD headers needed for matrix-acceleration detection APIs and
# for lib/TritonAMDGPUToLLVM/TargetInfo.h (AMD::TargetInfo).
rocmlir_add_triton_includes(MLIRRockOps)

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
  TritonIR
  TritonAMDGPUTransforms
  )

# ROCm on Windows does not support HSA.
if (NOT WIN32)
  # hip::host is an INTERFACE target: it only supplies the HIP headers and the
  # -D__HIP_PLATFORM_AMD__ compile definition needed to compile sources that
  # pull in HIP platform headers. The HIP/HSA *runtime* libraries are
  # deliberately not linked here: MLIRRockOps derives all architecture
  # information statically (via triton::AMD::TargetInfo) and references zero HIP
  # or HSA runtime symbols. Linking hip::amdhip64 / hsa-runtime64 would pull the
  # ROCm runtime -- and its bundled LLVM/comgr -- into every consumer of this
  # library (e.g. the test-only amd_arch_db pybind module), causing duplicate-
  # LLVM symbol clashes at load time.
  find_package(hip REQUIRED PATHS ${ROCM_PATH})
  target_link_libraries(MLIRRockOps PUBLIC hip::host)
  target_link_libraries(obj.MLIRRockOps PUBLIC hip::host)
endif()
