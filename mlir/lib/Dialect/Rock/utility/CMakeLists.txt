add_rocmlir_dialect_library(MLIRRockUtility
  builderUtils.cpp
  loweringUtils.cpp
  transformMapUtils.cpp
  fusionUtils.cpp
  tosaUtils.cpp
  RocmDeviceName.cpp
  compileUtils.cpp
  tritonUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRArithDialect
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRRockOps
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIRROCDLDialect

  PRIVATE
  TritonIR
  TritonAMDGPUTransforms
)

# Triton include directories needed by tritonUtils.cpp.
# Marked SYSTEM so our strict warnings (-Wshadow, etc.) don't fire on them.
target_include_directories(MLIRRockUtility SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/external/triton/third_party/amd/include
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
)
target_include_directories(obj.MLIRRockUtility SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/external/triton/third_party/amd/include
  ${TRITON_PROJECT_DIR}/include
  ${TRITON_BINARY_DIR}/include
)

# HIP needed by tritonUtils.cpp (launchKernel)
if (NOT WIN32)
  find_package(hip REQUIRED PATHS ${ROCM_PATH})
  target_link_libraries(MLIRRockUtility PUBLIC hip::host hip::amdhip64)
  target_link_libraries(obj.MLIRRockUtility PUBLIC hip::host hip::amdhip64)
endif()
