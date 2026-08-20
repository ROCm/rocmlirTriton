rocmlir_find_hip()

add_rocmlir_dialect_library(MLIRRockUtility
  builderUtils.cpp
  loweringUtils.cpp
  transformMapUtils.cpp
  fusionUtils.cpp
  tosaUtils.cpp
  RocmDeviceName.cpp
  compileUtils.cpp
  DeviceInfo.cpp
  tritonUtils.cpp
  KnobUtils.cpp

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

target_link_libraries(MLIRRockUtility PRIVATE hip::host hip::amdhip64)
set_source_files_properties(DeviceInfo.cpp PROPERTIES COMPILE_OPTIONS
  "-Wno-c++98-compat-extra-semi;-Wno-shadow;-Wno-undef")

# Triton include directories needed by tritonUtils.cpp.
rocmlir_add_triton_includes(MLIRRockUtility)
