add_rocmlir_dialect_library(MLIRRockUtility
  PARTIAL_SOURCES_INTENDED
  builderUtils.cpp
  loweringUtils.cpp
  transformMapUtils.cpp
  fusionUtils.cpp
  tosaUtils.cpp
  RocmDeviceName.cpp
  compileUtils.cpp
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

# Triton include directories needed by tritonUtils.cpp.
rocmlir_add_triton_includes(MLIRRockUtility)

# Native device queries need HIP at compile and link time, so they live in
# their own library instead of MLIRRockUtility. MLIRRockUtility is part of
# librockCompiler, and MIGraphX runs its own CU query and passes the result in
# as `rock.num_cu`, so the compiler library must never oblige its consumers to
# link libamdhip64. Only the tools and the amd_arch_db binding link this.
rocmlir_find_hip()

add_mlir_library(MLIRRockDeviceInfo
  PARTIAL_SOURCES_INTENDED
  EXCLUDE_FROM_LIBMLIR
  DeviceInfo.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRRockOps
  MLIRSupport
)

# `hip::host` carries the `__HIP_PLATFORM_AMD__` define and the HIP include
# path, `hip::amdhip64` carries the runtime itself. Both have to reach the
# `obj.` target as well, because that is where the sources are compiled; the
# include path only arrives as a system include through the imported target,
# which is what keeps the pedantic warnings in HIP's own headers quiet.
target_link_libraries(MLIRRockDeviceInfo PRIVATE hip::host hip::amdhip64)
if(TARGET obj.MLIRRockDeviceInfo)
  target_link_libraries(obj.MLIRRockDeviceInfo PRIVATE hip::host hip::amdhip64)
endif()
