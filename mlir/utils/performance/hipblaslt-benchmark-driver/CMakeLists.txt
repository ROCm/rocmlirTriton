# Find hipBLAS and hipBLASLt libraries directly
find_library(HIPBLAS_LIBRARY
  NAMES hipblas
  PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
  PATH_SUFFIXES lib lib64
)

find_library(HIPBLASLT_LIBRARY
  NAMES hipblaslt
  PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64
  PATH_SUFFIXES lib lib64
)

# Check if libraries and include directories exist
if (HIPBLAS_LIBRARY AND HIPBLASLT_LIBRARY AND
    EXISTS "${ROCM_PATH}/include/hipblas" AND EXISTS "${ROCM_PATH}/include/hipblaslt")
  set(LIBS ${HIPBLAS_LIBRARY} ${HIPBLASLT_LIBRARY} benchmark-driver-utils)

  add_executable(hipblaslt-benchmark-driver
    hipblaslt-benchmark-driver.cpp
  )

  target_link_libraries(hipblaslt-benchmark-driver PRIVATE ${LIBS})
  target_include_directories(hipblaslt-benchmark-driver PRIVATE
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hipblas
    ${ROCM_PATH}/include/hipblaslt
  )
  set_target_properties(hipblaslt-benchmark-driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  
  message(STATUS "Building hipBLASLt benchmark driver with libraries: ${LIBS}")
else()
  message(STATUS "Could not find hipBLAS/hipBLASLt libraries or include directories, not building hipBLASLt benchmark driver")
endif()
