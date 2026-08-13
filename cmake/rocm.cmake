# Find HIP, and optionally hiprtc, using the ROCm installation prefix. Several
# ROCm package files only find their own dependencies through CMAKE_PREFIX_PATH,
# so extend it for the duration of this function.
function(rocmlir_find_hip)
  cmake_parse_arguments(PARSE_ARGV 0 ROCMLIR_FIND_HIP "HIPRTC" "" "")

  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(_rocmlir_default_rocm_path "$ENV{ROCM_PATH}")
    else()
      set(_rocmlir_default_rocm_path "/opt/rocm")
    endif()
    set(ROCM_PATH "${_rocmlir_default_rocm_path}" CACHE PATH
      "Path to which ROCm has been installed")
  endif()

  list(APPEND CMAKE_PREFIX_PATH
    "${ROCM_PATH}"
    "${ROCM_PATH}/hip"
  )
  if(ROCMLIR_FIND_HIP_HIPRTC)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/hiprtc")
  endif()

  find_package(hip REQUIRED)
  if(ROCMLIR_FIND_HIP_HIPRTC)
    find_package(hiprtc REQUIRED)
  endif()
endfunction(rocmlir_find_hip)
