# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Ensure git submodules declared in .gitmodules are checked out.
function(rocmlir_ensure_submodules)
  if(ROCMLIRTRITON_SKIP_SUBMODULE_UPDATE)
    message(STATUS "Skipping git submodule update (ROCMLIRTRITON_SKIP_SUBMODULE_UPDATE=ON)")
    return()
  endif()

  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
    message(STATUS "Not a git checkout; assuming submodules are already present")
    return()
  endif()

  find_program(GIT_EXECUTABLE NAMES git REQUIRED)
  message(STATUS "Initializing git submodules (git submodule update --init --recursive)")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _rocmlir_submodule_result
    OUTPUT_VARIABLE _rocmlir_submodule_out
    ERROR_VARIABLE _rocmlir_submodule_err
  )
  if(NOT _rocmlir_submodule_result EQUAL 0)
    message(FATAL_ERROR
      "git submodule update --init --recursive failed (exit ${_rocmlir_submodule_result}):\n"
      "${_rocmlir_submodule_out}\n${_rocmlir_submodule_err}")
  endif()
endfunction()

