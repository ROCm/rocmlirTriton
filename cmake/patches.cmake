# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Apply the downstream patches kept in llvm-patches/ and triton-patches/ to the
# corresponding git submodules (external/llvm-project, external/triton).

option(ROCMLIRTRITON_SKIP_PATCHES
       "Skip applying llvm-patches/ and triton-patches/ to the submodules" OFF)

# Apply every *.patch in <patch_dir> to the git work tree at <target_dir>.
function(_rocmlir_apply_patch_dir patch_dir target_dir label)
  if(NOT EXISTS "${patch_dir}")
    message(STATUS "  [${label}] no patch directory at ${patch_dir}; nothing to apply")
    return()
  endif()

  file(GLOB _patches LIST_DIRECTORIES false "${patch_dir}/*.patch")
  if(NOT _patches)
    message(STATUS "  [${label}] no *.patch files in ${patch_dir}")
    return()
  endif()
  list(SORT _patches)

  foreach(_patch IN LISTS _patches)
    get_filename_component(_name "${_patch}" NAME)

    # Already applied? `git apply --check --reverse` succeeds only when the
    # patch can be undone, i.e. it is currently present in the work tree.
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" apply -p1 --check --reverse "${_patch}"
      WORKING_DIRECTORY "${target_dir}"
      RESULT_VARIABLE _reverse_result
      OUTPUT_QUIET ERROR_QUIET)
    if(_reverse_result EQUAL 0)
      message(STATUS "  [${label}] already applied, skipping: ${_name}")
      continue()
    endif()

    # Not applied yet: verify it applies forward cleanly before mutating the tree.
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" apply -p1 --check "${_patch}"
      WORKING_DIRECTORY "${target_dir}"
      RESULT_VARIABLE _forward_result
      OUTPUT_VARIABLE _forward_out
      ERROR_VARIABLE _forward_err)
    if(NOT _forward_result EQUAL 0)
      message(FATAL_ERROR
        "[${label}] patch does not apply cleanly and is not already applied: "
        "${_patch}\n"
        "Apply it manually in ${target_dir} or refresh the patch.\n"
        "${_forward_out}\n${_forward_err}")
    endif()

    message(STATUS "  [${label}] applying: ${_name}")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" apply -p1 "${_patch}"
      WORKING_DIRECTORY "${target_dir}"
      RESULT_VARIABLE _apply_result
      OUTPUT_VARIABLE _apply_out
      ERROR_VARIABLE _apply_err)
    if(NOT _apply_result EQUAL 0)
      message(FATAL_ERROR
        "[${label}] failed to apply patch ${_patch} (exit ${_apply_result}):\n"
        "${_apply_out}\n${_apply_err}")
    endif()
  endforeach()
endfunction()

# Apply llvm-patches/ and triton-patches/ to their respective submodules.
# Must run after the submodules are checked out and before LLVM/Triton are
# pulled into the build graph via add_subdirectory().
function(rocmlir_apply_patches)
  if(ROCMLIRTRITON_SKIP_PATCHES)
    message(STATUS "Skipping downstream patches (ROCMLIRTRITON_SKIP_PATCHES=ON)")
    return()
  endif()

  # Mirrors submodules.cmake: a non-git checkout (e.g. a source tarball) is
  # assumed to already carry the patches; git apply would not work there.
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
    message(STATUS "Not a git checkout; assuming downstream patches are already applied")
    return()
  endif()

  find_program(GIT_EXECUTABLE NAMES git REQUIRED)

  message(STATUS "Applying downstream patches")
  _rocmlir_apply_patch_dir(
    "${CMAKE_SOURCE_DIR}/llvm-patches"
    "${CMAKE_SOURCE_DIR}/external/llvm-project"
    "llvm")
  _rocmlir_apply_patch_dir(
    "${CMAKE_SOURCE_DIR}/triton-patches"
    "${CMAKE_SOURCE_DIR}/external/triton"
    "triton")

  message(STATUS "Applying downstream patches - Success")
endfunction()
