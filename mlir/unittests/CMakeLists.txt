set_target_properties(RocMLIRUnitTests PROPERTIES FOLDER "MLIR Rock Tests")

# gtest's bundled custom headers in
# external/llvm-project/third-party/unittest/googletest/ trip the strict
# warnings we add for our own sources in mlir/CMakeLists.txt -- e.g. -Wundef
# (`#if !GTEST_NO_LLVM_SUPPORT` with the macro undefined) and -Wshadow
# (`RawStreamProxy<T>::V` shadowed by an operator<< parameter). The vendored
# gtest target declares its include dirs PUBLIC but not SYSTEM, so consumers
# compile them with -I and the warnings fire on third-party code. Re-marking
# the interface include dirs as SYSTEM makes them -isystem, which silences
# warnings originating in those headers while preserving the strict-warning
# posture on our own test sources. (LLVM's own unittests don't hit this because
# upstream doesn't enable -Wundef/-Wshadow.)
foreach(_gtest_target llvm_gtest llvm_gtest_main)
  if(TARGET ${_gtest_target})
    get_target_property(_gtest_includes ${_gtest_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_gtest_includes)
      set_target_properties(${_gtest_target} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_gtest_includes}")
    endif()
  endif()
endforeach()

function(add_rocmlir_unittest test_dirname)
  add_unittest(RocMLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
