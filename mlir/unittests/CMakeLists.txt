set_target_properties(RocMLIRUnitTests PROPERTIES FOLDER "MLIR Rock Tests")

# Drop the strict warnings we add for our own sources in mlir/CMakeLists.txt:
# gtest's bundled custom headers in
# external/llvm-project/third-party/unittest/googletest/ trip -Wundef
# (`#if !GTEST_NO_LLVM_SUPPORT` with the macro undefined) and -Wshadow
# (`RawStreamProxy<T>::V` shadowed by an operator<< parameter). LLVM's own
# unittests don't see these because upstream doesn't enable -Wundef/-Wshadow;
# they're a rocmlirTriton addition for production code. Turning them off for
# this subdirectory keeps the strict-warning posture for our libraries and
# tools while letting third-party test infrastructure compile cleanly.
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-undef>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-shadow>
)

function(add_rocmlir_unittest test_dirname)
  add_unittest(RocMLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
