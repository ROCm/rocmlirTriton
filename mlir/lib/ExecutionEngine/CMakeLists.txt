set(disable_install "")
if (BUILD_FAT_LIBROCKCOMPILER)
  set(disable_install "DISABLE_INSTALL")
endif()
add_mlir_library(conv-validation-wrappers SHARED
  EXCLUDE_FROM_LIBMLIR
  ${disable_install}
  conv-validation-wrappers.cpp
)
set_target_properties(conv-validation-wrappers PROPERTIES EXCLUDE_FROM_ALL ${BUILD_FAT_LIBROCKCOMPILER})

target_link_libraries(conv-validation-wrappers
  PUBLIC
  LLVMSupport
)

# Hide the statically-linked LLVM symbols so they cannot clash with a second
# LLVM present in the same process. This SHARED library is dlopen-ed by
# mlir-runner via --shared-libs, frequently alongside libmlir_rocm_runtime
# (whose DT_NEEDED chain libamdhip64 -> libamd_comgr -> libLLVM.so drags in
# ROCm's own LLVM). With default visibility the statically-linked LLVMSupport
# data symbols (cl::opt / pass-registry / ManagedStatic singletons) would be
# interposed against the second LLVM and abort during dynamic init with
#   SmallPtrSet.h: Assertion `Bucket < End' failed.
# -Bsymbolic-functions alone does NOT cover this -- the clashing symbols are
# *data*. --exclude-libs,ALL localizes the symbols pulled from the static LLVM
# archives while leaving this library's own extern "C" wrappers exported.
# Mirrors the mlir_apfloat_wrappers / mlir_async_runtime treatment in the
# upstream MLIR ExecutionEngine CMakeLists.
target_link_options(conv-validation-wrappers PRIVATE
  $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>)

if (NOT MSVC)
  # Suppress some compiler warnings when building with clang.  The first two
  # suppressions regard a GNU extension, the third an incomplete-type warning
  # that apparently works out okay.
  set_source_files_properties(conv-validation-wrappers.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-gnu-anonymous-struct;-Wno-nested-anon-types;-Wno-return-type-c-linkage;-Wno-c++98-compat-extra-semi")
endif()

