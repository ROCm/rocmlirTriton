# Building rocmlirTriton for MIGraphX

## Dependency chain

```
rocmlirTriton/requirements.txt -- top level (consumed by MIGraphX)
└──> Triton  (separate repo)
     └──> requirements.txt
          └──>Triton_LLVM
```

## Required `requirements.txt` files

### rocmlirTriton level: `rocmlirTriton/requirements.txt`

```text
Triton,/path/to/rocmlirTriton-triton/ -DTRITON_CACHE_PATH=/tmp/triton-cache -DTRITON_CODEGEN_BACKENDS=amd;nvidia -DTRITON_BUILD_PYTHON_MODULE=OFF -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_UT=OFF
```

Notes:

- Replace the local path with the upstream Git URL once the fork is published.
- The trailing slash on the local path is required by cget's local-path detection.
- The `TRITON_*` flags previously lived implicitly in `cmake/triton.cmake`. They do not propagate to a separately built cget package, so they live on this line instead.

### Triton level: `rocmlirTriton-triton/requirements.txt`

```text
TritonLLVM,/path/to/rocmlirTriton-llvm -DLLVM_ENABLE_PROJECTS=mlir;llvm;lld;clang -DLLVM_TARGETS_TO_BUILD=Native;NVPTX;AMDGPU -DLLVM_ENABLE_ASSERTIONS=ON -DMLIR_ENABLE_ROCM_RUNNER=ON -DLLVM_OPTIMIZED_TABLEGEN=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LLD=ON -DLLVM_INSTALL_UTILS=ON
```

Notes:

- The flag set matches what `external/triton/scripts/build-llvm-project.sh` uses, so the LLVM binaries are functionally identical to the legacy build.
- `LLVM_INSTALL_UTILS=ON` is essential: without it, downstream `find_package` consumers cannot import `FileCheck`, `count`, `not`, `llvm-lit`, which rocmlirTriton's `check-rocmlir` lit target depends on.

## Required environment hygiene

Before invoking the build, make sure the shell is not contaminated by
the legacy workflow:

```bash
unset LLVM_BUILD_DIR
unset MLIR_DIR
unset LLVM_SYSPATH
```
