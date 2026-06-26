<!-- Badges -->
<!-- Uncomment when the project is made public -->
<!-- [![License](https://img.shields.io/github/license/ROCm/rocmlirTriton.svg?style=flat)](LICENSE) -->
<!-- [![Contributors](https://img.shields.io/github/contributors/ROCm/rocmlirTriton.svg?style=flat)](https://github.com/ROCm/rocmlirTriton/graphs/contributors) -->
<!-- Uncomment when CI is configured: -->
<!-- [![Build Status](https://github.com/ROCm/rocmlirTriton/actions/workflows/ci.yml/badge.svg)](https://github.com/ROCm/rocmlirTriton/actions) -->
<!-- [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/YOUR-ID/badge)](https://www.bestpractices.dev/projects/YOUR-ID) -->

# rocmlirTriton

> MLIR-based GEMM, convolution, attention, GEMM+GEMM, and CONV+GEMM kernel generator for AMD GPUs, built on a Triton compilation backend.

rocmlirTriton is a Triton-backed GPU kernel generator **derived from** [rocMLIR](https://github.com/ROCm/rocMLIR). Both share, for the most part, the same high-level lowering: `migraphx` -> `tosa` (or `linalg`) -> `rock`, and diverge only at the codegen step: rocMLIR lowers `rock` to MLIR's AMDGPU and ROCDL dialects to HSACO via the LLVM AMDGPU backend, while rocmlirTriton hands off to Triton's TTIR -> TTGIR -> LLIR pipeline (see `Pipelines.cpp` / `TritonToHsaco.cpp`), with vendored Triton (under `external/triton`) and Triton-pinned LLVM/MLIR (under `external/llvm-project`) producing the final HSACO.

It targets AMD CDNA and RDNA GPUs (gfx9xx / gfx10xx / gfx11xx / gfx12xx), and is primarily consumed as the static `librockCompiler` library by [MIGraphX](https://github.com/ROCm/AMDMIGraphX), though it can also be driven standalone for kernel generation, validation, and performance tuning.

## Prerequisites

- An AMD GPU and a working [ROCm](https://rocm.docs.amd.com/) installation (with `rocminfo` on `PATH`).
- `clang` / `clang++` 20 (defaults to `clang-20` / `clang++-20`; override via the `C_COMPILER` / `CXX_COMPILER` environment variables).
- `lld`, `ninja`, and CMake >= 3.20.
- Python 3 (only needed for in-tree development scripts and the LIT test runner; not required for production builds or MIGraphX integration).
- Git

## Installation

Triton and LLVM/MLIR are vendored in the repo (under `external/triton` and `external/llvm-project`), imported via `git subtree`. The build is driven by `cmake.sh`, which configures and builds LLVM/MLIR, Triton, and rocmlirTriton from those vendored trees. Patch files under `llvm-patches/` and `triton-patches/` are kept for provenance and to simplify the next upstream bump; they are not applied during CMake configure.

```sh
git clone https://github.com/ROCm/rocmlirTriton.git
cd rocmlirTriton
bash cmake.sh
```

To install `librockCompiler` so MIGraphX can find it:

```sh
cmake --install build --prefix /path/to/MIGraphX/deps
```

Additional developer documentation lives under [`docs/`](docs/).

## Usage

A typical standalone pipeline generates a kernel with `rocmlir-gen`, lowers it with `rocmlir-driver -c`, and runs it via `rocm-run` -- a wrapper around `mlir-runner` that auto-locates the rocMLIR build and the Triton-pinned LLVM build directory, and links the right runtime libraries (`libmlir_rocm_runtime`, `libconv-validation-wrappers`, runner utils, etc.):

```sh
ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)

build/bin/rocmlir-gen -pv -operation gemm -t f16 -out_datatype f32 \
    --arch "$ARCH" -g 1 -m 64 -k 256 -n 128 \
  | build/bin/rocmlir-driver -c \
  | build/bin/rocm-run
```

Useful `rocmlir-gen` flags:

- `--arch` -- target AMDGPU architecture (e.g. `gfx942`, `gfx950`, `gfx1100`); MFMA/WMMA support is inferred from the chosen architecture.
- `-t` / `--dtype` -- data type selector (e.g. `f16`, `f32`, `bf16`, `i8`, `fp8_fp8`).
- `-out_datatype` / `--out_dtype` / `-tc` -- override the output data type independently of `-t` (e.g. f16 input with f32 output).
- `--perf_config` -- supply a serialized tuning configuration.
- `-ph` -- emit host code alongside the kernel.
- `-pv` -- validate kernel results against a CPU reference.
- `-pr` -- print kernel results.

Run `build/bin/rocmlir-gen --help` for the full, current option list.

More examples live under `mlir/test/rocmlir-driver/` (notably `sanity.mlir`), with end-to-end PR tests under `mlir/test/fusion/pr-e2e/` (including the MIGraphX-dialect `mixr-*` tests) and `mlir/test/e2e/`. To build and run the full in-tree test suite (from the build directory):

```sh
cd build && ninja check-rocmlir
```

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the issue-reporting and pull-request workflow.

For bugs and feature requests, open a [GitHub Issue](../../issues).

---

## Security

To report a security vulnerability, **do not open a public GitHub issue**.
See [SECURITY.md](SECURITY.md) for our responsible disclosure policy.

---

## Contact

For questions, issues, or contributions, please reach out to the maintainers:

- Chris Austen — [@causten](https://github.com/causten) · chausten@amd.com

See [CODEOWNERS](.github/CODEOWNERS) for the full ownership list.

---

## License

This project is licensed under the [Apache License 2.0 with LLVM Exceptions](LICENSE).
