# Building rocmlirTriton on Windows

Two host toolchains are supported:

| Toolchain | Status |
|---|---|
| **clang-cl** (from the ROCm / HIP SDK) | Default |
| **MSVC `cl.exe`** | Opt-in, via `-DROCMLIR_ALLOW_MSVC=ON` |

Both target the MSVC ABI, so either build links against MSVC-built consumers such
as MIGraphX, and both produce identical GPU code — kernels are emitted at runtime
by the in-process LLVM AMDGPU backend, so the host compiler is not involved.

## Prerequisites

- An AMD GPU and a working [ROCm / HIP SDK for Windows](https://rocm.docs.amd.com/),
  with `hipInfo` or `rocminfo` on `PATH`. The build reads `ROCM_PATH`
  (default `C:/opt/rocm`).
- **Visual Studio 2022** (or the standalone Build Tools) with the
  *Desktop development with C++* workload. This provides the Windows SDK, the
  MSVC libraries, and `ninja.exe`. All three are required for both toolchains,
  because `lld-link` also resolves the Windows system libraries through the MSVC
  environment.
- CMake >= 3.20, and Python 3 for the lit test runner and dev scripts.
- Roughly 90 GB free disk and 32 GB RAM.

All commands must run inside the MSVC environment. Use a
**Developer PowerShell for VS 2022**, or import the environment into an
existing PowerShell session:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```

Calling `vcvars64.bat` from PowerShell does not work: it is a batch file, so it
runs in a child `cmd.exe` and its variables are gone when that process exits.
From `cmd.exe` the usual `vcvars64.bat` call is still correct.

## Quick start

`scripts/build-windows.ps1` is the Windows build entry point:

```powershell
git clone https://github.com/ROCm/rocmlirTriton.git
cd rocmlirTriton

# clang-cl (default)
pwsh scripts/build-windows.ps1 -GpuTargets gfx1151

# MSVC cl.exe
pwsh scripts/build-windows.ps1 -Msvc -BuildDir build-msvc -GpuTargets gfx1151
```

The flags relevant here:

| Flag | Meaning |
|---|---|
| `-Msvc` | Build with `cl.exe` instead of clang-cl. Implies `-DROCMLIR_ALLOW_MSVC=ON` |
| `-RocmPath <dir>` | ROCm / HIP SDK root. Defaults to `ROCM_PATH`, then `HIP_PATH`, then `C:/opt/rocm` |
| `-BuildDir <dir>` | Build directory, default `build` |
| `-GpuTargets <list>` | Semicolon-separated, e.g. `gfx1151` or `gfx1100;gfx1201` |
| `-BuildType <cfg>` | `Release`, `RelWithDebInfo` (default), `Debug`, `MinSizeRel` |
| `-Jobs <n>` | Parallel jobs; 0 (default) uses every logical processor |
| `-ConfigureOnly` | Stop after the configure step |
| `-Targets <list>` | Ninja targets, default `check-rocmlir-build-only` |

`-CMakeArgs` is forwarded verbatim to the configure step, and is applied last so
it overrides anything the script chooses:

```powershell
pwsh scripts/build-windows.ps1 -Msvc -CMakeArgs "-DLLVM_ENABLE_ASSERTIONS=OFF"
```

Use a separate `-BuildDir` per toolchain: the CMake cache pins the compiler, so
reusing one directory across clang-cl and `cl.exe` will not reconfigure cleanly.

## Configuring CMake directly

With clang-cl:

```powershell
cmake -G Ninja -S . -B build `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_FAT_LIBROCKCOMPILER=ON `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
  -DCMAKE_PREFIX_PATH=C:/opt/rocm `
  -DROCM_PATH=C:/opt/rocm `
  -DGPU_TARGETS=gfx1151 `
  -DROCM_TEST_CHIPSET=gfx1151 `
  -DCMAKE_RC_COMPILER=C:/opt/rocm/bin/llvm-rc.exe `
  -DLLVM_ENABLE_DIA_SDK=OFF

cmake --build build -- -j 16
```

With MSVC, drop `CMAKE_RC_COMPILER` and add:

```powershell
  -DROCMLIR_ALLOW_MSVC=ON `
  -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe `
  -DCMAKE_EXE_LINKER_FLAGS=/INCREMENTAL:NO `
  -DCMAKE_SHARED_LINKER_FLAGS=/INCREMENTAL:NO
```

Notes on the Windows-specific flags:

- `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` — selects the static CRT (`/MT`).
  Not a default: set it only to match the consumer, since a library and the
  application linking it must agree, or they end up with separate heaps.
- `LLVM_ENABLE_DIA_SDK=OFF` — avoids requiring the optional ATL component.
- `CMAKE_RC_COMPILER=llvm-rc.exe` — needed for clang-cl only. `llvm-rc.exe`
  sits beside `clang-cl`: under `lib/llvm/bin` on TheRock, under `bin` on the
  HIP SDK.
- `/INCREMENTAL:NO` with MSVC — its incremental linker writes multi-GB `.ilk`
  files alongside these targets. `build-windows.ps1 -Msvc` passes it for you.
- Prefer `Release` over `RelWithDebInfo` unless you need debug info; the latter
  adds tens of GB of `.pdb`.
- Do not set `LLVM_PARALLEL_COMPILE_JOBS` or `LLVM_PARALLEL_LINK_JOBS`; use
  `ninja -j` to bound parallelism.

## Build products

On Windows the individual static libraries are the deliverable rather than one
monolithic archive, since static libraries are capped at 4 GB. `rockCompiler` is
an `INTERFACE` target that propagates them, and `librockCompiler` is a
convenience alias. Consumers are unaffected and still use
`find_package(rocmlir)`.

Install for MIGraphX:

```powershell
cmake --install build --prefix C:/path/to/MIGraphX/deps
```

This produces `lib/*.lib`, `lib/llvm/*.lib`, `lib/cmake/rocmlir/*.cmake`, and the
public headers under `include/rocmlir/`.

## Verifying a build

```powershell
build\bin\rocmlir-opt.exe --version
build\bin\rocmlir-driver.exe --version

# unit tests
build\mlir\unittests\Dialect\Rock\MLIRRockUnitTests.exe

# a pipeline end to end
build\bin\rocmlir-driver.exe -kernel-pipeline=migraphx,highlevel `
  mlir\test\Dialect\Rock\rock-allow-fast-math-flags.mlir
```

`ninja check-rocmlir` additionally requires ROCm's `hip` Python module, which
`mlir/test/lit.cfg.py` imports for device detection. Install the ROCm Python
bindings to run the full lit suite, or invoke the tools directly as above.

## Troubleshooting

| Symptom | Action |
|---|---|
| `clang-cl not found (ROCM_PATH=...)` | Set `ROCM_PATH` to the HIP SDK, or pass `-DCMAKE_C_COMPILER` / `-DCMAKE_CXX_COMPILER`. |
| `Windows builds require clang-cl; configured compiler is MSVC` | Add `-DROCMLIR_ALLOW_MSVC=ON`. |
| `lld-link: error: could not open 'kernel32.lib'` | Run from a Developer PowerShell, or call `vcvars64.bat` first. |
| `fatal error: 'atlbase.h' file not found` | Add `-DLLVM_ENABLE_DIA_SDK=OFF`, or install the ATL component. |
| `ninja: error: duplicate pool 'compile_job_pool'` | Remove `LLVM_PARALLEL_COMPILE_JOBS` / `LLVM_PARALLEL_LINK_JOBS`. |
| `rc.exe` rejects an option (clang-cl) | Set `-DCMAKE_RC_COMPILER` to `llvm-rc.exe` beside `clang-cl` (`<rocm>/lib/llvm/bin` or `<rocm>/bin`). |
| `C1060: compiler is out of heap space` | Lower `-Jobs`; link steps are the memory peak. |
| Disk fills while linking | Use `Release`, and keep `/INCREMENTAL:NO` with MSVC. |
| `ModuleNotFoundError: No module named 'hip'` | Install ROCm's Python bindings, or run the tools directly. |
