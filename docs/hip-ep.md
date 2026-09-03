# HIP dialect integration

I found 2 different places to integrate HIP dialect:
- mlir/include/hip/Dialect/IR: No need to edit include paths. This is where HIP files expect to live, so integration is trivial
- mlir/include/mlir/Dialect/HIP/IR: This follows MIGraphX convention (which is at mlir/include/mlir/Dialect/MIGraphX/) however it requires re-wiring all the include paths. Also, when updating the HIP files, it would re-trigger the re-wiring each team.

I went for option 1.

## What was vendored

Just the dialect's IR files:

| Upstream (hip-ep)         | Here                           |
|---------------------------|--------------------------------|
| `include/hip/Dialect/IR/` | `mlir/include/hip/Dialect/IR/` |
| `lib/Dialect/IR/`         | `mlir/lib/Dialect/HIP/IR/`     |

These stay byte-identical to upstream so re-syncing is a plain `diff -r`.
`mlir/include/hip/README.md` records the imported hip-ep commit and the
deviations from it.

## Build wiring

- `add_subdirectory(include/hip)` in `mlir/CMakeLists.txt`, plus two one-line
  forwarding files (`mlir/include/hip/CMakeLists.txt` and
  `mlir/include/hip/Dialect/CMakeLists.txt`) to reach the vendored
  `Dialect/IR/CMakeLists.txt` that runs TableGen.
- `add_subdirectory(HIP)` in `mlir/lib/Dialect/CMakeLists.txt`.
- `mlir/lib/Dialect/HIP/CMakeLists.txt` replaces hip-ep's raw `add_library` with
  `add_rocmlir_dialect_library(MLIRHIPDialect ...)`. That helper registers the
  target in the `ROCMLIR_DIALECT_LIBS` global property, which `rocmlir-opt`
  already links, so no tool-side CMake change was needed.
- `hip::HipDialect` registered in `mlir/include/mlir/InitRocMLIRDialects.h`.

## LLVM version drift

- hip-ep uses LLVM 22.1.0
- We use LLVM 23

Two fixes:
- `mlir::OpaqueProperties` became `mlir::PropertyRef` in the `inferReturnTypes` signature
- The `static_cast<int64_t>` around `MemRefType::getNumDynamicDims` was dropped now that it returns an unsigned type.
