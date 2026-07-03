# F32 Dot BF16x3 on WMMA Targets

## Summary

This branch moves f32 `tt.dot` lowering to `inputPrecision = bf16x3`
only for WMMA architectures that do not have native f32 matrix
acceleration. This lets gfx12-class targets such as `gfx1201` use BF16
WMMA for f32 dots instead of falling back to scalar FMA, while preserving
native f32 MFMA/WMMA paths on architectures such as `gfx942` and `gfx1250`.

The architecture decision is made through the Rock AMD architecture
database. The bridge checks whether f32 inputs already have matrix
acceleration and only requests bf16x3 when f32 acceleration is absent but
BF16 WMMA is available.

## Tests

Coverage added in this branch:

- `mlir/test/Dialect/Rock/rock-to-ttir-f32-dot-input-precision.mlir`
  checks the Rock-to-TTIR decision for `gfx1201`, `gfx1250`, and `gfx942`.
- `mlir/test/rocmlir-driver/conv-f32-im2col-bf16x3-dot.mlir` checks the
  driver GPU pipeline for the same arch split on an im2col f32 convolution.
- `mlir/test/e2e/PrConvF32Bf16x3Dot.toml` adds a PR E2E f32 convolution
  validation case that runs through full codegen on the local test arch.
