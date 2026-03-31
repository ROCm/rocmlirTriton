# Scaled GEMMs

## 1. Handling the f8E8M0FNU Type

`f8E8M0FNU` is a special 8-bit floating-point format:

| Component | Meaning |
|-----------|---------|
| E8 | 8 exponent bits |
| M0 | **0 mantissa bits** |
| FN | Finite only (no inf/NaN) |
| U | Unsigned |

With zero mantissa bits, this format can **only represent powers of 2**. It's designed specifically for scale factors, not general computation.

## Why f8E8M0FNU is NOT in Triton's TT_Float

Triton's `TT_Float` types includes general-purpose floating-point types:
```
TT_Float = {F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64}
```

`f8E8M0FNU` is excluded because:
1. It cannot represent arbitrary floating-point values (only powers of 2)
2. It's not used for computation, only for scaling
3. Triton expects this kind of scale factors as raw `i8` bytes, not as a float type

## RockLegalizeFloatTypesPass

- We convert `f8E8M0FNU` to `i8`.
- Records original element types of A/B matrices on `BlockwiseGemmOp` for later use

## References

- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- Triton's `TT_Float` definition: `external/triton/include/triton/Dialect/Triton/IR/TritonTypes.td`
- `tt.dot_scaled` definition: `external/triton/include/triton/Dialect/Triton/IR/TritonOps.td`
