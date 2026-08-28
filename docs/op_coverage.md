# Op Coverage

This documents describes the operations that rocmlirTriton can compile, and what restrictions apply.

[`mlir/include/mlir/Dialect/MIGraphX/IR/MIGraphX.td`](../mlir/include/mlir/Dialect/MIGraphX/IR/MIGraphX.td)
lists the ops that can be *represented*. This document lists what can
actually be *compiled*, which is a more restrictive set:

1. The graph must have a specific shape (Section 1).
2. Some specific ops have restrictions not specified in the `.td` (Section 2).

---

## 1. Supported graph shapes

**Kernels must contain exactly one anchor.** 

An anchor is a convolution, GEMM, or attention op. A kernel that only contains one of those ops (i.e., a kernel with just a GEMM) is supported too. Pointwise ops and reductions are supported only as fusions attached to an anchor, but never on their own (i.e., a pointwise add alone is not supported right now).

### 1.1 Anchors

| Anchor                    | MIGraphX ops                                   | Rock op                      | Fusion support                |
|---------------------------|------------------------------------------------|------------------------------|-------------------------------|
| GEMM                      | `dot`, `quant_dot`                             | `rock.gemm`                  | input and output              |
| Convolution               | `convolution`, `quant_convolution`             | `rock.conv`                  | input and output              |
| Backward-data convolution | `backwards_data_convolution`                   | `rock.conv_bwd_data`         | none                          |
| Attention                 | `dot` → `softmax` → `dot`                      | `rock.attention`             | input, pre-softmax and output |
| Fused gemm-gemm           | `dot` → pointwise → `dot`                      | `rock.gemm_elementwise_gemm` | restricted, see §1.3          |
| Fused conv-gemm           | `convolution` → pointwise → `dot`              | `rock.conv_elementwise_gemm` | restricted, see §1.3          |

The last three are single anchors, not two. The intermediate softmax or
pointwise stage is part of the anchor and is what makes the pattern match; it
cannot be omitted.

### 1.2 Fusions

Fusions may happen on either pointwise or reduction ops. Those ops must be attached (close) to an anchor, in any combination:

| Fusion    | Where                                           |
|-----------|-------------------------------------------------|
| Pointwise | on the anchor's operands (input fusion)         |
| Pointwise | on the anchor's result (output fusion)          |
| Reduction | on the anchor's result, after any output fusion |

Attention additionally accepts fusion between its first GEMM and the softmax,
which is where scale, bias and masking attach.

### 1.3 Fusion restrictions

Backward-data convolution accepts no fusion at all.

split-K has some output fusion restrictions:

- `add`/`sub` of the anchor result with another tensor, or with itself
- `mul`/`div` of the anchor result with another tensor
- `neg`
- type conversions

---

## 2. Per-op support

Restrictions below are in addition to the types declared in `MIGraphX.td`,
and in addition to the graph-shape rules of Section 1.

### 2.1 Anchor ops

| MIGraphX op                        | Rock op              | Restrictions |
|------------------------------------|----------------------|---|
| `dot`                              | `rock.gemm`          | Unsigned output unsupported. |
| `quant_dot`                        | `rock.gemm`          | Same as `dot`. Scales must be both present or both absent. The scaled version requires `f4E2M1FN` inputs, `f32` output, and K a multiple of 32. |
| `convolution`, `quant_convolution` | `rock.conv`          | 1D, 2D and 3D only. Unsigned output unsupported. 1D requires exactly 1 stride, 1 dilation and 2 pads. |
| `backwards_data_convolution`       | `rock.conv_bwd_data` | Same rank rules. No fusion. |
| -                                  | `rock.attention`     | Not lowered from a single MIGraphX op (see below) |

MIGraphX has no attention op. It emits the decomposed graph (two `dot`s with
a `softmax` between them) and rocmlirTriton recognises that shape and builds
`rock.attention` itself.

### 2.2 Pointwise ops

Supported as fusions: `add`, `sub`, `mul`, `div`, `pow`, `max`, `clip`,
`where`, `greater`, `equal`, `abs`, `ceil`, `erf`, `exp`, `floor`, `log`,
`neg`, `recip`, `relu`, `rsqrt`, `sigmoid`, `sqrt`, `tanh`, `convert`,
`quantizelinear`, `dequantizelinear`.

Restrictions:

| Op                 | Restriction                                                          |
|--------------------|----------------------------------------------------------------------|
| `neg`              | Unsigned integers unsupported.                                       |
| `div`              | On unsigned integers, both operands must have the same type.         |
| `max`              | 1-bit integers unsupported.                                          |
| `greater`, `equal` | Result element type is the input element type, not `i1`.             |
| `where`            | Condition must be `i8`/`si8`/`ui8`.                                  |
| `convert`          | float to float, float to int and unsigned conversions are supported. |

### 2.3 Reductions and softmax

| Op                                        | Restrictions                                        |
|-------------------------------------------|-----------------------------------------------------|
| `reduce_sum`, `reduce_max`, `reduce_mean` | Single axis only.                                   |
| `softmax`                                 | Supported only as part of the attention shape (§1). |

### 2.4 Shape and layout ops

`transpose`, `reshape`, `slice`, `broadcast` and `multibroadcast` are
supported and normally generate no code.

| Op               | Restrictions                                                                       |
|------------------|------------------------------------------------------------------------------------|
| `reshape`        | Static shapes only. At most one `-1` dimension, and `-1` cannot be mixed with `0`. |
| `multibroadcast` | Cannot reduce rank.                                                                |
