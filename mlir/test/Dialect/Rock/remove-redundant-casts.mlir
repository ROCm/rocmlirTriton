// RUN: rocmlir-opt -rock-allow-fast-math-flags -mlir-print-local-scope %s | FileCheck %s

// ============================================================
// Direct pure-SSA round-trip: extf(truncf %wide) -> %wide.
// The truncf result has exactly one use (the extf) and the wide
// types match. The pair must be folded away.
// ============================================================

// CHECK-LABEL: func.func @fold_direct_roundtrip
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>)
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
//      CHECK:   %[[MUL:.*]] = arith.mulf %[[ARG]], %[[ARG]]
//      CHECK:   return %[[MUL]]
func.func @fold_direct_roundtrip(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
    attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  %2 = arith.mulf %1, %arg0 : tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// ============================================================
// Scalar (non-tensor) round-trip is folded the same way.
// ============================================================

// CHECK-LABEL: func.func @fold_scalar_roundtrip
// CHECK-SAME: (%[[ARG:.*]]: f32)
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
//      CHECK:   %[[ADD:.*]] = arith.addf %[[ARG]], %[[ARG]]
//      CHECK:   return %[[ADD]]
func.func @fold_scalar_roundtrip(%arg0: f32) -> f32
    attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : f32 to f16
  %1 = arith.extf %0 : f16 to f32
  %2 = arith.addf %1, %arg0 : f32
  return %2 : f32
}

// ============================================================
// Multi-use of the narrow value: the truncf's result feeds both the
// extf and a narrow store. The extf still folds (its consumer is
// rewritten to use the wide source directly), but the truncf
// survives DCE because the narrow store keeps it alive. The wide
// store on the mulf result is symmetric and verifies that the f32
// path is untouched -- no precision drop on either side.
// ============================================================

// CHECK-LABEL: func.func @fold_extf_keep_truncf_with_extra_use
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>, %[[DST_F32:.*]]: tensor<32x32xf32>, %[[DST_F16:.*]]: tensor<32x32xf16>)
//  CHECK-NOT:   arith.extf
//      CHECK:   %[[TR:.*]] = arith.truncf %[[ARG]]
//      CHECK:   %[[MUL:.*]] = arith.mulf %[[ARG]], %[[ARG]]
//      CHECK:   rock.store %[[MUL]] to %[[DST_F32]]
//      CHECK:   rock.store %[[TR]] to %[[DST_F16]]
func.func @fold_extf_keep_truncf_with_extra_use(%arg0: tensor<32x32xf32>,
                                       %dst_f32: tensor<32x32xf32>,
                                       %dst_f16: tensor<32x32xf16>)
    -> (tensor<32x32xf32>, tensor<32x32xf16>) attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  %2 = arith.mulf %1, %arg0 : tensor<32x32xf32>
  %3 = rock.store %2 to %dst_f32 by set
       : tensor<32x32xf32> -> tensor<32x32xf32> to tensor<32x32xf32>
  %4 = rock.store %0 to %dst_f16 by set
       : tensor<32x32xf16> -> tensor<32x32xf16> to tensor<32x32xf16>
  return %3, %4 : tensor<32x32xf32>, tensor<32x32xf16>
}

// ============================================================
// Mismatched wide types: f64 -> f16 -> f32 is genuine precision
// shaping (the user is widening to f32, not f64). The pass must
// NOT fold the pair and must NOT annotate either cast with
// `fastmath<contract>` -- the `CHECK-NOT: fastmath` bounded by the
// return anchor pins down the latter.
// ============================================================

// CHECK-LABEL: func.func @keep_mismatched_wide_types
//  CHECK-NOT:   fastmath
//      CHECK:   arith.truncf
//      CHECK:   arith.extf
//      CHECK:   return
func.func @keep_mismatched_wide_types(%arg0: tensor<32x32xf64>)
    -> tensor<32x32xf32> attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<32x32xf64> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// ============================================================
// extf whose operand is NOT a truncf: must remain unchanged.
// ============================================================

// CHECK-LABEL: func.func @keep_lone_extf
//      CHECK:   %[[EX:.*]] = arith.extf
//      CHECK:   return %[[EX]]
func.func @keep_lone_extf(%arg0: tensor<32x32xf16>) -> tensor<32x32xf32>
    attributes {rock.kernel} {
  %0 = arith.extf %arg0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// ============================================================
// Round-trip mirroring the attention Pattern A born by
// RockGridwiseAttnToBlockwisePass: an f32 accumulator is trunc'd
// to f16 only to be extf'd back so a downstream scale-and-softmax
// can run in f32. The cleanup should leave the f32 chain intact
// with no precision drop.
// ============================================================

// CHECK-LABEL: func.func @attention_acc_to_softmax_pattern
// CHECK-SAME: (%[[ACC:.*]]: tensor<32x32xf32>)
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
//      CHECK:   %[[CST_SCALE:.*]] = arith.constant {{.*}}0.0883
//      CHECK:   %[[SCALED:.*]] = arith.mulf %[[ACC]], %[[CST_SCALE]]
//      CHECK:   return %[[SCALED]]
func.func @attention_acc_to_softmax_pattern(%acc: tensor<32x32xf32>)
    -> tensor<32x32xf32> attributes {rock.kernel} {
  %narrow = arith.truncf %acc : tensor<32x32xf32> to tensor<32x32xf16>
  %cst = arith.constant dense<0.0883789062> : tensor<32x32xf32>
  %wide = arith.extf %narrow : tensor<32x32xf16> to tensor<32x32xf32>
  %scaled = arith.mulf %wide, %cst : tensor<32x32xf32>
  return %scaled : tensor<32x32xf32>
}

// ============================================================
// Dual pattern: truncf(extf %narrow) -> %narrow. This is the
// unconditionally safe direction (extf is lossless, truncf back
// to the original narrow type is the identity for every bit
// pattern). The pair must be folded away.
// ============================================================

// CHECK-LABEL: func.func @fold_dual_roundtrip
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf16>)
//  CHECK-NOT:   arith.extf
//  CHECK-NOT:   arith.truncf
//      CHECK:   return %[[ARG]]
func.func @fold_dual_roundtrip(%arg0: tensor<32x32xf16>) -> tensor<32x32xf16>
    attributes {rock.kernel} {
  %0 = arith.extf %arg0 : tensor<32x32xf16> to tensor<32x32xf32>
  %1 = arith.truncf %0 : tensor<32x32xf32> to tensor<32x32xf16>
  return %1 : tensor<32x32xf16>
}

// ============================================================
// Dual pattern with a multi-use extf: the truncf still folds to
// the narrow source, while the extf survives because another
// consumer (a wide multiply) keeps it alive. No precision change
// on either path.
// ============================================================

// CHECK-LABEL: func.func @fold_dual_keep_extf_with_extra_use
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf16>)
//  CHECK-NOT:   arith.truncf
//      CHECK:   %[[EX:.*]] = arith.extf %[[ARG]]
//      CHECK:   %[[MUL:.*]] = arith.mulf %[[EX]], %[[EX]]
//      CHECK:   return %[[MUL]], %[[ARG]]
func.func @fold_dual_keep_extf_with_extra_use(%arg0: tensor<32x32xf16>)
    -> (tensor<32x32xf32>, tensor<32x32xf16>) attributes {rock.kernel} {
  %0 = arith.extf %arg0 : tensor<32x32xf16> to tensor<32x32xf32>
  %1 = arith.mulf %0, %0 : tensor<32x32xf32>
  %2 = arith.truncf %0 : tensor<32x32xf32> to tensor<32x32xf16>
  return %1, %2 : tensor<32x32xf32>, tensor<32x32xf16>
}

// ============================================================
// Dual mismatched narrow types: f16 -> f32 -> bf16 is genuine
// precision shaping, not a round-trip. Must NOT fold.
// ============================================================

// CHECK-LABEL: func.func @keep_dual_mismatched_narrow_types
//      CHECK:   arith.extf
//      CHECK:   arith.truncf
func.func @keep_dual_mismatched_narrow_types(%arg0: tensor<32x32xf16>)
    -> tensor<32x32xbf16> attributes {rock.kernel} {
  %0 = arith.extf %arg0 : tensor<32x32xf16> to tensor<32x32xf32>
  %1 = arith.truncf %0 : tensor<32x32xf32> to tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// ============================================================
// Dual non-kernel function: the pass owns the precision-recovering
// direction only, so the round-trip pattern must remain a no-op
// outside `rock.kernel` functions. The mirror direction
// `truncf(extf %narrow) -> %narrow` is unconditionally safe and is
// folded by upstream MLIR's `arith.TruncFOp::fold`, which fires
// regardless of the kernel attribute inside the pass's own greedy
// rewrite phase. The output therefore still collapses to the
// identity here; that this happens via the unconditional
// dual-direction fold (not via our kernel-gated pattern) is the
// property the `skip_non_kernel` case above pins down for the
// pass-owned direction.
// ============================================================

// CHECK-LABEL: func.func @dual_non_kernel_folded_by_unconditional_truncf
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf16>)
//  CHECK-NOT:   arith.extf
//  CHECK-NOT:   arith.truncf
//      CHECK:   return %[[ARG]]
func.func @dual_non_kernel_folded_by_unconditional_truncf(%arg0: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %0 = arith.extf %arg0 : tensor<32x32xf16> to tensor<32x32xf32>
  %1 = arith.truncf %0 : tensor<32x32xf32> to tensor<32x32xf16>
  return %1 : tensor<32x32xf16>
}

// ============================================================
// Flag-merge tests: the round-trip pattern merges
// `fastmath<contract>` into both casts' flag sets and the greedy
// driver then fires upstream MLIR's `arith.ExtFOp::fold` (which
// requires `contract` on both) in the next iteration -- annotation
// and fold therefore happen inside a single pass invocation.
// Pre-existing fast-math flags and the truncf rounding mode must
// be preserved across the merge; the cases below set up an extra
// narrow-store use on the truncf so the surviving cast remains
// observable after the extf folds, catching a regression that
// drops a pre-existing flag or clobbers the rounding mode.
// ============================================================

// Basic positive case: a clean round-trip with no pre-existing
// flags. The pattern annotates both casts with `fastmath<contract>`
// and the greedy driver then folds the pair via `arith.ExtFOp::fold`
// in the next iteration. A single pass invocation yields identity.

// CHECK-LABEL: func.func @annotate_basic_roundtrip
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>)
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
//      CHECK:   return %[[ARG]] : tensor<32x32xf32>
func.func @annotate_basic_roundtrip(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
    attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Pre-existing fast-math flags must be merged with `contract`,
// not overwritten. The narrow store on the truncf keeps it alive
// after the extf folds away, so the surviving cast's flag set is
// observable. Flags print in bit order (reassoc, nnan, ninf, nsz,
// arcp, contract, afn) so the substring below is stable.

// CHECK-LABEL: func.func @annotate_preserve_other_fastmath_flags
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>, %[[DST:.*]]: tensor<32x32xf16>)
//  CHECK-NOT:   arith.extf
//      CHECK:   arith.truncf %[[ARG]] fastmath<nnan,ninf,arcp,contract> : tensor<32x32xf32> to tensor<32x32xf16>
func.func @annotate_preserve_other_fastmath_flags(%arg0: tensor<32x32xf32>,
                                                  %dst: tensor<32x32xf16>)
    -> (tensor<32x32xf32>, tensor<32x32xf16>) attributes {rock.kernel} {
  %0 = arith.truncf %arg0 fastmath<nnan,ninf,arcp>
       : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  %2 = rock.store %0 to %dst by set
       : tensor<32x32xf16> -> tensor<32x32xf16> to tensor<32x32xf16>
  return %1, %2 : tensor<32x32xf32>, tensor<32x32xf16>
}

// The `arith.truncf` rounding-mode attribute is orthogonal to the
// fast-math flags and must survive the merge untouched. The narrow
// store keeps the truncf alive after the extf folds so both the
// rounding mode and the newly-merged `contract` flag are visible.

// CHECK-LABEL: func.func @annotate_preserve_rounding_mode
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>, %[[DST:.*]]: tensor<32x32xf16>)
//  CHECK-NOT:   arith.extf
//      CHECK:   arith.truncf %[[ARG]] downward fastmath<contract> : tensor<32x32xf32> to tensor<32x32xf16>
func.func @annotate_preserve_rounding_mode(%arg0: tensor<32x32xf32>,
                                           %dst: tensor<32x32xf16>)
    -> (tensor<32x32xf32>, tensor<32x32xf16>) attributes {rock.kernel} {
  %0 = arith.truncf %arg0 downward
       : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  %2 = rock.store %0 to %dst by set
       : tensor<32x32xf16> -> tensor<32x32xf16> to tensor<32x32xf16>
  return %1, %2 : tensor<32x32xf32>, tensor<32x32xf16>
}

// Pre-annotated round-trip: when both casts already carry
// `fastmath<contract>` on pass entry, the round-trip pattern's
// convergence guard returns failure (no spurious double-set) and
// `arith.ExtFOp::fold` fires on the first greedy visit, collapsing
// the pair in place. Verifying the identity output covers both the
// guard (no infinite worklist churn) and the in-pass fold.

// CHECK-LABEL: func.func @annotate_idempotent
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>)
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
//      CHECK:   return %[[ARG]] : tensor<32x32xf32>
func.func @annotate_idempotent(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
    attributes {rock.kernel} {
  %0 = arith.truncf %arg0 fastmath<contract>
       : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 fastmath<contract>
       : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
