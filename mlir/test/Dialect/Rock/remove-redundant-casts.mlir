// RUN: rocmlir-opt -rock-remove-redundant-casts -canonicalize -mlir-print-local-scope %s | FileCheck %s
// RUN: rocmlir-opt -rock-remove-redundant-casts -mlir-print-local-scope %s | FileCheck %s --check-prefix=ANNOTATE

// ============================================================
// Direct pure-SSA round-trip: extf(truncf %wide) -> %wide.
// The truncf result has exactly one use (the extf) and the wide
// types match. The pair must be folded away.
// ============================================================

// CHECK-LABEL: func.func @fold_direct_roundtrip
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>)
//      CHECK:   %[[MUL:.*]] = arith.mulf %[[ARG]], %[[ARG]]
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
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
//      CHECK:   %[[ADD:.*]] = arith.addf %[[ARG]], %[[ARG]]
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
//      CHECK:   return %[[ADD]]
func.func @fold_scalar_roundtrip(%arg0: f32) -> f32
    attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : f32 to f16
  %1 = arith.extf %0 : f16 to f32
  %2 = arith.addf %1, %arg0 : f32
  return %2 : f32
}

// ============================================================
// Multi-use of the narrow value: the truncf's result also feeds a
// narrow store. The extf still folds (its consumer is rewritten to
// use the wide source directly), but the truncf survives DCE
// because the store keeps it alive. No precision drop is introduced
// on the f32 side; the narrow store path is untouched.
// ============================================================

// CHECK-LABEL: func.func @fold_extf_keep_truncf_with_extra_use
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf32>, %[[DST:.*]]: tensor<32x32xf16>)
//      CHECK:   %[[TR:.*]] = arith.truncf %[[ARG]]
//  CHECK-NOT:   arith.extf
//      CHECK:   %[[MUL:.*]] = arith.mulf %[[ARG]], %[[ARG]]
//      CHECK:   rock.store %[[TR]] to %[[DST]]
func.func @fold_extf_keep_truncf_with_extra_use(%arg0: tensor<32x32xf32>,
                                       %dst: tensor<32x32xf16>)
    -> (tensor<32x32xf32>, tensor<32x32xf16>) attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  %2 = arith.mulf %1, %arg0 : tensor<32x32xf32>
  %3 = rock.store %0 to %dst by set
       : tensor<32x32xf16> -> tensor<32x32xf16> to tensor<32x32xf16>
  return %2, %3 : tensor<32x32xf32>, tensor<32x32xf16>
}

// ============================================================
// Mismatched wide types: f64 -> f16 -> f32 is genuine precision
// shaping (the user is widening to f32, not f64). Must NOT fold.
// ============================================================

// CHECK-LABEL: func.func @keep_mismatched_wide_types
//      CHECK:   arith.truncf
//      CHECK:   arith.extf
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
// Non-kernel function: pass must be a no-op even when the pattern
// matches, since rocmlirTriton uses the rock.kernel attribute to
// gate kernel-only rewrites.
// ============================================================

// CHECK-LABEL: func.func @skip_non_kernel
//      CHECK:   arith.truncf
//      CHECK:   arith.extf
func.func @skip_non_kernel(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = arith.truncf %arg0 : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
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
//      CHECK:   %[[CST_SCALE:.*]] = arith.constant {{.*}}0.0883
//  CHECK-NOT:   arith.truncf
//  CHECK-NOT:   arith.extf
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
//      CHECK:   %[[EX:.*]] = arith.extf %[[ARG]]
//  CHECK-NOT:   arith.truncf
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
// direction only, so it must remain a no-op outside `rock.kernel`
// functions. The mirror direction `truncf(extf %narrow) -> %narrow`
// is unconditionally safe and is folded by upstream MLIR's
// `arith.TruncFOp::fold`, which `-canonicalize` runs regardless of
// the kernel attribute. The combined output therefore collapses to
// the identity here; that this happens via canonicalize (not via
// our pass) is the property the dual `skip_non_kernel` case above
// pins down for the pass-owned direction.
// ============================================================

// CHECK-LABEL: func.func @dual_non_kernel_folded_by_canonicalize
// CHECK-SAME: (%[[ARG:.*]]: tensor<32x32xf16>)
//  CHECK-NOT:   arith.extf
//  CHECK-NOT:   arith.truncf
//      CHECK:   return %[[ARG]]
func.func @dual_non_kernel_folded_by_canonicalize(%arg0: tensor<32x32xf16>) -> tensor<32x32xf16> {
  %0 = arith.extf %arg0 : tensor<32x32xf16> to tensor<32x32xf32>
  %1 = arith.truncf %0 : tensor<32x32xf32> to tensor<32x32xf16>
  return %1 : tensor<32x32xf16>
}

// ============================================================
// Annotation-only tests. With canonicalize disabled, the pass
// must leave both casts in place but merge `fastmath<contract>`
// into their flag sets. These cases exercise the annotation
// behaviour directly so a future regression that drops the flag,
// overwrites pre-existing flags, or clobbers the rounding mode
// fails loudly.
// ============================================================

// Basic positive case: a clean round-trip with no pre-existing
// flags. The pass must add `fastmath<contract>` to both casts.

// ANNOTATE-LABEL: func.func @annotate_basic_roundtrip
// ANNOTATE-SAME: (%[[ARG:.*]]: tensor<32x32xf32>)
//      ANNOTATE: arith.truncf %[[ARG]] fastmath<contract> : tensor<32x32xf32> to tensor<32x32xf16>
//      ANNOTATE: arith.extf {{.*}} fastmath<contract> : tensor<32x32xf16> to tensor<32x32xf32>
func.func @annotate_basic_roundtrip(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
    attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Pre-existing fast-math flags must be preserved verbatim. The
// trunc keeps `nnan,ninf` and gains `contract`; the extf keeps
// `arcp` and gains `contract`. Flags print in bit order
// (reassoc, nnan, ninf, nsz, arcp, contract, afn) so the
// concrete substrings below are stable.

// ANNOTATE-LABEL: func.func @annotate_preserve_other_fastmath_flags
//      ANNOTATE: arith.truncf {{.*}} fastmath<nnan,ninf,contract> : tensor<32x32xf32> to tensor<32x32xf16>
//      ANNOTATE: arith.extf {{.*}} fastmath<arcp,contract> : tensor<32x32xf16> to tensor<32x32xf32>
func.func @annotate_preserve_other_fastmath_flags(%arg0: tensor<32x32xf32>)
    -> tensor<32x32xf32> attributes {rock.kernel} {
  %0 = arith.truncf %arg0 fastmath<nnan,ninf>
       : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 fastmath<arcp>
       : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// The `arith.truncf` rounding-mode attribute is orthogonal to the
// fast-math flags; it must survive the annotation untouched.

// ANNOTATE-LABEL: func.func @annotate_preserve_rounding_mode
//      ANNOTATE: arith.truncf {{.*}} downward fastmath<contract> : tensor<32x32xf32> to tensor<32x32xf16>
//      ANNOTATE: arith.extf {{.*}} fastmath<contract> : tensor<32x32xf16> to tensor<32x32xf32>
func.func @annotate_preserve_rounding_mode(%arg0: tensor<32x32xf32>)
    -> tensor<32x32xf32> attributes {rock.kernel} {
  %0 = arith.truncf %arg0 downward
       : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Idempotency: casts that already carry `fastmath<contract>` must
// come out of the pass with exactly the same flag set (no double
// bit, no spurious rewrite).

// ANNOTATE-LABEL: func.func @annotate_idempotent
//      ANNOTATE: arith.truncf {{.*}} fastmath<contract> : tensor<32x32xf32> to tensor<32x32xf16>
//      ANNOTATE: arith.extf {{.*}} fastmath<contract> : tensor<32x32xf16> to tensor<32x32xf32>
func.func @annotate_idempotent(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
    attributes {rock.kernel} {
  %0 = arith.truncf %arg0 fastmath<contract>
       : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 fastmath<contract>
       : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Mismatched wide types are not a real round-trip (f64 -> f16 -> f32
// is genuine precision shaping), so the pass must NOT annotate
// either cast. The `return` anchor bounds the NOT scope to this
// function body.

// ANNOTATE-LABEL: func.func @annotate_skip_mismatched_wide_types
//  ANNOTATE-NOT: fastmath
//      ANNOTATE: return
func.func @annotate_skip_mismatched_wide_types(%arg0: tensor<32x32xf64>)
    -> tensor<32x32xf32> attributes {rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<32x32xf64> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Non-kernel (host) functions are skipped entirely, so even a
// well-formed round-trip must come out unannotated. This pins
// down the kernel gate at the flag level.

// ANNOTATE-LABEL: func.func @annotate_skip_non_kernel
//  ANNOTATE-NOT: fastmath
//      ANNOTATE: return
func.func @annotate_skip_non_kernel(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = arith.truncf %arg0 : tensor<32x32xf32> to tensor<32x32xf16>
  %1 = arith.extf %0 : tensor<32x32xf16> to tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
