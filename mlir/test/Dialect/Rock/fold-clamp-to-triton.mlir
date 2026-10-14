// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-fold-clamp-to-triton --split-input-file | FileCheck %s

// rock-fold-clamp-to-triton only looks at arith.minnumf/maxnumf that already carry
// `nnan`, which in the real pipeline rock-allow-fast-math-flags attaches first.
// The inputs here spell the flags out so the pass is exercised on its own.

// Test: a `maxnumf(minnumf(x, hi), lo)` clamp whose operands may ignore NaNs
// becomes a single tt.clampf, which AMD lowers to one v_med3.

// CHECK-LABEL: @test_clamp_nnan_to_clampf
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf32>)
//  CHECK-DAG:   %[[LO:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
//  CHECK-DAG:   %[[HI:.*]] = arith.constant dense<6.000000e+00> : tensor<64xf32>
//      CHECK:   %[[RESULT:.*]] = tt.clampf %[[INPUT]], %[[LO]], %[[HI]], propagateNan = none : tensor<64xf32>
//  CHECK-NOT:   arith.minnumf
//  CHECK-NOT:   arith.maxnumf
func.func @test_clamp_nnan_to_clampf(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: the same idiom on f16, the other element type AMD contracts to v_med3.

// CHECK-LABEL: @test_clamp_f16_to_clampf
//      CHECK:   tt.clampf {{.*}}propagateNan = none : tensor<64xf16>
//  CHECK-NOT:   arith.minnumf
func.func @test_clamp_f16_to_clampf(%arg0: tensor<64xf16>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf16>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf16>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf16>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf16>
  return %1 : tensor<64xf16>
}

// -----

// Test: without `nnan` the pair stays put. minnumf/maxnumf send a NaN input to
// `hi`, whereas v_med3 would return `lo`, so the rewrite is not equivalent.

// CHECK-LABEL: @test_clamp_no_nnan_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_no_nnan_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: `nnan` on only one of the two ops is not enough.

// CHECK-LABEL: @test_clamp_partial_nnan_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_partial_nnan_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: the NaN-propagating spelling is left alone. A tt.clampf with
// propagateNan = none would not be a faithful replacement for it.

// CHECK-LABEL: @test_clamp_propagating_ops_unchanged
//      CHECK:   arith.minimumf
//      CHECK:   arith.maximumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_propagating_ops_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minimumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maximumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: unordered bounds are rejected. For lo > hi the pair saturates to `lo`
// while v_med3 would return `hi`.

// CHECK-LABEL: @test_clamp_unordered_bounds_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_unordered_bounds_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: non-constant bounds cannot be proven ordered, so the pair stays put.

// CHECK-LABEL: @test_clamp_dynamic_bound_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_dynamic_bound_unchanged(%arg0: tensor<64xf32>, %hi: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: bf16 converts too. AMD's v_med3 path skips it, but Triton's generic
// lowering emits the min-outer form that LLVM can contract into fmed3.

// CHECK-LABEL: @test_clamp_bf16_to_clampf
//      CHECK:   tt.clampf {{.*}}propagateNan = none : tensor<64xbf16>
//  CHECK-NOT:   arith.minnumf
func.func @test_clamp_bf16_to_clampf(%arg0: tensor<64xbf16>) -> tensor<64xbf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xbf16>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xbf16>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xbf16>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xbf16>
  return %1 : tensor<64xbf16>
}

// -----

// Test: f64 converts as well, for the same reason as bf16.

// CHECK-LABEL: @test_clamp_f64_to_clampf
//      CHECK:   tt.clampf {{.*}}propagateNan = none : tensor<64xf64>
//  CHECK-NOT:   arith.minnumf
func.func @test_clamp_f64_to_clampf(%arg0: tensor<64xf64>) -> tensor<64xf64> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf64>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf64>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf64>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf64>
  return %1 : tensor<64xf64>
}

// -----

// Test: FP8 is rejected. tt.clampf would accept it, but Triton maps FP8 to i8
// and the lowering would end up comparing integers.

// CHECK-LABEL: @test_clamp_f8_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_f8_unchanged(%arg0: tensor<64xf8E4M3FN>) -> tensor<64xf8E4M3FN> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf8E4M3FN>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf8E4M3FN>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf8E4M3FN>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf8E4M3FN>
  return %1 : tensor<64xf8E4M3FN>
}

// -----

// Test: a min feeding a second consumer is left alone, since folding it into
// the clamp would duplicate it rather than remove it.

// CHECK-LABEL: @test_clamp_multi_use_min_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_multi_use_min_unchanged(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1, %0 : tensor<64xf32>, tensor<64xf32>
}

// -----

// Test: the min-outer spelling `minnumf(maxnumf(x, lo), hi)`, which is what
// migraphx.clip lowers to, folds just like the max-outer one. Under nnan the
// two orderings agree.

// CHECK-LABEL: @test_clamp_min_outer_to_clampf
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf32>)
//  CHECK-DAG:   %[[LO:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
//  CHECK-DAG:   %[[HI:.*]] = arith.constant dense<6.000000e+00> : tensor<64xf32>
//      CHECK:   %[[RESULT:.*]] = tt.clampf %[[INPUT]], %[[LO]], %[[HI]], propagateNan = none : tensor<64xf32>
//  CHECK-NOT:   arith.minnumf
//  CHECK-NOT:   arith.maxnumf
func.func @test_clamp_min_outer_to_clampf(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.maxnumf %arg0, %lo fastmath<nnan> : tensor<64xf32>
  %1 = arith.minnumf %0, %hi fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: the min-outer spelling still needs nnan on both operations.

// CHECK-LABEL: @test_clamp_min_outer_no_nnan_unchanged
//      CHECK:   arith.maxnumf
//      CHECK:   arith.minnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_min_outer_no_nnan_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.maxnumf %arg0, %lo : tensor<64xf32>
  %1 = arith.minnumf %0, %hi : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: unordered bounds are rejected in the min-outer spelling too, where the
// original saturates to hi while v_med3 would return lo.

// CHECK-LABEL: @test_clamp_min_outer_unordered_bounds_unchanged
//      CHECK:   arith.maxnumf
//      CHECK:   arith.minnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_min_outer_unordered_bounds_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %0 = arith.maxnumf %arg0, %lo fastmath<nnan> : tensor<64xf32>
  %1 = arith.minnumf %0, %hi fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

