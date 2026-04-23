// Unit tests for rock-regularize-inter-gemm-fusion pass.
//
// The pass canonicalizes the elementwise body of a `rock.gemm_elementwise_gemm`
// op so that it contains only pure arith/math elementwise ops in arg0's shape.
// All `rock.transform` ops that lived inside the body get hoisted out to the
// op's external operands (composed with the inverse of arg0's chain), so the
// lowered body can be tiled and streamed by the gridwise/blockwise lowerings.
//
// In particular, when the body is a DAG (an intermediate value with multiple
// consumers, e.g. a residual that feeds both a downstream mul and a sigmoid),
// the pass MUST NOT clone the multi-use sub-DAG. The yield-boundary transform
// is erased up front in `eraseYieldBoundaryTransform`, after which all
// remaining body transforms are linear chains rooted at block args; the body
// is then converted to arg0's shape entirely by type substitution. See
// `RegularizeInterGemmFusion.cpp` for details.

// RUN: rocmlir-opt -rock-regularize-inter-gemm-fusion -split-input-file %s | FileCheck %s

// ============================================================
// Common transform maps used across the test cases below.
//
//   tmap_arg0      : 1x4x4 -> 4x4   (Merge unit dim 0, PassThrough n)
//   tmap_yield     : 4x4   -> 1x4x4 (inverse of tmap_arg0)
//   tmap_bias_addim: 4     -> 1x4   (AddDim unit g, PassThrough n)
//   tmap_bias_bcast: 1x4   -> 4x4   (Broadcast unit g to m, PassThrough n)
// ============================================================

#tmap_arg0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d0, d1)>
  by [<Merge{1, 4} ["m"] at [0] -> ["g", "m_in"] at [0, 1]>,
      <PassThrough ["n"] at [1] -> ["n_in"] at [2]>]
  bounds = [4, 4] -> [1, 4, 4]>

#tmap_yield = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d2)>
  by [<Unmerge{4} ["m_in"] at [1] -> ["m"] at [0]>,
      <PassThrough ["n_in"] at [2] -> ["n"] at [1]>,
      <AddDim{1} ["g"] at [0] -> [] at []>]
  bounds = [1, 4, 4] -> [4, 4]>

#tmap_bias_addim = #rock.transform_map<affine_map<(d0, d1) -> (d1)>
  by [<Unmerge{4} ["n"] at [1] -> ["dim0"] at [0]>,
      <AddDim{1} ["g"] at [0] -> [] at []>]
  bounds = [1, 4] -> [4]>

#tmap_bias_bcast = #rock.transform_map<affine_map<(d0, d1) -> (0, d1)>
  by [<Broadcast{1} ["m"] at [0] -> ["g"] at [0]>,
      <PassThrough ["n"] at [1] -> ["n"] at [1]>]
  bounds = [4, 4] -> [1, 4]>

#gemm_params0 = #rock.gemm_params<mPerBlock = 4, nPerBlock = 4, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params1 = #rock.gemm_params<mPerBlock = 4, nPerBlock = 2, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// DAG body — regression test for the original failing case.
//
// The intermediate %v = addf(...) is consumed by BOTH %x = mulf(%v, ...) and
// %y = mulf(..., %v). Without the up-front yield-boundary erase, the legacy
// `sinkTransformsToLeaves` would push the boundary transform inward through
// the multi-use %v, leaving block-arg-rooted transforms with multiple
// TransformOp users. `collectArgTransformChains` would then reject the body
// with "non-linear transform chain". With the fix, the body is regularized
// without any SSA cloning: %v stays multi-use, and the body's DAG structure
// is preserved exactly, just retyped from 4x4 to 1x4x4.
// ============================================================

// CHECK-LABEL: func.func @dag_body
// CHECK-SAME:   (%[[A:.*]]: tensor<1x4x4xf32>, %[[B:.*]]: tensor<1x4x4xf32>,
// CHECK-SAME:    %[[C:.*]]: tensor<1x4x2xf32>, %[[BIAS:.*]]: tensor<4xf32>)
//
// The bias chain has been hoisted OUTSIDE the gemm op as one transform per
// link, plus the inverse of arg0's chain at the end so the external operand
// lands in arg0's shape (1x4x4xf32).
// CHECK: %[[B0:.*]] = rock.transform %[[BIAS]]      {{.*}} : tensor<4xf32>     to tensor<1x4xf32>
// CHECK: %[[B1:.*]] = rock.transform %[[B0]]        {{.*}} : tensor<1x4xf32>   to tensor<4x4xf32>
// CHECK: %[[B2:.*]] = rock.transform %[[B1]]        {{.*}} : tensor<4x4xf32>   to tensor<1x4x4xf32>
//
// The gemm_elementwise_gemm op sees the externalized bias; its block arg has
// been retyped from tensor<4xf32> to arg0's shape tensor<1x4x4xf32>.
// CHECK: rock.gemm_elementwise_gemm
// CHECK: ab = %[[A]] * %[[B]]
// CHECK: ab = elementwise otherIns(%[[B2]] : tensor<1x4x4xf32>)
// CHECK-NEXT: ^bb0(%[[ARG8:.*]]: tensor<1x4x4xf32>, %[[ARG9:.*]]: tensor<1x4x4xf32>):
//
// Body is now transform-free, in arg0's shape, with the multi-use %v
// preserved exactly (no cloning).
// CHECK-NOT: rock.transform
// CHECK: %[[V:.*]]  = arith.addf %[[ARG9]], %[[ARG8]] : tensor<1x4x4xf32>
// CHECK: %[[X:.*]]  = arith.mulf %[[V]], %[[ARG9]]   : tensor<1x4x4xf32>
// CHECK: %[[Y:.*]]  = arith.mulf %[[X]], %[[V]]      : tensor<1x4x4xf32>
// CHECK: rock.yield %[[Y]] : tensor<1x4x4xf32>
// CHECK: out = ab * %[[C]] : tensor<1x4x2xf32>
func.func @dag_body(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>,
                    %c: tensor<1x4x2xf32>, %bias: tensor<4xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "gfx950", rock.block_size = 64 : i32, rock.kernel} {
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise otherIns(%bias : tensor<4xf32>) {
  ^bb0(%arg8: tensor<1x4x4xf32>, %arg9: tensor<4xf32>):
    %a8 = rock.transform %arg8 by #tmap_arg0 : tensor<1x4x4xf32> to tensor<4x4xf32>
    %a9_1d = rock.transform %arg9 by #tmap_bias_addim : tensor<4xf32> to tensor<1x4xf32>
    %a9 = rock.transform %a9_1d by #tmap_bias_bcast : tensor<1x4xf32> to tensor<4x4xf32>
    %v = arith.addf %a9, %a8 : tensor<4x4xf32>
    %x = arith.mulf %v, %a9 : tensor<4x4xf32>
    %y = arith.mulf %x, %v : tensor<4x4xf32>
    %y_back = rock.transform %y by #tmap_yield : tensor<4x4xf32> to tensor<1x4x4xf32>
    rock.yield %y_back : tensor<1x4x4xf32>
   }
   out = ab * %c : tensor<1x4x2xf32>
  } {params0 = #gemm_params0, params1 = #gemm_params1} -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// -----

// ============================================================
// Tree body — sanity check that ordinary tree-shaped bodies still regularize
// correctly. No multi-use intermediate, just a single addf into the yield.
// The fix's no-op fast paths must not regress this.
// ============================================================

#tmap_arg0_t = #rock.transform_map<affine_map<(d0, d1) -> (0, d0, d1)>
  by [<Merge{1, 4} ["m"] at [0] -> ["g", "m_in"] at [0, 1]>,
      <PassThrough ["n"] at [1] -> ["n_in"] at [2]>]
  bounds = [4, 4] -> [1, 4, 4]>

#tmap_yield_t = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d2)>
  by [<Unmerge{4} ["m_in"] at [1] -> ["m"] at [0]>,
      <PassThrough ["n_in"] at [2] -> ["n"] at [1]>,
      <AddDim{1} ["g"] at [0] -> [] at []>]
  bounds = [1, 4, 4] -> [4, 4]>

#tmap_bias_addim_t = #rock.transform_map<affine_map<(d0, d1) -> (d1)>
  by [<Unmerge{4} ["n"] at [1] -> ["dim0"] at [0]>,
      <AddDim{1} ["g"] at [0] -> [] at []>]
  bounds = [1, 4] -> [4]>

#tmap_bias_bcast_t = #rock.transform_map<affine_map<(d0, d1) -> (0, d1)>
  by [<Broadcast{1} ["m"] at [0] -> ["g"] at [0]>,
      <PassThrough ["n"] at [1] -> ["n"] at [1]>]
  bounds = [4, 4] -> [1, 4]>

#gemm_params0_t = #rock.gemm_params<mPerBlock = 4, nPerBlock = 4, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params1_t = #rock.gemm_params<mPerBlock = 4, nPerBlock = 2, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// CHECK-LABEL: func.func @tree_body
// CHECK: rock.transform %{{.*}} : tensor<4xf32>     to tensor<1x4xf32>
// CHECK: rock.transform %{{.*}} : tensor<1x4xf32>   to tensor<4x4xf32>
// CHECK: rock.transform %{{.*}} : tensor<4x4xf32>   to tensor<1x4x4xf32>
// CHECK: rock.gemm_elementwise_gemm
// CHECK: ^bb0(%[[ARG8:.*]]: tensor<1x4x4xf32>, %[[ARG9:.*]]: tensor<1x4x4xf32>):
// CHECK-NOT: rock.transform
// CHECK: %[[V:.*]] = arith.addf %[[ARG9]], %[[ARG8]] : tensor<1x4x4xf32>
// CHECK: rock.yield %[[V]] : tensor<1x4x4xf32>
func.func @tree_body(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>,
                     %c: tensor<1x4x2xf32>, %bias: tensor<4xf32>)
    -> tensor<1x4x2xf32>
    attributes {rock.arch = "gfx950", rock.block_size = 64 : i32, rock.kernel} {
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   ab = elementwise otherIns(%bias : tensor<4xf32>) {
  ^bb0(%arg8: tensor<1x4x4xf32>, %arg9: tensor<4xf32>):
    %a8 = rock.transform %arg8 by #tmap_arg0_t : tensor<1x4x4xf32> to tensor<4x4xf32>
    %a9_1d = rock.transform %arg9 by #tmap_bias_addim_t : tensor<4xf32> to tensor<1x4xf32>
    %a9 = rock.transform %a9_1d by #tmap_bias_bcast_t : tensor<1x4xf32> to tensor<4x4xf32>
    %v = arith.addf %a9, %a8 : tensor<4x4xf32>
    %v_back = rock.transform %v by #tmap_yield_t : tensor<4x4xf32> to tensor<1x4x4xf32>
    rock.yield %v_back : tensor<1x4x4xf32>
   }
   out = ab * %c : tensor<1x4x2xf32>
  } {params0 = #gemm_params0_t, params1 = #gemm_params1_t} -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}

// -----

// ============================================================
// No body fusion — the elementwise region is empty (no `ab = elementwise ...`
// clause). Pass should be a complete no-op on this op via the early-return in
// `regularizeGemmGemmBody` that calls `gemmGemmHasPreSecondGemmFusion`.
// ============================================================

#gemm_params0_n = #rock.gemm_params<mPerBlock = 4, nPerBlock = 4, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params1_n = #rock.gemm_params<mPerBlock = 4, nPerBlock = 2, kPerBlock = 4, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// CHECK-LABEL: func.func @no_fusion
// CHECK: %[[R:.*]] = rock.gemm_elementwise_gemm
// CHECK: ab = %{{.*}} * %{{.*}}
// CHECK-NOT: ab = elementwise
// CHECK: out = ab * %{{.*}}
// CHECK: return %[[R]]
func.func @no_fusion(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>,
                     %c: tensor<1x4x2xf32>) -> tensor<1x4x2xf32>
    attributes {rock.arch = "gfx950", rock.block_size = 64 : i32, rock.kernel} {
  %r = rock.gemm_elementwise_gemm{
   ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
   out = ab * %c : tensor<1x4x2xf32>
  } {params0 = #gemm_params0_n, params1 = #gemm_params1_n} -> tensor<1x4x2xf32>
  return %r : tensor<1x4x2xf32>
}
