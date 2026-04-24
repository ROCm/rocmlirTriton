// RUN: rocmlir-opt -rock-regularize-inter-gemm-fusion -split-input-file %s | FileCheck %s

#gemm_params = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// Reshape arg0 (32x256x256) up to (1x32x256x256) inside the body so it can be
// added with the bias %arg4 (1x32x256x256). The bias chain is empty (it is
// used directly by arith.addf), which used to trigger a self-RAUW assertion
// in externalizeBodyTransforms ("cannot RAUW a value with itself"). After
// the pass, the body must contain no rock.transform ops, the inverse of the
// arg0 transform must be applied to the bias externally, and the bias block
// argument must be retyped to the body's working shape (32x256x256).

#tf_addgroup = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
  by [<Unmerge{1} ["exp1"] at [1] -> ["dim0"] at [0]>,
      <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>,
      <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>,
      <AddDim{1} ["unit0"] at [0] -> [] at []>]
  bounds = [1, 1, 256, 256] -> [1, 256, 256]>

// CHECK-LABEL: func.func @attn_arg0_only_transform_bias_passthrough
// CHECK-SAME:    %[[Q:.*0]]: tensor<1x256x64xf32>
// CHECK-SAME:    %[[K:.*1]]: tensor<1x64x256xf32>
// CHECK-SAME:    %[[V:.*2]]: tensor<1x256x64xf32>
// CHECK-SAME:    %[[BIAS:.*3]]: tensor<1x1x256x256xf32>
// The body's transform on arg0 is externalized: bias gets the inverse transform
// (4D -> 3D) so the body can operate in the post-first-GEMM shape.
// CHECK:       rock.transform %[[BIAS]]
// CHECK-SAME:    : tensor<1x1x256x256xf32> to tensor<1x256x256xf32>
// CHECK:       rock.attention
// CHECK:         qk = elementwise otherIns(%{{.*}} : tensor<1x256x256xf32>)
// CHECK-NEXT:    ^bb0(%[[QK:.*]]: tensor<1x256x256xf32>, %[[BIAS_ARG:.*]]: tensor<1x256x256xf32>):
// The body must contain no rock.transform; the addf operates in arg0's shape.
// CHECK-NOT:       rock.transform
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[QK]], %[[BIAS_ARG]] : tensor<1x256x256xf32>
// CHECK-NEXT:      rock.yield %[[ADD]] : tensor<1x256x256xf32>
func.func @attn_arg0_only_transform_bias_passthrough(
    %q: tensor<1x256x64xf32>, %k: tensor<1x64x256xf32>,
    %v: tensor<1x256x64xf32>, %bias: tensor<1x1x256x256xf32>)
    -> tensor<1x256x64xf32> attributes {
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx1100"
    } {
  %result = rock.attention {
    qk = %q * %k : tensor<1x256x64xf32>, tensor<1x64x256xf32>
    qk = elementwise otherIns(%bias : tensor<1x1x256x256xf32>) {
    ^bb0(%qk: tensor<1x256x256xf32>, %bias_in: tensor<1x1x256x256xf32>):
      %qk_4d = rock.transform %qk by #tf_addgroup
        : tensor<1x256x256xf32> to tensor<1x1x256x256xf32>
      %sum = arith.addf %qk_4d, %bias_in : tensor<1x1x256x256xf32>
      rock.yield %sum : tensor<1x1x256x256xf32>
    }
    softmax(qk) * %v : tensor<1x256x64xf32>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32,
     params0 = #gemm_params, params1 = #gemm_params}
    -> tensor<1x256x64xf32>
  return %result : tensor<1x256x64xf32>
}

// -----

// Both arg0 and the otherIns have a single transform inside the body. After
// the pass each body transform should be externalized, the body should be
// transform-free, and both block arguments should be retyped to arg0's
// working shape.

#gemm_params = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

#tf_addgroup = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
  by [<Unmerge{1} ["exp1"] at [1] -> ["dim0"] at [0]>,
      <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>,
      <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>,
      <AddDim{1} ["unit0"] at [0] -> [] at []>]
  bounds = [1, 1, 256, 256] -> [1, 256, 256]>

#tf_bias_addgroup = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
  by [<Unmerge{1} ["exp1"] at [1] -> ["dim0"] at [0]>,
      <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>,
      <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>,
      <AddDim{1} ["unit0"] at [0] -> [] at []>]
  bounds = [1, 1, 256, 256] -> [1, 256, 256]>

// CHECK-LABEL: func.func @attn_both_args_have_transforms
// CHECK-SAME:    %[[Q:.*0]]: tensor<1x256x64xf32>
// CHECK-SAME:    %[[K:.*1]]: tensor<1x64x256xf32>
// CHECK-SAME:    %[[V:.*2]]: tensor<1x256x64xf32>
// CHECK-SAME:    %[[BIAS:.*3]]: tensor<1x256x256xf32>
// CHECK:       %[[BIAS_T0:.*]] = rock.transform %[[BIAS]]
// CHECK-SAME:    : tensor<1x256x256xf32> to tensor<1x1x256x256xf32>
// CHECK:       %[[BIAS_T1:.*]] = rock.transform %[[BIAS_T0]]
// CHECK-SAME:    : tensor<1x1x256x256xf32> to tensor<1x256x256xf32>
// CHECK:       rock.attention
// CHECK:         qk = elementwise otherIns(%[[BIAS_T1]] : tensor<1x256x256xf32>)
// CHECK-NEXT:    ^bb0(%[[QK:.*]]: tensor<1x256x256xf32>, %[[BIAS_ARG:.*]]: tensor<1x256x256xf32>):
// CHECK-NOT:       rock.transform
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[QK]], %[[BIAS_ARG]] : tensor<1x256x256xf32>
// CHECK-NEXT:      rock.yield %[[ADD]] : tensor<1x256x256xf32>
func.func @attn_both_args_have_transforms(
    %q: tensor<1x256x64xf32>, %k: tensor<1x64x256xf32>,
    %v: tensor<1x256x64xf32>, %bias: tensor<1x256x256xf32>)
    -> tensor<1x256x64xf32> attributes {
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx1100"
    } {
  %result = rock.attention {
    qk = %q * %k : tensor<1x256x64xf32>, tensor<1x64x256xf32>
    qk = elementwise otherIns(%bias : tensor<1x256x256xf32>) {
    ^bb0(%qk: tensor<1x256x256xf32>, %bias_in: tensor<1x256x256xf32>):
      %qk_4d = rock.transform %qk by #tf_addgroup
        : tensor<1x256x256xf32> to tensor<1x1x256x256xf32>
      %bias_4d = rock.transform %bias_in by #tf_bias_addgroup
        : tensor<1x256x256xf32> to tensor<1x1x256x256xf32>
      %sum = arith.addf %qk_4d, %bias_4d : tensor<1x1x256x256xf32>
      rock.yield %sum : tensor<1x1x256x256xf32>
    }
    softmax(qk) * %v : tensor<1x256x64xf32>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32,
     params0 = #gemm_params, params1 = #gemm_params}
    -> tensor<1x256x64xf32>
  return %result : tensor<1x256x64xf32>
}
