// RUN: rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope --split-input-file %s | FileCheck %s

// The pass should rewrite the convolution into
// an im2col-style transpose + collapse, followed by a 4-dim batched matmul
// linalg.generic, plus a final expand_shape that restores the [N,H,W,G,C]
// output layout.

#input_map  = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3 + d6, d4 + d7, d1, d5)>
#filter_map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5, d6, d7, d2)>
#output_map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3, d4, d1, d2)>

// CHECK-LABEL: func.func @conv_bwd_data_cpu_3x3(
// CHECK-SAME:    %[[INPUT:[A-Za-z0-9_]+]]: tensor<1x4x4x1x2xf32>,
// CHECK-SAME:    %[[FILTER:[A-Za-z0-9_]+]]: tensor<1x2x3x3x2xf32>,
// CHECK-SAME:    %[[INIT:[A-Za-z0-9_]+]]: tensor<1x2x2x1x2xf32>)

// The original 8-dim conv linalg.generic must be gone.
// CHECK-NOT: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]

// A zero f32 constant is reused to fill the GEMM accumulator below.
// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32

// im2col materializes an explicit [G, N, Ho, Wo, C, Fh, Fw] tensor sourced
// from the padded input, then collapses it to a 3-D LHS.
// CHECK:      %[[IM2COL_INIT:.+]] = tensor.empty() : tensor<1x1x2x2x2x3x3xf32>
// CHECK:      %[[IM2COL:.+]] = linalg.generic
// CHECK-SAME:   ins(%[[INPUT]] : tensor<1x4x4x1x2xf32>)
// CHECK-SAME:   outs(%[[IM2COL_INIT]] : tensor<1x1x2x2x2x3x3xf32>)
// CHECK:      %[[LHS:.+]] = tensor.collapse_shape %[[IM2COL]]
// CHECK-SAME:   tensor<1x1x2x2x2x3x3xf32> into tensor<1x4x18xf32>

// Filter is collapsed to a 3-D RHS that matches the GEMM contraction dim.
// CHECK:      %[[RHS:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:   tensor<1x2x3x3x2xf32> into tensor<1x18x2xf32>

// GEMM accumulator is a fresh zero-filled 1x4x2 tensor.
// CHECK:      %[[ACC_INIT:.+]] = tensor.empty() : tensor<1x4x2xf32>
// CHECK:      %[[ACC:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[ACC_INIT]] : tensor<1x4x2xf32>)

// The contraction is now a 4-dim parallel/parallel/parallel/reduction matmul
// over (g, m, n, k).
// CHECK:      %[[GEMM:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[LHS]], %[[RHS]] : tensor<1x4x18xf32>, tensor<1x18x2xf32>)
// CHECK-SAME:   outs(%[[ACC]] : tensor<1x4x2xf32>)
// CHECK:        arith.mulf
// CHECK:        arith.addf
// CHECK:        linalg.yield {{.*}} : f32
// CHECK:      } -> tensor<1x4x2xf32>

// Result expands back to [G, N, Ho, Wo, C] and is transposed to the original
// [N, Ho, Wo, G, C] output layout.
// CHECK:      %[[EXPANDED:.+]] = tensor.expand_shape %[[GEMM]]
// CHECK-SAME:   tensor<1x4x2xf32> into tensor<1x1x2x2x2xf32>
// CHECK:      linalg.generic
// CHECK-SAME:   ins(%[[EXPANDED]] : tensor<1x1x2x2x2xf32>)
// CHECK-SAME:   outs({{.*}} : tensor<1x2x2x1x2xf32>)

// Testcase extracted from:
// ./build/bin/rocmlir-gen -batchsize=64 -in_channels=512 -in_h=16 -in_w=16 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t bf16
func.func @conv_bwd_data_cpu_3x3(%arg0: tensor<1x4x4x1x2xf32>, %arg1: tensor<1x2x3x3x2xf32>, %arg2: tensor<1x2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32> attributes {rock.cpu_verifier} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<1x2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %1 = linalg.generic {indexing_maps = [#input_map, #filter_map, #output_map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x4x4x1x2xf32>, tensor<1x2x3x3x2xf32>) outs(%0 : tensor<1x2x2x1x2xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.mulf %in, %in_0 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x2x2x1x2xf32>
  return %1 : tensor<1x2x2x1x2xf32>
}

// -----

// Larger 7x7 backward-data convolution (ImageNet-style first-layer shape):
//   N=256, G=1, K=64 (in-channels of fwd), C=3 (out-channels of fwd),
//   Fh=Fw=7, padded input 256x236x236, output 256x230x230.

#input_map_big  = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3 + d6, d4 + d7, d1, d5)>
#filter_map_big = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5, d6, d7, d2)>
#output_map_big = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3, d4, d1, d2)>

// CHECK-LABEL: func.func @conv_bwd_data_cpu_7x7(
// CHECK-SAME:    %[[INPUT:[A-Za-z0-9_]+]]: tensor<256x236x236x1x64xf32>,
// CHECK-SAME:    %[[FILTER:[A-Za-z0-9_]+]]: tensor<1x64x7x7x3xf32>,
// CHECK-SAME:    %[[INIT:[A-Za-z0-9_]+]]: tensor<256x230x230x1x3xf32>)

// CHECK-NOT: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]

// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32

// im2col expands the padded input into a 7-D tensor [G, N, Ho, Wo, K, Fh, Fw],
// then collapses to a 3-D LHS of shape [G, N*Ho*Wo, K*Fh*Fw] = [1, 13542400, 3136].
// CHECK:      %[[IM2COL_INIT:.+]] = tensor.empty() : tensor<1x256x230x230x64x7x7xf32>
// CHECK:      %[[IM2COL:.+]] = linalg.generic
// CHECK-SAME:   ins(%[[INPUT]] : tensor<256x236x236x1x64xf32>)
// CHECK-SAME:   outs(%[[IM2COL_INIT]] : tensor<1x256x230x230x64x7x7xf32>)
// CHECK:      %[[LHS:.+]] = tensor.collapse_shape %[[IM2COL]]
// CHECK-SAME:   tensor<1x256x230x230x64x7x7xf32> into tensor<1x13542400x3136xf32>

// Filter collapses to [G, K*Fh*Fw, C] = [1, 3136, 3].
// CHECK:      %[[RHS:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:   tensor<1x64x7x7x3xf32> into tensor<1x3136x3xf32>

// CHECK:      %[[ACC_INIT:.+]] = tensor.empty() : tensor<1x13542400x3xf32>
// CHECK:      %[[ACC:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[ACC_INIT]] : tensor<1x13542400x3xf32>)

// CHECK:      %[[GEMM:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[LHS]], %[[RHS]] : tensor<1x13542400x3136xf32>, tensor<1x3136x3xf32>)
// CHECK-SAME:   outs(%[[ACC]] : tensor<1x13542400x3xf32>)
// CHECK:        arith.mulf
// CHECK:        arith.addf
// CHECK:        linalg.yield {{.*}} : f32
// CHECK:      } -> tensor<1x13542400x3xf32>

// Result expands back to [G, N, Ho, Wo, C] and is transposed to the original
// [N, Ho, Wo, G, C] output layout.
// CHECK:      %[[EXPANDED:.+]] = tensor.expand_shape %[[GEMM]]
// CHECK-SAME:   tensor<1x13542400x3xf32> into tensor<1x256x230x230x3xf32>
// CHECK:      linalg.generic
// CHECK-SAME:   ins(%[[EXPANDED]] : tensor<1x256x230x230x3xf32>)
// CHECK-SAME:   outs({{.*}} : tensor<256x230x230x1x3xf32>)

// Testcase extracted from:
// ./build/bin/rocmlir-gen --cpu-timers -batchsize=256 -in_channels=3 -in_h=230 -in_w=230 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=1 --padding_w=1 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32  --arch gfx942 -pv
func.func @conv_bwd_data_cpu_7x7(%arg0: tensor<256x236x236x1x64xf32>, %arg1: tensor<1x64x7x7x3xf32>, %arg2: tensor<256x230x230x1x3xf32>) -> tensor<256x230x230x1x3xf32> attributes {rock.cpu_verifier} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<256x230x230x1x3xf32>) -> tensor<256x230x230x1x3xf32>
  %1 = linalg.generic {indexing_maps = [#input_map_big, #filter_map_big, #output_map_big], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<256x236x236x1x64xf32>, tensor<1x64x7x7x3xf32>) outs(%0 : tensor<256x230x230x1x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.mulf %in, %in_0 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<256x230x230x1x3xf32>
  return %1 : tensor<256x230x230x1x3xf32>
}

