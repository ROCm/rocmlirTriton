// RUN: rocmlir-opt --migraphx-to-tosa %s | FileCheck %s

func.func @add_dynamic_m(%arg0: !migraphx.shaped<?x64xf32, 64x1>)
    -> !migraphx.shaped<?x64xf32, 64x1> {
  %0 = migraphx.add %arg0, %arg0 : <?x64xf32, 64x1>, <?x64xf32, 64x1> -> <?x64xf32, 64x1>
  return %0 : !migraphx.shaped<?x64xf32, 64x1>
}

// -----

func.func @dot_dynamic_m(%arg0: !migraphx.shaped<?x72xf32, 72x1>,
                         %arg1: !migraphx.shaped<72x64xf32, 64x1>)
    -> !migraphx.shaped<?x64xf32, 64x1> {
  %0 = migraphx.dot %arg0, %arg1 : <?x72xf32, 72x1>, <72x64xf32, 64x1> -> <?x64xf32, 64x1>
  return %0 : !migraphx.shaped<?x64xf32, 64x1>
}

// -----

// The batch is the slowest-moving axis of a convolution in both the NCHW
// logical shape and the NHWC memory layout, so it is the one that may be
// dynamic.
// CHECK-LABEL: func.func @conv_dynamic_batch
// CHECK-SAME: (%{{.*}}: tensor<?xf32>, %{{.*}}: tensor<9408xf32>) -> tensor<?xf32>
// CHECK: tosa.transpose {{.*}} -> tensor<?x224x224x3xf32>
// CHECK: tosa.conv2d {{.*}} : (tensor<?x224x224x3xf32>, tensor<64x7x7x3xf32>, {{.*}}) -> tensor<?x112x112x64xf32>
// CHECK: tosa.transpose {{.*}} -> tensor<?x64x112x112xf32>
func.func @conv_dynamic_batch(%arg0: !migraphx.shaped<?x3x224x224xf32, 150528x50176x224x1>,
                              %arg1: !migraphx.shaped<64x3x7x7xf32, 147x49x7x1>)
    -> !migraphx.shaped<?x64x112x112xf32, 802816x12544x112x1> {
  %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : <?x3x224x224xf32, 150528x50176x224x1>, <64x3x7x7xf32, 147x49x7x1> -> <?x64x112x112xf32, 802816x12544x112x1>
  return %0 : !migraphx.shaped<?x64x112x112xf32, 802816x12544x112x1>
}

// -----

// No padding case, which needs a different handling. MIGraphX drops the partial
// trailing window that TOSA's exact stride division would keep, and with no high
// padding to shrink, the extra row and column have to come off the input. That
// slice is the only place a dynamic extent reaches a tosa.slice size operand,
// where -1 means "the rest of the dimension" rather than "infer this one".
// CHECK-LABEL: func.func @conv_dynamic_batch_needs_slice
// CHECK: %[[SIZES:.*]] = tosa.const_shape {values = dense<[-1, 4, 4, 3]>
// CHECK: tosa.slice {{.*}}, %[[SIZES]] : (tensor<?x5x5x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x4x4x3xf32>
// CHECK: tosa.conv2d {{.*}} -> tensor<?x2x2x64xf32>
func.func @conv_dynamic_batch_needs_slice(%arg0: !migraphx.shaped<?x3x5x5xf32, 75x25x5x1>,
                                          %arg1: !migraphx.shaped<64x3x2x2xf32, 12x4x2x1>)
    -> !migraphx.shaped<?x64x2x2xf32, 256x4x2x1> {
  %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2]} : <?x3x5x5xf32, 75x25x5x1>, <64x3x2x2xf32, 12x4x2x1> -> <?x64x2x2xf32, 256x4x2x1>
  return %0 : !migraphx.shaped<?x64x2x2xf32, 256x4x2x1>
}

// -----

// CHECK-LABEL: func.func @conv1d_dynamic_batch
// CHECK: tosa.conv2d {{.*}} : (tensor<?x5x1x3xf32>, tensor<64x2x1x3xf32>, {{.*}}) -> tensor<?x4x1x64xf32>
func.func @conv1d_dynamic_batch(%arg0: !migraphx.shaped<?x3x5xf32, 15x5x1>,
                                %arg1: !migraphx.shaped<64x3x2xf32, 6x2x1>)
    -> !migraphx.shaped<?x64x4xf32, 256x4x1> {
  %0 = migraphx.convolution %arg0, %arg1 {dilation = [1], group = 1 : i64, padding = [0, 0], padding_mode = 0 : i64, stride = [1]} : <?x3x5xf32, 15x5x1>, <64x3x2xf32, 6x2x1> -> <?x64x4xf32, 256x4x1>
  return %0 : !migraphx.shaped<?x64x4xf32, 256x4x1>
}

// -----

// Backwards data convolution leaves through a rock custom op rather than a
// tosa.conv2d, so its result type has to carry the dynamic batch too.
// CHECK-LABEL: func.func @backwards_conv_dynamic_batch
// CHECK: tosa.custom {{.*}} operator_name = "conv_bwd_data"{{.*}} : (tensor<?x4x4x64xf32>, tensor<64x2x2x3xf32>, {{.*}}) -> tensor<?x5x5x3xf32>
func.func @backwards_conv_dynamic_batch(%arg0: !migraphx.shaped<?x64x4x4xf32, 1024x16x4x1>,
                                        %arg1: !migraphx.shaped<64x3x2x2xf32, 12x4x2x1>)
    -> !migraphx.shaped<?x3x5x5xf32, 75x25x5x1> {
  %0 = migraphx.backwards_data_convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <?x64x4x4xf32, 1024x16x4x1>, <64x3x2x2xf32, 12x4x2x1> -> <?x3x5x5xf32, 75x25x5x1>
  return %0 : !migraphx.shaped<?x3x5x5xf32, 75x25x5x1>
}

// -----

// CHECK-LABEL: func.func @softmax_dynamic_batch
// CHECK: tosa.reduce_max %{{.*}} {axis = 2 : i32} : (tensor<?x64x64xf32>) -> tensor<?x64x1xf32>
// CHECK: tosa.sub {{.*}} : (tensor<?x64x64xf32>, tensor<?x64x1xf32>) -> tensor<?x64x64xf32>
// CHECK: tosa.exp {{.*}} -> tensor<?x64x64xf32>
// CHECK: tosa.reduce_sum %{{.*}} {axis = 2 : i32} : (tensor<?x64x64xf32>) -> tensor<?x64x1xf32>
// CHECK: tosa.reciprocal {{.*}} -> tensor<?x64x1xf32>
// CHECK: tosa.mul {{.*}} : (tensor<?x64x64xf32>, tensor<?x64x1xf32>, {{.*}}) -> tensor<?x64x64xf32>
func.func @softmax_dynamic_batch(%arg0: !migraphx.shaped<?x64x64xf32, 4096x64x1>)
    -> !migraphx.shaped<?x64x64xf32, 4096x64x1> {
  %0 = migraphx.softmax %arg0 {axis = 2 : i64} : <?x64x64xf32, 4096x64x1> -> <?x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<?x64x64xf32, 4096x64x1>
}

// -----

// The rule is about stride order, not about position in the logical shape. Here
// the transpose moves the unknown shape to logical dimension 1, but it keeps
// the largest stride, so it is still the slowest-moving dimension in memory and
// the layout is representable.
// CHECK-LABEL: func.func @transpose_dynamic_not_outermost
// CHECK-SAME: (%{{.*}}: tensor<?xf32>) -> tensor<?xf32>
// CHECK: tosa.transpose %{{.*}} {perms = array<i32: 1, 0, 2>} : (tensor<?x64x72xf32>) -> tensor<64x?x72xf32>
func.func @transpose_dynamic_not_outermost(%arg0: !migraphx.shaped<?x64x72xf32, 4608x72x1>)
    -> !migraphx.shaped<64x?x72xf32, 72x4608x1> {
  %0 = migraphx.transpose %arg0 {permutation = [1, 0, 2]} : <?x64x72xf32, 4608x72x1> -> <64x?x72xf32, 72x4608x1>
  return %0 : !migraphx.shaped<64x?x72xf32, 72x4608x1>
}

// -----

// A GEMM with a bias add is the shape MIGraphX actually sends. The dynamic M
// has to line up between the matmul result and the elementwise operand.
// CHECK-LABEL: func.func @dot_add_dynamic_m
// CHECK: tosa.matmul {{.*}} -> tensor<1x?x64xf32>
// CHECK: tosa.add {{.*}} : (tensor<?x64xf32>, tensor<?x64xf32>) -> tensor<?x64xf32>
func.func @dot_add_dynamic_m(%arg0: !migraphx.shaped<?x72xf32, 72x1>,
                             %arg1: !migraphx.shaped<72x64xf32, 64x1>,
                             %arg2: !migraphx.shaped<?x64xf32, 64x1>)
    -> !migraphx.shaped<?x64xf32, 64x1> {
  %0 = migraphx.dot %arg0, %arg1 : <?x72xf32, 72x1>, <72x64xf32, 64x1> -> <?x64xf32, 64x1>
  %1 = migraphx.add %0, %arg2 : <?x64xf32, 64x1>, <?x64xf32, 64x1> -> <?x64xf32, 64x1>
  return %1 : !migraphx.shaped<?x64xf32, 64x1>
}