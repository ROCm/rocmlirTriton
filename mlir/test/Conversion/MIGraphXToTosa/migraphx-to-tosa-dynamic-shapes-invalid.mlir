// RUN: rocmlir-opt --migraphx-to-tosa -verify-diagnostics -split-input-file %s

// Only M may be dynamic
func.func @dot_dynamic_batch(%arg0: !migraphx.shaped<?x32x72xf32, 2304x72x1>,
                             %arg1: !migraphx.shaped<?x72x64xf32, 4608x64x1>)
    -> !migraphx.shaped<?x32x64xf32, 2048x64x1> {
  // expected-error @+2 {{only the M dimension of a dot may be dynamic}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
  %0 = migraphx.dot %arg0, %arg1 : <?x32x72xf32, 2304x72x1>, <?x72x64xf32, 4608x64x1> -> <?x32x64xf32, 2048x64x1>
  return %0 : !migraphx.shaped<?x32x64xf32, 2048x64x1>
}

// -----

// Attention is a batched dot, a softmax and a second batched dot. The softmax
// handles a dynamic batch, but the dots flatten their batch extents into one,
// which needs a number.
func.func @attention_dynamic_batch(%q: !migraphx.shaped<?x64x64xf32, 4096x64x1>,
                                   %k: !migraphx.shaped<?x64x64xf32, 4096x64x1>,
                                   %v: !migraphx.shaped<?x64x64xf32, 4096x64x1>)
    -> !migraphx.shaped<?x64x64xf32, 4096x64x1> {
  // expected-error @+2 {{only the M dimension of a dot may be dynamic}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.dot' that was explicitly marked illegal}}
  %0 = migraphx.dot %q, %k : <?x64x64xf32, 4096x64x1>, <?x64x64xf32, 4096x64x1> -> <?x64x64xf32, 4096x64x1>
  %1 = migraphx.softmax %0 {axis = 2 : i64} : <?x64x64xf32, 4096x64x1> -> <?x64x64xf32, 4096x64x1>
  %2 = migraphx.dot %1, %v : <?x64x64xf32, 4096x64x1>, <?x64x64xf32, 4096x64x1> -> <?x64x64xf32, 4096x64x1>
  return %2 : !migraphx.shaped<?x64x64xf32, 4096x64x1>
}

// -----

// The channel dimension is not the slowest-moving one, so reject it.
func.func @conv_dynamic_channel(%arg0: !migraphx.shaped<2x?x5x5xf32, 75x25x5x1>,
                                %arg1: !migraphx.shaped<64x3x2x2xf32, 12x4x2x1>)
    -> !migraphx.shaped<2x64x4x4xf32, 1024x16x4x1> {
  // expected-error @+2 {{only the batch dimension of a convolution may be dynamic}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.convolution' that was explicitly marked illegal}}
  %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <2x?x5x5xf32, 75x25x5x1>, <64x3x2x2xf32, 12x4x2x1> -> <2x64x4x4xf32, 1024x16x4x1>
  return %0 : !migraphx.shaped<2x64x4x4xf32, 1024x16x4x1>
}

// -----

// A dynamic length on anything but the slowest-moving dimension has no memory
// layout, so the kernel argument cannot be flattened.
// expected-error @+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @dynamic_inner_dim(%arg0: !migraphx.shaped<32x?xf32, 72x1>)
    -> !migraphx.shaped<32x?xf32, 72x1> {
  %0 = migraphx.add %arg0, %arg0 : <32x?xf32, 72x1>, <32x?xf32, 72x1> -> <32x?xf32, 72x1>
  return %0 : !migraphx.shaped<32x?xf32, 72x1>
}

// -----

// Two dynamic lengths would need two inferable dimensions in one tosa.reshape.
// expected-error @+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @two_dynamic_dims(%arg0: !migraphx.shaped<?x?x72xf32, 2304x72x1>)
    -> !migraphx.shaped<?x?x72xf32, 2304x72x1> {
  %0 = migraphx.add %arg0, %arg0 : <?x?x72xf32, 2304x72x1>, <?x?x72xf32, 2304x72x1> -> <?x?x72xf32, 2304x72x1>
  return %0 : !migraphx.shaped<?x?x72xf32, 2304x72x1>
}

// -----

// A dynamic stride is rejected outright: the memory layout is derived by
// arithmetic on the strides, and nothing supplies one at runtime.
// expected-error @+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @dynamic_stride(%arg0: !migraphx.shaped<32x64xf32, ?x1>)
    -> !migraphx.shaped<32x64xf32, ?x1> {
  %0 = migraphx.add %arg0, %arg0 : <32x64xf32, ?x1>, <32x64xf32, ?x1> -> <32x64xf32, ?x1>
  return %0 : !migraphx.shaped<32x64xf32, ?x1>
}

// -----

// A dynamic length on a broadcast axis collapses to 1 in memory, so the layout
// is fine, but rebuilding the logical shape has to broadcast back out to the
// unknown length.
func.func @dynamic_broadcast_axis(%arg0: !migraphx.shaped<?x64xf32, 0x1>)
    -> !migraphx.shaped<?x64xf32, 0x1> {
  // expected-error @+2 {{cannot broadcast out to a dynamic shape}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.mlir.as.logical.shape' that was explicitly marked illegal}}
  %0 = migraphx.add %arg0, %arg0 : <?x64xf32, 0x1>, <?x64xf32, 0x1> -> <?x64xf32, 0x1>
  return %0 : !migraphx.shaped<?x64xf32, 0x1>
}

// -----

// Lowerings that build a dense constant sized to the result still need every
// extent, since a dense attribute cannot have a dynamic type.
func.func @relu_dynamic(%arg0: !migraphx.shaped<?x64xf32, 64x1>)
    -> !migraphx.shaped<?x64xf32, 64x1> {
  // expected-error @+2 {{dynamic shapes are not supported}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.relu' that was explicitly marked illegal}}
  %0 = migraphx.relu %arg0 : <?x64xf32, 64x1> -> <?x64xf32, 64x1>
  return %0 : !migraphx.shaped<?x64xf32, 64x1>
}
