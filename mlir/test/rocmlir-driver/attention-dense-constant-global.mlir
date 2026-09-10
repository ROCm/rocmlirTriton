// RUN: rocmlir-driver %s -kernel-pipeline=migraphx,highlevel,gpu,triton -arch=gfx950 | FileCheck %s

// Verify that a dense non-splat pre-softmax constant survives the complete
// MIGraphX-to-LLVM pipeline as compiler-owned GPU storage. The LLVM address
// is bridged through tt.int_to_ptr, which is supported by the vendored
// Triton pipeline without downstream changes.
// Attention's splat fake tensors are used only for indexing and must not create
// additional compiler-owned globals before or after the real dense bias.
// CHECK-NOT: llvm.mlir.global internal constant @__rock_constant_
// CHECK: llvm.mlir.global internal constant @[[$GLOBAL:__rock_constant_[0-9]+]]({{.*}}) {addr_space = 1 : i32, alignment = 16 : i64} : !llvm.array<16 x f32>
// CHECK-NOT: llvm.mlir.global internal constant @__rock_constant_
// CHECK-LABEL: llvm.func @constant_attention
// CHECK: %[[ADDRESS:.*]] = llvm.mlir.addressof @[[$GLOBAL]] : !llvm.ptr<1>
// CHECK: %[[INTEGER_ADDRESS:.*]] = llvm.ptrtoint %[[ADDRESS]] : !llvm.ptr<1> to i64
// CHECK: %[[POINTER:.*]] = llvm.inttoptr %[[INTEGER_ADDRESS]] : i64 to !llvm.ptr<1>
// CHECK: rocdl.make.buffer.rsrc %[[POINTER]]

module {
  func.func @constant_attention(
      %q: !migraphx.shaped<1x1x4x4xf32, 16x16x4x1>,
      %k: !migraphx.shaped<1x1x4x4xf32, 16x16x4x1>,
      %v: !migraphx.shaped<1x1x4x4xf32, 16x16x4x1>)
      -> !migraphx.shaped<1x1x4x4xf32, 16x16x4x1>
      attributes {rock.arch = "gfx950", rock.kernel = "mixr"} {
    %cst = migraphx.literal(
        dense<[[[[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0],
                  [13.0, 14.0, 15.0, 16.0]]]]>
        : tensor<1x1x4x4xf32>) : <1x1x4x4xf32, 16x16x4x1>
    %kt = migraphx.transpose %k {permutation = [0, 1, 3, 2]}
        : <1x1x4x4xf32, 16x16x4x1>
        -> <1x1x4x4xf32, 16x16x1x4>
    %scores = migraphx.dot %q, %kt
        : <1x1x4x4xf32, 16x16x4x1>,
          <1x1x4x4xf32, 16x16x1x4>
        -> <1x1x4x4xf32, 16x16x4x1>
    %scaled = migraphx.mul %scores, %cst
        : <1x1x4x4xf32, 16x16x4x1>,
          <1x1x4x4xf32, 16x16x4x1>
        -> <1x1x4x4xf32, 16x16x4x1>
    %probabilities = migraphx.softmax %scaled {axis = 3 : i64}
        : <1x1x4x4xf32, 16x16x4x1>
        -> <1x1x4x4xf32, 16x16x4x1>
    %result = migraphx.dot %probabilities, %v
        : <1x1x4x4xf32, 16x16x4x1>,
          <1x1x4x4xf32, 16x16x4x1>
        -> <1x1x4x4xf32, 16x16x4x1>
    return %result : !migraphx.shaped<1x1x4x4xf32, 16x16x4x1>
  }
}
