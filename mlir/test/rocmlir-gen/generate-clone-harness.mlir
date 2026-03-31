// RUN: rocmlir-gen -fut mlir_reshape_convolution  --arch gfx90a:sramecc+:xnack- --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel |rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_reshape_convolution_wrapper --verifier clone - | FileCheck %s

// UNSUPPORTED: true
// TODO(rocTriton): Fails with error: "type mismatch: expected operand type 'tensor', but provided 'memref'"
// RUN: rocmlir-gen -fut mlir_reshape_convolution  --arch gfx90a:sramecc+:xnack- --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel |rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_reshape_convolution_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full| FileCheck %s  --check-prefixes=CHECK_FULL

func.func private @mlir_reshape_convolution(%arg0: !migraphx.shaped<1x1x16x2x16x2xf32, 256x256x16x0x1x0>, %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>) -> (!migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>) {
    %1 = migraphx.reshape %arg0 {dims = [1, 1, 32, 32]} : <1x1x16x2x16x2xf32, 256x256x16x0x1x0> -> <1x1x32x32xf32, 1024x1024x32x1>
    %2 = migraphx.convolution %1, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x32x32xf32, 1024x1024x32x1>, <1x1x3x3xf32, 9x9x3x1> -> <1x1x32x32xf32, 1024x1024x32x1>
    return %2 : !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>
}

// CHECK-LABEL: module
// CHECK: "func.func"() <{function_type = (tensor<256xf32>, tensor<9xf32>) -> tensor<1024xf32>, sym_name = "mlir_reshape_convolution", sym_visibility = "private"}
// CHECK: "func.func"() <{function_type = (tensor<256xf32>, tensor<9xf32>) -> tensor<1024xf32>, sym_name = "mlir_reshape_convolution_wrapper"}
// CHECK: "builtin.module"() <{sym_name = "__xmodule_"}>
// CHECK: "func.func"() <{function_type = (tensor<256xf32>, tensor<9xf32>, tensor<1024xf32>) -> tensor<1024xf32>, sym_name = "mlir_reshape_convolution", sym_visibility = "private"}
// CHECK-LABEL: "func.func"() <{function_type = () -> (), sym_name = "main"}>
// CHECK: "func.call"{{.*}} <{callee = @mlir_reshape_convolution_wrapper}> : (memref<256xf32>, memref<9xf32>) -> ()
// CHECK: "func.call"{{.*}} <{callee = @mlir_reshape_convolution_wrapper_cloned}> : (memref<256xf32>, memref<9xf32>) -> ()
// CHECK: "func.call"{{.*}} <{callee = @mlir_reshape_convolution_wrapper_verify1}> : (memref<9xf32>, memref<9xf32>) -> ()
// CHECK: "func.func"() <{function_type = (tensor<256xf32>, tensor<9xf32>) -> tensor<1024xf32>, sym_name = "mlir_reshape_convolution_wrapper_cloned"}> ({

// CHECK_FULL-LABEL: module
// CHECK_FULL: func.func private @mlir_reshape_convolution
