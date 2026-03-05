// RUN: rocmlir-gen -fut mlir_reshape_convolution  --arch gfx90a:sramecc+:xnack- --clone-harness %s | rocmlir-driver -pipeline=migraphx,highlevel |rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_reshape_convolution_wrapper --verifier clone - | FileCheck %s

func.func private @mlir_reshape_convolution(%arg0: !migraphx.shaped<1x1x16x2x16x2xf32, 256x256x16x0x1x0>, %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>) -> (!migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>) {
    %1 = migraphx.reshape %arg0 {dims = [1, 1, 32, 32]} : <1x1x16x2x16x2xf32, 256x256x16x0x1x0> -> <1x1x32x32xf32, 1024x1024x32x1>
    %2 = migraphx.convolution %1, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x32x32xf32, 1024x1024x32x1>, <1x1x3x3xf32, 9x9x3x1> -> <1x1x32x32xf32, 1024x1024x32x1>
    return %2 : !migraphx.shaped<1x1x32x32xf32, 1024x1024x32x1>
}

// After rocmlir-gen --clone-harness + rocmlir-driver -pipeline=migraphx,highlevel:
//   - mlir_reshape_convolution: GPU kernel (has rock.kernel attr)
//   - mlir_reshape_convolution_cpu: CPU clone (no kernel attr, lowered via linalg)
//   - mlir_reshape_convolution_wrapper: calls GPU kernel
// After rocmlir-gen -ph --verifier clone:
//   - main calls wrapper (GPU path) and wrapper_cloned (CPU path via _cpu)
//   - wrapper_cloned redirects calls to _cpu version

// CHECK-LABEL: module
// CHECK: func.func private @mlir_reshape_convolution({{.*}}) attributes {rock.kernel
// CHECK: func.func private @mlir_reshape_convolution_cpu({{.*}})
// CHECK-NOT: rock.kernel
// CHECK: func.func @mlir_reshape_convolution_wrapper({{.*}})
// CHECK-NOT: module @__xmodule_
// CHECK-LABEL: @main
// CHECK: call @mlir_reshape_convolution_wrapper({{.*}})
// CHECK: call @mlir_reshape_convolution_wrapper_cloned({{.*}})
// CHECK: func.func @mlir_reshape_convolution_wrapper_cloned({{.*}})
// CHECK: call @mlir_reshape_convolution_cpu({{.*}})
