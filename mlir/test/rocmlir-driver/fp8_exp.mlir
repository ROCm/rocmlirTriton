// RUN: sed s/##TOKEN_ARCH##/%arch/g %s \
// RUN:   | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel --arch %arch \
// RUN:   | rocmlir-driver -c --arch %arch -o /dev/null --mlir-print-ir-after=math-extend-to-supported-types 2>&1 \
// RUN:   | FileCheck %s

// Regression test for fp8 math fusion lowering. The kernel pipeline must
// promote fp8 math operands before fp8 tensors are legalized to integer storage
// types, otherwise LLVM math lowering can see i8 operands for math.exp.
//
// CHECK: IR Dump After MathExtendToSupportedTypes
// CHECK: arith.extf {{.*}} : tensor<[[SHAPE:[0-9]+x[0-9]+]]xf8E4M3FN> to tensor<[[SHAPE]]xf32>
// CHECK-NEXT: math.exp {{.*}} : tensor<[[SHAPE]]xf32>
// CHECK-NEXT: arith.truncf {{.*}} : tensor<[[SHAPE]]xf32> to tensor<[[SHAPE]]xf8E4M3FN>

module {
  func.func @mlir_convolution_exp(%arg0: !migraphx.shaped<1x16x4x4xf8E4M3FN, 256x16x4x1>, %arg1: !migraphx.shaped<2x16x3x3xf8E4M3FN, 144x9x3x1>) -> !migraphx.shaped<1x2x2x2xf8E4M3FN, 8x4x2x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x16x4x4xf8E4M3FN, 256x16x4x1>, <2x16x3x3xf8E4M3FN, 144x9x3x1> -> <1x2x2x2xf8E4M3FN, 8x4x2x1>
    %1 = migraphx.exp %0 : <1x2x2x2xf8E4M3FN, 8x4x2x1> -> <1x2x2x2xf8E4M3FN, 8x4x2x1>
    return %1 : !migraphx.shaped<1x2x2x2xf8E4M3FN, 8x4x2x1>
  }
}
