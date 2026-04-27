// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-driver -c --arch "gfx908:sramecc+:xnack-" -mlir-print-ir-after=rock-lower-stores -o /dev/null 2>&1 | FileCheck %s
module {
  // CHECK-COUNT-2: rock.blockwise_load
  // CHECK: rock.blockwise_load
  // CHECK: arith.addf
  // CHECK: arith.maximumf
  // CHECK: rock.blockwise_store
  // CHECK-NOT: memref.copy

  func.func @main(%arg0: !migraphx.shaped<1x1x1x1xf32, 1x1x1x1>, %arg1: !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>, %arg2: !migraphx.shaped<1x3x5x5xf32, 75x25x5x1>) -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1> attributes {rock.arch = "gfx908:sramecc+:xnack-", rock.kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 1, 28, 28]} : <1x1x1x1xf32, 1x1x1x1> -> <1x1x28x28xf32, 1x1x0x0>
    %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x3x32x32xf32, 3072x1024x32x1>, <1x3x5x5xf32, 75x25x5x1> -> <1x1x28x28xf32, 784x784x28x1>
    %2 = migraphx.add %1, %0 : <1x1x28x28xf32, 784x784x28x1>, <1x1x28x28xf32, 1x1x0x0> -> <1x1x28x28xf32, 784x784x28x1>
    %3 = migraphx.relu %2 : <1x1x28x28xf32, 784x784x28x1> -> <1x1x28x28xf32, 784x784x28x1>
    return %3 : !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
  }
}
