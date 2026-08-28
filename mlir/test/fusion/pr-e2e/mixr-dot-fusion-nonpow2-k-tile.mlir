// RUN: rocmlir-gen -fut mlir_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// A non-power-of-two K block tile (kPerBlock = 48) is requested via the dot's
// perf_config, so the K loop is decomposed into power-of-two segments (32 + 16).
// K = 96 is a multiple of 48, so the peeling is exact and no K padding is
// needed. The output reduce_sum also exercises the blockwise reduction of the
// resulting power-of-two output tile.

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%a0: !migraphx.shaped<1x128x96xf32, 12288x96x1>,
                      %a1: !migraphx.shaped<1x128x96xf32, 12288x96x1>,
                      %b0: !migraphx.shaped<1x96x128xf32, 12288x128x1>,
                      %b1: !migraphx.shaped<1x96x128xf32, 12288x128x1>,
                      %bias1: !migraphx.shaped<1x128x128xf32, 16384x128x1>,
                      %bias2: !migraphx.shaped<1x128x128xf32, 16384x128x1>)
      -> !migraphx.shaped<1x128x1xf32, 128x1x1> attributes {rock.kernel} {
    // Input fusion: A = a0 + a1, B = b0 + b1.
    %A = migraphx.add %a0, %a1 : <1x128x96xf32, 12288x96x1>, <1x128x96xf32, 12288x96x1> -> <1x128x96xf32, 12288x96x1>
    %B = migraphx.add %b0, %b1 : <1x96x128xf32, 12288x128x1>, <1x96x128xf32, 12288x128x1> -> <1x96x128xf32, 12288x128x1>
    // The dot drives the non-power-of-two (kPerBlock = 48) block tiling.
    %dot = migraphx.dot %A, %B {perf_config = "gemm:v1:64,64,48,1,1,4,16,1,1,1,1"} : <1x128x96xf32, 12288x96x1>, <1x96x128xf32, 12288x128x1> -> <1x128x128xf32, 16384x128x1>
    // Output fusion: reduce_sum((dot * bias1) + bias2) along N.
    %mul = migraphx.mul %dot, %bias1 : <1x128x128xf32, 16384x128x1>, <1x128x128xf32, 16384x128x1> -> <1x128x128xf32, 16384x128x1>
    %add = migraphx.add %mul, %bias2 : <1x128x128xf32, 16384x128x1>, <1x128x128xf32, 16384x128x1> -> <1x128x128xf32, 16384x128x1>
    %red = migraphx.reduce_sum %add {axes = [2]} : <1x128x128xf32, 16384x128x1> -> <1x128x1xf32, 128x1x1>
    return %red : !migraphx.shaped<1x128x1xf32, 128x1x1>
  }
}
