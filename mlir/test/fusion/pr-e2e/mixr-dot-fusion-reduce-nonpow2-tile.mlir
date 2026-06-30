// RUN: rocmlir-gen -fut mlir_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// Non-power-of-two M and N block tiles (mPerBlock/nPerBlock = 80) are requested
// via the dot's perf_config, so rock-decompose-nonpow2-tiles splits the GEMM
// into a 2x2 grid of power-of-two sub-tiles (64 + 16 along both M and N). The
// dot has input fusion (A and B are each the sum of two inputs) and a
// reduce_sum output fusion along N.
//
// This exercises the decompose pass replicating a reduction output-fusion DAG
// across the pow2 sub-tiles, including across the split N segments.

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%a0: !migraphx.shaped<1x160x64xf32, 10240x64x1>,
                      %a1: !migraphx.shaped<1x160x64xf32, 10240x64x1>,
                      %b0: !migraphx.shaped<1x64x160xf32, 10240x160x1>,
                      %b1: !migraphx.shaped<1x64x160xf32, 10240x160x1>,
                      %bias1: !migraphx.shaped<1x160x160xf32, 25600x160x1>)
      -> !migraphx.shaped<1x160x1xf32, 160x1x1> attributes {rock.kernel} {
    // Input fusion: A = a0 + a1, B = b0 + b1.
    %A = migraphx.add %a0, %a1 : <1x160x64xf32, 10240x64x1>, <1x160x64xf32, 10240x64x1> -> <1x160x64xf32, 10240x64x1>
    %B = migraphx.add %b0, %b1 : <1x64x160xf32, 10240x160x1>, <1x64x160xf32, 10240x160x1> -> <1x64x160xf32, 10240x160x1>
    // The dot drives the non-power-of-two (80x80) block tiling.
    %dot = migraphx.dot %A, %B {perf_config = "gemm:v1:80,80,16,1,1,4,16,1,1,1,1"} : <1x160x64xf32, 10240x64x1>, <1x64x160xf32, 10240x160x1> -> <1x160x160xf32, 25600x160x1>
    // Output fusion: (dot * bias1) then reduce_sum along N.
    %mul = migraphx.mul %dot, %bias1 : <1x160x160xf32, 25600x160x1>, <1x160x160xf32, 25600x160x1> -> <1x160x160xf32, 25600x160x1>
    %red = migraphx.reduce_sum %mul {axes = [2]} : <1x160x160xf32, 25600x160x1> -> <1x160x1xf32, 160x1x1>
    return %red : !migraphx.shaped<1x160x1xf32, 160x1x1>
  }
}
