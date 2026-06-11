// RUN: rocmlir-gen -fut mlir_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// A non-power-of-two block tile (mPerBlock/nPerBlock = 80) is requested via the
// dot's perf_config, so rock-decompose-nonpow2-tiles splits the GEMM into a 2x2
// grid of power-of-two sub-tiles. The dot also has input fusion (A and B are
// each the sum of two inputs) and output fusion (the result is multiplied then
// offset), exercising recursive splitting through both fusion DAGs.

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%a0: !migraphx.shaped<1x160x64xf32, 10240x64x1>,
                      %a1: !migraphx.shaped<1x160x64xf32, 10240x64x1>,
                      %b0: !migraphx.shaped<1x64x160xf32, 10240x160x1>,
                      %b1: !migraphx.shaped<1x64x160xf32, 10240x160x1>,
                      %bias0: !migraphx.shaped<1x160x160xf32, 25600x160x1>,
                      %bias1: !migraphx.shaped<1x160x160xf32, 25600x160x1>)
      -> !migraphx.shaped<1x160x160xf32, 25600x160x1> attributes {rock.kernel} {
    // Input fusion: A = a0 + a1, B = b0 + b1.
    %A = migraphx.add %a0, %a1 : <1x160x64xf32, 10240x64x1>, <1x160x64xf32, 10240x64x1> -> <1x160x64xf32, 10240x64x1>
    %B = migraphx.add %b0, %b1 : <1x64x160xf32, 10240x160x1>, <1x64x160xf32, 10240x160x1> -> <1x64x160xf32, 10240x160x1>
    // The dot drives the non-power-of-two (80x80) block tiling.
    %dot = migraphx.dot %A, %B {perf_config = "gemm:v1:80,80,16,1,1,4,16,1,1,1,1"} : <1x160x64xf32, 10240x64x1>, <1x64x160xf32, 10240x160x1> -> <1x160x160xf32, 25600x160x1>
    // Output fusion: (dot * bias1) + bias0.
    %mul = migraphx.mul %dot, %bias1 : <1x160x160xf32, 25600x160x1>, <1x160x160xf32, 25600x160x1> -> <1x160x160xf32, 25600x160x1>
    %add = migraphx.add %mul, %bias0 : <1x160x160xf32, 25600x160x1>, <1x160x160xf32, 25600x160x1> -> <1x160x160xf32, 25600x160x1>
    return %add : !migraphx.shaped<1x160x160xf32, 25600x160x1>
  }
}
