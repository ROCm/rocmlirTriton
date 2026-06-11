// RUN: rocmlir-gen -fut mlir_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

// A non-power-of-two block tile (mPerBlock = 80) is requested via the dot's
// perf_config, so rock-decompose-nonpow2-tiles splits the GEMM along M.
//
// The twist vs. mixr-dot-fusion-power-of-two-tile.mlir: a single "shared" tensor
// is used in BOTH the input-fusion DAG (added to A) and the output-fusion DAG
// (added to the dot result). Because K == N == 64, A (MxK) and the output (MxN)
// have the same shape, so the shared tensor needs no broadcast. This is the
// exact shape the reviewer worried about for the shared-memoization dominance
// hole in TileSplitter: one value reachable both as a GEMM input and through the
// output fusion. We let the real lowering decide whether that ever becomes a
// single loop-invariant rock value (and thus a real hazard) or two distinct
// loads.

module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%a0: !migraphx.shaped<1x160x64xf32, 10240x64x1>,
                      %b0: !migraphx.shaped<1x64x64xf32, 4096x64x1>,
                      %bias1: !migraphx.shaped<1x160x64xf32, 10240x64x1>,
                      %shared: !migraphx.shaped<1x160x64xf32, 10240x64x1>)
      -> !migraphx.shaped<1x160x64xf32, 10240x64x1> attributes {rock.kernel} {
    // Input fusion: A = a0 + shared.
    %A = migraphx.add %a0, %shared : <1x160x64xf32, 10240x64x1>, <1x160x64xf32, 10240x64x1> -> <1x160x64xf32, 10240x64x1>
    // The dot drives the non-power-of-two (mPerBlock = 80) block tiling.
    %dot = migraphx.dot %A, %b0 {perf_config = "gemm:v1:80,64,16,1,1,4,16,1,1,1,1"} : <1x160x64xf32, 10240x64x1>, <1x64x64xf32, 4096x64x1> -> <1x160x64xf32, 10240x64x1>
    // Output fusion: (dot * bias1) + shared.
    %mul = migraphx.mul %dot, %bias1 : <1x160x64xf32, 10240x64x1>, <1x160x64xf32, 10240x64x1> -> <1x160x64xf32, 10240x64x1>
    %add = migraphx.add %mul, %shared : <1x160x64xf32, 10240x64x1>, <1x160x64xf32, 10240x64x1> -> <1x160x64xf32, 10240x64x1>
    return %add : !migraphx.shaped<1x160x64xf32, 10240x64x1>
  }
}
