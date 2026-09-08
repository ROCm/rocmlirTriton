// Pins the extra ADCE run that TritonToHsaco adds after LLVM's default
// pipeline.
//
// This GEMM has M = 77 against mPerBlock = 128, so some accumulator tiles lie
// entirely outside the output and Triton emits no store for them. What is left
// is a closed K-loop cycle of phi, insertelement, wmma and extractelement whose
// every instruction has a use, so only ADCE can reclaim it. The run has to come
// after the default pipeline, since the dead lanes only become separable once
// BreakStructPhiNodesPass splits the aggregate accumulator phi.
//
// None of this involves removing a store, so it is independent of
// rock-fold-oob-buffer-ops; the count below is the same with that fold disabled.
//
// The arch and tuning parameters are fixed rather than taken from the test
// environment because the dead tiles only exist for a tile size that overhangs
// M: a configuration dividing 77 evenly has nothing to eliminate.

// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c --arch=gfx1100 --perf-config=gemm:mPerBlock=128,nPerBlock=128,kPerBlock=32,kpack=1,numCTAs=1,numWaves=8,matrixInstrNonkdim=0,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0 -o /dev/null 2>&1 | FileCheck %s

// Only the loop body is checked. The accumulators that survive are flushed
// after the loop, and that trailing wmma count is the same either way; the dead
// tiles show up purely as extra work per iteration (16 rather than 12).
// CHECK: ; =>This Inner Loop Header: Depth=1
// CHECK-COUNT-12: v_wmma_f32_16x16x16_f16
// CHECK-NOT: v_wmma_f32_16x16x16_f16
// CHECK: s_cbranch_scc1

module {
  func.func @mlir_unpack_int4_reshape_dequantizelinear_transpose_reshape_unsqueeze_transpose_dot_add(%arg0: !migraphx.shaped<4096x2048xui8, 2048x1>, %arg1: !migraphx.shaped<4096x32x1xf16, 32x1x1>, %arg2: !migraphx.shaped<1x64x77x64xf16, 315392x4928x64x1>, %arg3: !migraphx.shaped<1x77x4096xf16, 315392x4096x1>) -> !migraphx.shaped<1x77x4096xf16, 315392x4096x1> attributes {rock.arch = "gfx1100", rock.kernel} {
    %0 = migraphx.literal(dense<8> : tensor<1xui8>) : <1xui8, 0>
    %1 = migraphx.unpack %arg0 {axis = 1 : i64} : <4096x2048xui8, 2048x1> -> <4096x4096xui8, 4096x1>
    %2 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [4096, 32, 128]} : <4096x32x1xf16, 32x1x1> -> <4096x32x128xf16, 32x1x0>
    %3 = migraphx.reshape %2 {dims = [4096, 4096]} : <4096x32x128xf16, 32x1x0> -> <4096x4096xf16, 4096x1>
    %4 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [4096, 4096]} : <1xui8, 0> -> <4096x4096xui8, 0x0>
    %5 = migraphx.dequantizelinear %1, %3, %4 : <4096x4096xui8, 4096x1>, <4096x4096xf16, 4096x1>, !migraphx.shaped<4096x4096xui8, 0x0> -> <4096x4096xf16, 4096x1>
    %6 = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <1x64x77x64xf16, 315392x4928x64x1> -> <1x77x64x64xf16, 315392x64x4928x1>
    %7 = migraphx.reshape %6 {dims = [1, 77, 4096]} : <1x77x64x64xf16, 315392x64x4928x1> -> <1x77x4096xf16, 315392x4096x1>
    %8 = migraphx.reshape %5 {dims = [1, 4096, 4096]} : <4096x4096xf16, 4096x1> -> <1x4096x4096xf16, 16777216x4096x1>
    %9 = migraphx.transpose %8 {permutation = [0, 2, 1]} : <1x4096x4096xf16, 16777216x4096x1> -> <1x4096x4096xf16, 16777216x1x4096>
    %10 = migraphx.dot %7, %9 : <1x77x4096xf16, 315392x4096x1>, <1x4096x4096xf16, 16777216x1x4096> -> <1x77x4096xf16, 315392x4096x1>
    %11 = migraphx.add %arg3, %10 : <1x77x4096xf16, 315392x4096x1>, <1x77x4096xf16, 315392x4096x1> -> <1x77x4096xf16, 315392x4096x1>
    return %11 : !migraphx.shaped<1x77x4096xf16, 315392x4096x1>
  }
}
