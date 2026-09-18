// E2E test that a fused kernel still fits the LDS budget its perf config was
// tuned against, when the epilogue reaches the accumulator through broadcasts.
//
// MIGraphX keys its problem cache on the convolution descriptor alone, so this
// module and the plain conv-plus-bias it was fused from share one cache entry:
// the perf config and the `rock.max_lds` below are what was stored while
// compiling the unfused variant. The unfused kernel allocates exactly 13824
// bytes under this config, which is the two operand tiles of the main loop and
// nothing else (64x18 + 128x18 f32 = 4608 + 9216).
//
// Fusing has to stay LDS-neutral or the replayed config no longer fits. It
// does not on its own here: two of the epilogue's side operands are per-channel
// biases that arrive as `tt.broadcast` of an Mx1 load in the coalesced layout,
// while the accumulator is in the FMA dot's blocked layout. That leaves a
// 64x128 f32 round trip between the two, which peaks at 16384 bytes and
// overruns the budget by 2560.
//
// OptimizeEpilogue removes it by issuing the whole epilogue in the accumulator
// layout, which it can only do if it relayouts through the broadcasts instead
// of halting at them. So this pins the budget being met, and a regression in
// that handling fails the run with the `rock.max_lds` diagnostic rather than
// silently costing LDS.

// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1100 %s \
// RUN:   | rocmlir-driver -kernel-pipeline=full -arch gfx1100 \
// RUN:   | FileCheck %s

// CHECK: ttg.shared = 13824 : i32

module {
  func.func @mlir_slice_convolution_broadcast_broadcast_slice_add_add_mul_add_mul_max_mul_add(%arg0: !migraphx.shaped<1x512x64x64xf32, 2097152x4096x64x1>, %arg1: !migraphx.shaped<256x256x3x3xf32, 2304x9x3x1>, %arg2: !migraphx.shaped<256xf32, 1>, %arg3: !migraphx.shaped<1x256x64x64xf32, 1048576x4096x64x1>, %arg4: !migraphx.shaped<256xf32, 1>, %arg5: !migraphx.shaped<1x512x64x64xf32, 2097152x4096x64x1>, %arg6: !migraphx.shaped<1x256x64x64xf32, 1048576x4096x64x1>) -> !migraphx.shaped<1x256x64x64xf32, 1048576x4096x64x1> attributes {rock.arch = "gfx1100", rock.kernel = "mixr", rock.max_lds = 13824 : i64, rock.num_chiplets = 1 : i64, rock.num_cu = 48 : i64} {
    %0 = migraphx.literal(dense<2.000000e-01> : tensor<1xf32>) : <1xf32, 1>
    %1 = migraphx.literal(dense<1.41421354> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.slice %arg0 {axes = [1], ends = [512], starts = [256]} : <1x512x64x64xf32, 2097152x4096x64x1> -> <1x256x64x64xf32, 2097152x4096x64x1>
    %3 = migraphx.convolution %2, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, perf_config = "gemm:mPerBlock=64,nPerBlock=128,kPerBlock=18,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=0,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1", stride = [1, 1]} : <1x256x64x64xf32, 2097152x4096x64x1>, <256x256x3x3xf32, 2304x9x3x1> -> <1x256x64x64xf32, 1048576x4096x64x1>
    %4 = migraphx.broadcast %arg2 {axis = 1 : i64, out_dyn_dims = [], out_lens = [1, 256, 64, 64]} : <256xf32, 1> -> <1x256x64x64xf32, 0x1x0x0>
    %5 = migraphx.broadcast %arg4 {axis = 1 : i64, out_dyn_dims = [], out_lens = [1, 256, 64, 64]} : <256xf32, 1> -> <1x256x64x64xf32, 0x1x0x0>
    %6 = migraphx.slice %arg5 {axes = [1], ends = [512], starts = [256]} : <1x512x64x64xf32, 2097152x4096x64x1> -> <1x256x64x64xf32, 2097152x4096x64x1>
    %7 = migraphx.add %3, %4 : <1x256x64x64xf32, 1048576x4096x64x1>, <1x256x64x64xf32, 0x1x0x0> -> <1x256x64x64xf32, 1048576x4096x64x1>
    %8 = migraphx.add %arg3, %5 : <1x256x64x64xf32, 1048576x4096x64x1>, <1x256x64x64xf32, 0x1x0x0> -> <1x256x64x64xf32, 1048576x4096x64x1>
    %9 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 256, 64, 64]} : <1xf32, 0> -> <1x256x64x64xf32, 0x0x0x0>
    %10 = migraphx.mul %6, %9 : <1x256x64x64xf32, 2097152x4096x64x1>, <1x256x64x64xf32, 0x0x0x0> -> <1x256x64x64xf32, 2097152x4096x64x1>
    %11 = migraphx.add %10, %arg6 : <1x256x64x64xf32, 2097152x4096x64x1>, <1x256x64x64xf32, 1048576x4096x64x1> -> <1x256x64x64xf32, 1048576x4096x64x1>
    %12 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 256, 64, 64]} : <1xf32, 1> -> <1x256x64x64xf32, 0x0x0x0>
    %13 = migraphx.mul %11, %12 : <1x256x64x64xf32, 1048576x4096x64x1>, <1x256x64x64xf32, 0x0x0x0> -> <1x256x64x64xf32, 1048576x4096x64x1>
    %14 = migraphx.max %11, %13 : <1x256x64x64xf32, 1048576x4096x64x1>, <1x256x64x64xf32, 1048576x4096x64x1> -> <1x256x64x64xf32, 1048576x4096x64x1>
    %15 = migraphx.mul %14, %8 : <1x256x64x64xf32, 1048576x4096x64x1>, <1x256x64x64xf32, 1048576x4096x64x1> -> <1x256x64x64xf32, 1048576x4096x64x1>
    %16 = migraphx.add %15, %7 : <1x256x64x64xf32, 1048576x4096x64x1>, <1x256x64x64xf32, 1048576x4096x64x1> -> <1x256x64x64xf32, 1048576x4096x64x1>
    return %16 : !migraphx.shaped<1x256x64x64xf32, 1048576x4096x64x1>
  }
}
