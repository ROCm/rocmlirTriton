// Peak register pressure gates a configuration out of the tuning space on the
// unoptimized LLVM IR, and carried pressure gates it on the optimized IR. The
// peak bounds are absolute lane counts, 3328 unoptimized and 3072 optimized,
// the same on every target and every element type. The arch is pinned rather than %arch-substituted
// because the carried bound has a floor at the addressable VGPR count, so the
// verdicts below are gfx1100's: 10x the accumulator carried, and no carried
// rejection under 256 lanes.
//
// This 1x512x8x8 f16 conv over a 16x64 tile on a single wave is the family that
// motivated the gate. Sweeping kPerBlock walks it from configs that compile in
// half a second to ones that take a minute. Its accumulator is 32 lanes: a
// 16x64 tile of f32 over a 32-thread block.

// kPerBlock=256 holds 4149 lanes live, so it never reaches the optimizer. It
// took 60s to compile and is refused in 1.7s, most of the saving being the trip
// to the optimized IR, which is why this screen runs first.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | not rocmlir-driver -c --arch=gfx1100 --mlir-disable-threading \
// RUN:     --mlir-print-ir-after-failure --mlir-print-ir-module-scope \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=64,kPerBlock=256,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PEAK

// kPerBlock=128 is why peak width alone is not enough. It holds 1237 lanes live
// after optimization, fewer than the 1300 the kPerBlock=144 config below holds
// before it, yet it takes 15s to compile where that one takes 0.9s. What
// separates them is what they carry: 417 lanes against 33, thirteen
// accumulators against one.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | not rocmlir-driver -c --arch=gfx1100 \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=64,kPerBlock=128,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CARRIED

// kPerBlock=144 carries 33 lanes, one accumulator, and compiles in 0.9s. It
// must survive: this is the config that a peak-only bound tight enough to catch
// kPerBlock=128 would take with it.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | rocmlir-driver -c --arch=gfx1100 \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=64,kPerBlock=144,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ACCEPT

// kPerBlock=64 carries 225 lanes, seven accumulators, and still compiles in
// 1.2s. It is the closest a quick config in this family comes to the bound, so
// it is what keeps the multiple off seven.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | rocmlir-driver -c --arch=gfx1100 \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=64,kPerBlock=64,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ACCEPT

// The gate reports what it measured and where, and marks the module so that
// tuning records the config as inapplicable rather than as a compiler bug.
// PEAK: error: 'mlir_convolution' needs 4149 32-bit register lanes live at once in the unoptimized IR
// PEAK-SAME: over the limit of 3328
// PEAK: module attributes
// PEAK-SAME: rock.not_applicable

// CARRIED: error: 'mlir_convolution' carries 417 32-bit register lanes across a loop back edge in the optimized IR
// CARRIED-SAME: 13 times the 32 its accumulator needs
// CARRIED-SAME: over the limit of 10x on gfx1100

// ACCEPT-NOT: register lanes
// ACCEPT: triton.hsaco

module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>, %arg1: !migraphx.shaped<512x512x3x3xf16, 4608x9x3x1>) -> !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1> attributes {rock.arch = "gfx1100", rock.kernel = "mixr", rock.num_chiplets = 1 : i64, rock.num_cu = 48 : i64} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x512x8x8xf16, 32768x64x8x1>, <512x512x3x3xf16, 4608x9x3x1> -> <1x512x8x8xf16, 32768x64x8x1>
    return %0 : !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>
  }
}
