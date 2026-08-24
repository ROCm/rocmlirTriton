// Peak register pressure gates a configuration out of the tuning space before
// codegen runs. The arch is pinned instead of %arch-substituted because the
// limit scales with the addressable VGPR count, so the lane counts and verdicts
// below are gfx1100's (a limit of 4 * 256).
//
// This 1x512x8x8 f16 conv over a 16x64 tile on a single wave is the family that
// motivated the gate: sweeping kPerBlock walks peak pressure from 397 lanes to
// 2846, and compile time from 0.5s to over 60s along with it.

// kPerBlock=256 measures 2846 lanes. Let through it takes ~60s to compile,
// spills ~9800 times, and still runs 6x slower than the best config here.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | not rocmlir-driver -c --arch=gfx1100 --mlir-disable-threading \
// RUN:     --mlir-print-ir-after-failure --mlir-print-ir-module-scope \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=64,kPerBlock=256,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REJECT

// kPerBlock=32 measures 397 lanes, well inside the limit, and must compile.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | rocmlir-driver -c --arch=gfx1100 \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=64,kPerBlock=32,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ACCEPT

// kPerBlock=144 measures 892 lanes, the highest of any config worth keeping for
// this problem: it spills nothing and is the fastest of the family at 420us.
// It has to survive, which is what rules out a tighter limit.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | rocmlir-driver -c --arch=gfx1100 \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=64,kPerBlock=144,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ACCEPT

// The gate reports the measurement and the limit it broke, and marks the module
// so that tuning records the config as inapplicable rather than as a failure.
// REJECT: error: peak register pressure (2846 32-bit lanes)
// REJECT-SAME: exceeds the limit of 1024 for gfx1100
// REJECT: module attributes
// REJECT-SAME: rock.not_applicable

// ACCEPT-NOT: peak register pressure
// ACCEPT: triton.hsaco

module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>, %arg1: !migraphx.shaped<512x512x3x3xf16, 4608x9x3x1>) -> !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1> attributes {rock.arch = "gfx1100", rock.kernel = "mixr", rock.num_chiplets = 1 : i64, rock.num_cu = 48 : i64} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x512x8x8xf16, 32768x64x8x1>, <512x512x3x3xf16, 4608x9x3x1> -> <1x512x8x8xf16, 32768x64x8x1>
    return %0 : !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>
  }
}
