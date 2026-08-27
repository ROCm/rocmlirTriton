// The peak bounds are absolute lane counts and the carried floor is the
// addressable VGPR count, so gfx950 is where the difference between the two
// shows: a thread there addresses 512 registers, twice what it does on gfx1100,
// and each config below turns on one of the halves. Same 1x512x8x8 f16 conv as
// the gfx1100 test, so the only thing that varies is the target.

// 3667 lanes live before optimization, so the config never reaches it. Under a
// bound set as a fraction of the register file this passed, because the same
// fraction is twice as many lanes here as on gfx1100, and it went on to spend
// over 30s compiling. It is refused in under two seconds instead. This is the
// case that took the peak bound off the register file: what it costs to
// allocate follows the number of live values, and that does not change because
// the chip has more registers to spill into.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | not rocmlir-driver -c --arch=gfx950 \
// RUN:     --perf-config="gemm:mPerBlock=64,nPerBlock=64,kPerBlock=512,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PEAK

// The floor, by contrast, has to stay on the register file. This config carries
// 257 lanes against a 16-lane accumulator, sixteen times over the multiple, and
// compiles in 2.1s: the ratio is meaningless because 257 lanes is half of what
// a gfx950 thread addresses and nothing here has to spill. The same 257 lanes
// would be worth refusing on gfx1100, where they exceed the whole file. A floor
// pinned to either chip's number is wrong on the other one.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN:   | rocmlir-driver -c --arch=gfx950 \
// RUN:     --perf-config="gemm:mPerBlock=128,nPerBlock=16,kPerBlock=512,kpack=1,numCTAs=1,numWaves=2,matrixInstrNonkdim=32,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ACCEPT

// PEAK: error: 'mlir_convolution' needs 3667 32-bit register lanes live at once in the unoptimized IR
// PEAK-SAME: over the limit of 3328

// ACCEPT-NOT: register lanes
// ACCEPT: triton.hsaco

module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>, %arg1: !migraphx.shaped<512x512x3x3xf16, 4608x9x3x1>) -> !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1> attributes {rock.arch = "gfx950", rock.kernel = "mixr", rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x512x8x8xf16, 32768x64x8x1>, <512x512x3x3xf16, 4608x9x3x1> -> <1x512x8x8xf16, 32768x64x8x1>
    return %0 : !migraphx.shaped<1x512x8x8xf16, 32768x64x8x1>
  }
}
