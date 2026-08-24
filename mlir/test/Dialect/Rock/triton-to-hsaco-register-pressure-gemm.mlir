// The carried bound is a multiple of the kernel's accumulator, floored at the
// addressable VGPR count, and both halves are needed. This 2048x2048 f16 gemm
// on gfx1100, where a thread addresses 256 registers, holds the two ends of the
// tuning space that pin them down. Neither config here is worth rejecting: the
// whole space compiles, the slowest config in it takes 6.7s.

// The largest tile carries 513 lanes, twice everything a thread can address,
// and compiles in 3.6s. It is ordinary: a 256x256 f32 tile over a 128-thread
// block is a 512-lane accumulator, so the tile carries the one thing it cannot
// do without and one lane more. A bound set at a fraction of the register file
// rejects this, and rejected fifteen configs of this space, because the
// accumulator of a large tile does not shrink to fit a small register file.
// Against the accumulator it sits at 1x on every chip.
// RUN: rocmlir-driver -c \
// RUN:     --perf-config="gemm:mPerBlock=256,nPerBlock=256,kPerBlock=32,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" %s 2>&1 \
// RUN:   | FileCheck %s

// A 16x32 tile over a 256-thread block is a one-lane accumulator, so the 22
// lanes this config carries for its addresses and bounds run to twenty-two
// accumulators while meaning nothing at all: it compiles in 0.14s. This is what
// the floor is for. Without it the multiple rejects the smallest tiles in the
// space, which is the same mistake as rejecting the largest, from the other
// end.
// RUN: rocmlir-driver -c \
// RUN:     --perf-config="gemm:mPerBlock=16,nPerBlock=32,kPerBlock=32,kpack=1,numCTAs=1,numWaves=8,matrixInstrNonkdim=0,splitKFactor=1,numStages=3,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1" %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK-NOT: register lanes
// CHECK: triton.hsaco

#map = affine_map<(d0, d1, d2) -> (d1 * 2048 + d2)>
#map1 = affine_map<(d0) -> (0, d0 floordiv 2048, d0 mod 2048)>
#transform_map = #rock.transform_map<#map by [<Unmerge{2048, 2048} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 2048, 2048] -> [4194304]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{2048, 2048} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 2048, 2048] -> [4194304]>
#transform_map2 = #rock.transform_map<#map1 by [<Merge{2048, 2048} ["raw"] at [0] -> ["m", "n"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [4194304] -> [1, 2048, 2048]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<4194304xf16>, %arg1: tensor<4194304xf16>, %arg2: tensor<4194304xf16>) -> tensor<4194304xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 48 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<4194304xf16> to tensor<1x2048x2048xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<4194304xf16> to tensor<1x2048x2048xf16>
    %2 = rock.gemm %0 * %1 : tensor<1x2048x2048xf16> * tensor<1x2048x2048xf16> -> tensor<1x2048x2048xf16>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x2048x2048xf16> to tensor<4194304xf16>
    %4 = rock.store %3 to %arg2 by set : tensor<4194304xf16> -> tensor<4194304xf16> to tensor<4194304xf16>
    return %4 : tensor<4194304xf16>
  }
}
