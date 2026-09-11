// E2E test to exercise the OptimizeEpilogue pass, and show that bypassing
// the epilogue relayout for an FMA dot removes an LDS round trip.
//
// The pass rewrites `store(convert_layout(val))` into `store(convert_layout(ptr))`,
// so the accumulator is written straight from the registers the dot left it in.
// The relayout it replaces is not free: it needs its own LDS buffer plus the
// write/barrier/read sequence to move the tile through it. Neither the buffer
// nor the barriers survive the rewrite, and both are visible in the lowered
// module, so they are what this test pins.
//
// An f32 dot on gfx1101 has no matrix core to target, so it lowers to FMA with a
// blocked result layout -- the case the pass used to bail out on. The two runs
// differ only in the `useOptimizeEpilogue` knob, and since the pass declined
// every FMA dot before this change, the `useOptimizeEpilogue=0` run doubles as
// the pre-change baseline. That also confirms the knob reaches the FMA path.
//
// perf_config is pinned so the LDS and barrier counts do not move with the
// tuning heuristics. The 64x64 tile matters: the dot result layout
// (sizePerThread = [4, 4]) and the coalesced store layout (sizePerThread =
// [1, 4]) only diverge once the tile is wide enough to need a relayout at all.

// Default (-1) lets the pass bypass the relayout.
// RUN: rocmlir-gen --arch gfx1101 --operation gemm -t f32 -g 1 -m 1024 -n 1024 -k 1024 \
// RUN:   --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=16,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=0,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton -arch gfx1101 \
// RUN:   | FileCheck %s --check-prefix=BYPASS

// useOptimizeEpilogue=0 drops the pass, which is how the epilogue looked before
// blocked sources were accepted.
// RUN: rocmlir-gen --arch gfx1101 --operation gemm -t f32 -g 1 -m 1024 -n 1024 -k 1024 \
// RUN:   --perf_config=gemm:mPerBlock=64,nPerBlock=64,kPerBlock=16,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=0,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=0,useBf16x3ForF32=-1 \
// RUN:   | rocmlir-driver --kernel-pipeline=gpu,triton -arch gfx1101 \
// RUN:   | FileCheck %s --check-prefix=LDS

// Only the two operand tiles need LDS once the epilogue stores from registers.
// Keeping the relayout asks for a second buffer of the same size.
// BYPASS: ttg.shared = 4096 : i32
// LDS: ttg.shared = 8192 : i32

// The remaining barriers are the operand double-buffering inside the reduction
// loop. The relayout adds one more write/read pair after it, and each half of
// that pair has to be fenced, so the count goes from four to eight.
// BYPASS-COUNT-4: rocdl.s.barrier
// BYPASS-NOT: rocdl.s.barrier

// LDS-COUNT-8: rocdl.s.barrier
// LDS-NOT: rocdl.s.barrier
