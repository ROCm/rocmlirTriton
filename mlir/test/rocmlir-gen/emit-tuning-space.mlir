//===----------------------------------------------------------------------===//
// MFMA tuning space
//===----------------------------------------------------------------------===//

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=104 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=4,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=32,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-SPLITKFACTOR
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=3,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=4,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-NUMSTAGES
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=3,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1

// Attention emits the gemm+gemm (attn) perfConfig with the same seven default
// knob fields, plus the second-GEMM N tile nPerBlockG1 (0 == untiled).
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN
// CHECK-EXHAUSTIVE-ATTN: attn:mPerBlockG0=16,nPerBlockG0=16,nPerBlockG1=0,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1

// The second-GEMM N tile (nPerBlockG1) is only tuned when the head dim
// (head_dim_v) is large; for head_dim_v=512 the space spans the untiled case
// (0) and the power-of-two tiles {64,128,256}. Pin both the untiled and a
// tiled (nPerBlockG1=64) config to prove the knob is swept.
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 512 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN-G1
// CHECK-EXHAUSTIVE-ATTN-G1: attn:mPerBlockG0=16,nPerBlockG0=16,nPerBlockG1=0,kPerBlock=32,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1
// CHECK-EXHAUSTIVE-ATTN-G1: attn:mPerBlockG0=16,nPerBlockG0=16,nPerBlockG1=64,kPerBlock=32,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1

//===----------------------------------------------------------------------===//
// Attention tuning-space SIZE guards
//===----------------------------------------------------------------------===//
// These count the *number* of configs in the full attention tuning space, the
// gemm+gemm analogue of the `full set = N` GEMM check in
// test/CAPI/mixr_full.cpp. They make the effect of attention perfConfig knobs
// on the search space explicit: whenever a knob is added to / removed from the
// attn perfConfig (or its swept range changes), these totals move and must be
// updated -- a deliberate, reviewable signal.
//
// Baseline: a small head_dim_v keeps the second GEMM untiled (nPerBlockG1 == 0),
// so the nPerBlockG1 knob contributes only a single value.
// RUN: rocmlir-gen --arch gfx942 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 64 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --num_cu=304 --emit-tuning-space=full 2>/dev/null | wc -l | FileCheck %s --check-prefix=CHECK-ATTN-SPACE-UNTILED
// CHECK-ATTN-SPACE-UNTILED: {{^ *900$}}
//
// Same shape but a large head_dim_v (512) sweeps the second-GEMM N tile
// nPerBlockG1 over {0,64,128,256} (the untiled case plus 3 more),
// growing the space 4x. This is the knob added by the second-GEMM N
// (nPerBlockG1) head-dim tiling; varying only head_dim_v isolates its effect.
// RUN: rocmlir-gen --arch gfx942 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 512 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --num_cu=304 --emit-tuning-space=full 2>/dev/null | wc -l | FileCheck %s --check-prefix=CHECK-ATTN-SPACE-TILED
// CHECK-ATTN-SPACE-TILED: {{^ *3600$}}

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-GFX950-KPACK \
// RUN:       --implicit-check-not='kpack=2,'
// CHECK-MFMA-GFX950-KPACK: gemm:{{.*kpack=1,}}

// MAX_K_PER_BLOCK caps the per-block K tile at 512: even for a huge K (here
// 16384) the GEMM tuning space never emits a kPerBlock beyond 512. A tile
// larger than PowerOf2Ceil(K) would only pad K, and tiles above 512 waste LDS
// and are ~never optimal, so the PowerOf2Ceil(K) capping tile is not added once
// it would exceed 512. (With K thus capped and M/N <= 256, the lowered
// kPerBlock x (M|N) index/mask tensor also stays well under Triton's
// 2^20-element per-tensor cap.)
// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 256 -k 16384 -n 256 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-TENSOR-CAP-MFMA \
// RUN:       --implicit-check-not='kPerBlock=16384,'
// CHECK-TENSOR-CAP-MFMA: gemm:{{.*kPerBlock=512,}}

// The same MAX_K_PER_BLOCK cap applies to attention's gemm0 K tile: gemm0's K
// tile is PowerOf2Ceil(head_dim_qk), which can be huge (here head_dim_qk=8192),
// but the swept kPerBlock is still capped at 512.
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f16 -g 1 -head_dim_qk 8192 -head_dim_v 128 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-TENSOR-CAP-ATTN \
// RUN:       --implicit-check-not='kPerBlock=8192,'
// CHECK-TENSOR-CAP-ATTN: attn:{{.*kPerBlock=512,}}

// gemm1's N tile is not tuned: gemm1NPerBlock = PowerOf2Ceil(head_dim_v) and
// its contraction tile is gemm0NPerBlock, so gemm1 lowers a gemm0NPerBlock x
// max(gemm0MPerBlock, gemm1NPerBlock) index/mask tensor. A large head_dim_v can
// therefore blow the same 2^20 cap on a config gemm0's own check would pass.
// For head_dim_v=8192 the untiled gemm1NPerBlock is 8192, so an untiled
// (nPerBlockG1=0) gemm0NPerBlock=256 combo needs 256*8192 == 2097152 > cap and
// drops, regardless of gemm0's K tile (here head_dim_qk=128 keeps gemm0
// in-cap). A non-zero nPerBlockG1 shrinks the gemm1 N dim, so those tiled
// gemm0NPerBlock=256 combos can fit and are allowed; only the untiled 256
// combos (`nPerBlockG0=256,nPerBlockG1=0`) must be excluded.
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f16 -g 1 -head_dim_qk 128 -head_dim_v 8192 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-TENSOR-CAP-ATTN-V \
// RUN:       --implicit-check-not='nPerBlockG0=256,nPerBlockG1=0,'
// CHECK-TENSOR-CAP-ATTN-V: attn:mPerBlockG0=16,nPerBlockG0=128,

// Same gemm1 guard with a transposed output (-transO): head_dim_v then lives at
// a different position in C's shape, so the derivation must follow
// getTransposedC() (matching PopulateParamsGemmGemm::getGemm1Params). The cap
// behaviour must be identical -- every gemm0NPerBlock=256 combo still drops for
// head_dim_v=8192.
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f16 -g 1 -head_dim_qk 128 -head_dim_v 8192 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 -transO --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-TENSOR-CAP-ATTN-V-TRANSO \
// RUN:       --implicit-check-not='nPerBlockG0=256,nPerBlockG1=0,'
// CHECK-TENSOR-CAP-ATTN-V-TRANSO: attn:mPerBlockG0=16,nPerBlockG0=128,

//===----------------------------------------------------------------------===//
// LDS-overflow blacklist
//===----------------------------------------------------------------------===//

// Tile shapes known to overflow LDS for an (arch, input dtype) are pruned from
// the emitted tuning space by the compiled-in blacklist (LdsBlacklist.h,
// populated offline by generateLDSBlacklist.py). The 64x64x128 tile with
// numStages=3 is a shipped gfx950/f32 blacklist entry, so it must be absent
// from the default exhaustive space.
// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LDS-BLACKLIST \
// RUN:       --implicit-check-not='gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=3,'
// CHECK-LDS-BLACKLIST: gemm:mPerBlock=

// The ROCMLIR_DISABLE_LDS_BLACKLIST escape hatch makes LdsBlacklist::lookupGemm
// return an empty set, so the blacklisted config reappears -- this is what lets
// generateLDSBlacklist.py enumerate the *unfiltered* space to (re)generate the
// table idempotently.
// RUN: ROCMLIR_DISABLE_LDS_BLACKLIST=1 rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LDS-NOBLACKLIST
// CHECK-LDS-NOBLACKLIST: gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=3,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1

//===----------------------------------------------------------------------===//
// WMMA tuning space
//===----------------------------------------------------------------------===//

// RUN: rocmlir-gen --arch gfx1100 --operation=gemm -t f16 -g 1 -m 256 -k 128 -n 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-GEMM-KPACK-F16 \
// RUN:       --implicit-check-not='kpack=2,'
// CHECK-WMMA-GEMM-KPACK-F16: gemm:{{.*kpack=1,}}

// The WMMA attention tuning space applies the same hardcoding in
// `getAccelRangeGemmGemm`.
// RUN: rocmlir-gen --arch gfx1100 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-ATTN-KPACK \
// RUN:       --implicit-check-not='kpack=2,'
// CHECK-WMMA-ATTN-KPACK: attn:{{.*kpack=1,}}

//===----------------------------------------------------------------------===//
// Non-accel tuning space
//===----------------------------------------------------------------------===//

// NOTE ON STRUCTURE: `--implicit-check-not` and `CHECK-NOT` are only scanned
// over the input ranges *between* ordered directives, so adding a `CHECK-*-DAG`
// to a prefix silently stops that prefix's negative patterns from being
// enforced (a `-DAG` block spans the input, collapsing those ranges). Positive
// `-DAG` assertions therefore get their own prefix and RUN line here, and every
// negative assertion lives in a prefix that has *no* `-DAG` in it. Don't merge
// them back together: the test would still pass, but would stop checking.

// f32 on gfx1100 (RDNA3) has no matrix-accel instruction (WMMA has no f32
// mode), so this exercises the non-accel (FMA) path. Negative half: K tile is
// one of {4,8,16}, no kPack, no matrix-accel instruction.
// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI \
// RUN:       --implicit-check-not='{{kPerBlock=(32|64|128|256|512),}}' \
// RUN:       --implicit-check-not='kpack=2,' \
// RUN:       --implicit-check-not='{{matrixInstrNonkdim=(16|32),}}'
// CHECK-NAVI: gemm:{{mPerBlock=(16|32|64|128|256),nPerBlock=(16|32|64|128|256),kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1}}

// Positive half: the alternation above is a superset of the pre-AIROCMLIR-938
// {32,64,128} space, so it alone can't prove the extension took effect. Assert
// the two newly added non-accel tiles (16 and 256) on *both* the M and N axes;
// these fail if computeDPerBlock's non-accel extension is reverted.
// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-TILES
// CHECK-NAVI-TILES-DAG: gemm:mPerBlock=16,{{nPerBlock=[0-9]+,kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1}}
// CHECK-NAVI-TILES-DAG: gemm:mPerBlock=256,{{nPerBlock=[0-9]+,kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1}}
// CHECK-NAVI-TILES-DAG: gemm:{{mPerBlock=[0-9]+}},nPerBlock=16,{{kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1}}
// CHECK-NAVI-TILES-DAG: gemm:{{mPerBlock=[0-9]+}},nPerBlock=256,{{kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1}}

//===----------------------------------------------------------------------===//
// Non-accel overwide M/N pair filter (isOverwideNonAccelMNPair)
//===----------------------------------------------------------------------===//

// On the non-accel (FMA) path a 256 M/N tile is only tuned when the *other*
// tile stays below 128. The blockwise GEMM is a fully unrolled scalar loop, so
// `{128,256}`, `{256,128}` and `{256,256}` cost ~70% of the whole space's
// compile time on gfx1201 (256x256 alone is ~10x a 128x128), and exhaustive
// Navi tuning never picked one of them as the best config for any GEMM or
// convolution shape (AIROCMLIR-938). These RUNs use `exhaustive` so that no
// `couldBePerformant` heuristic can mask the filter.
// RUN: rocmlir-gen --arch gfx1201 --operation=gemm -t f32 -g 1 -m 2048 -n 2048 -k 2048 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-WIDE-MN-GEMM
// CHECK-NO-WIDE-MN-GEMM-NOT: gemm:mPerBlock=128,nPerBlock=256,
// CHECK-NO-WIDE-MN-GEMM-NOT: gemm:mPerBlock=256,nPerBlock=128,
// CHECK-NO-WIDE-MN-GEMM-NOT: gemm:mPerBlock=256,nPerBlock=256,

// The other half: the filter must drop only the wide *pairs*, leaving the
// narrow 256 tiles on both the M and N axes (those are what actually won) and
// the 128x128 baseline pair untouched. Fails if the predicate is too broad.
// RUN: rocmlir-gen --arch gfx1201 --operation=gemm -t f32 -g 1 -m 2048 -n 2048 -k 2048 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NARROW-MN-GEMM
// CHECK-NARROW-MN-GEMM-DAG: gemm:mPerBlock=128,nPerBlock=128,
// CHECK-NARROW-MN-GEMM-DAG: gemm:mPerBlock=256,nPerBlock=64,
// CHECK-NARROW-MN-GEMM-DAG: gemm:mPerBlock=64,nPerBlock=256,

// Convolution reaches the same filter through `createGemmTuningRangeBF`, and is
// the operation where tile 256 won most often (15/99 shapes), so pin it here
// too rather than relying on the plain-GEMM RUNs above.
// RUN: rocmlir-gen --arch gfx1201 --operation=conv -t f32 --groupsize=1 --batchsize=8 --in_channels=256 --in_h=32 --in_w=32 --out_channels=256 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-WIDE-MN-CONV
// CHECK-NO-WIDE-MN-CONV-NOT: gemm:mPerBlock=128,nPerBlock=256,
// CHECK-NO-WIDE-MN-CONV-NOT: gemm:mPerBlock=256,nPerBlock=128,
// CHECK-NO-WIDE-MN-CONV-NOT: gemm:mPerBlock=256,nPerBlock=256,

// RUN: rocmlir-gen --arch gfx1201 --operation=conv -t f32 --groupsize=1 --batchsize=8 --in_channels=256 --in_h=32 --in_w=32 --out_channels=256 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NARROW-MN-CONV
// CHECK-NARROW-MN-CONV-DAG: gemm:mPerBlock=128,nPerBlock=128,
// CHECK-NARROW-MN-CONV-DAG: gemm:mPerBlock=256,nPerBlock=64,
// CHECK-NARROW-MN-CONV-DAG: gemm:mPerBlock=64,nPerBlock=256,

// Negative control: the filter is gated on MatrixAccelKind::None, so an
// accelerated path must keep the wide pairs. f16 on gfx1201 (RDNA4) has WMMA,
// where a 256x256 tile compiles ~4x faster than the FMA one despite a K tile
// four times larger. This fails if the filter is applied unconditionally.
// RUN: rocmlir-gen --arch gfx1201 --operation=gemm -t f16 -g 1 -m 2048 -n 2048 -k 2048 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-WIDE-MN
// CHECK-WMMA-WIDE-MN-DAG: gemm:mPerBlock=128,nPerBlock=256,
// CHECK-WMMA-WIDE-MN-DAG: gemm:mPerBlock=256,nPerBlock=128,
// CHECK-WMMA-WIDE-MN-DAG: gemm:mPerBlock=256,nPerBlock=256,

// f32 attention on gfx1100 (RDNA3) has no matrix-accel instruction either,
// so this exercises the non-accel `getRangeGemmGemm` path.
// RUN: rocmlir-gen --arch gfx1100 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-ATTN \
// RUN:       --implicit-check-not='kpack=2,' \
// RUN:       --implicit-check-not='{{matrixInstrNonkdim=(16|32),}}' \
// RUN:       --implicit-check-not='{{mPerBlockG0=(16|256),}}' \
// RUN:       --implicit-check-not='{{nPerBlockG0=(16|256),}}'
// CHECK-NAVI-ATTN: attn:{{mPerBlockG0=(32|64|128),nPerBlockG0=(32|64|128),nPerBlockG1=0,kPerBlock=[0-9]+,kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=(1|2),numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1,useBf16x3ForF32=-1}}

//===----------------------------------------------------------------------===//
// Non-power-of-two kPerBlock candidates that evenly divide K
//===----------------------------------------------------------------------===//

// When K has non-power-of-two divisors (here K = 576 = 64*3*3, i.e. a conv's
// C*fil_h*fil_w), the accelerated paths offer, per (m,n) tile, the divisors of K
// that peel into two pow2 segments and fall under the window
// [min(m,n)/2, min(m,n)). On WMMA (gfx1201) a min(m,n)=64 tile
// gains 36 and 48 on top of the pow2 base {32,64}.
// RUN: rocmlir-gen --arch gfx1201 --operation=gemm -t f16 -g 1 -m 256 -k 576 -n 256 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-DIVK
// CHECK-WMMA-DIVK: gemm:{{.*kPerBlock=48,kpack=1,}}

// The MFMA path grows the same way; 48 (= 576/12) is offered on gfx942.
// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t f16 -g 1 -m 256 -k 576 -n 256 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-DIVK
// CHECK-MFMA-DIVK: gemm:{{.*kPerBlock=48,kpack=1,}}

// An integer GEMM accumulates in i32 and keeps that accumulator exact only
// while every K decomposed segment is at least 4 wide (enforced in
// rock-gridwise-gemm-to-blockwise), so tiles that would peel into a 1- or
// 2-wide segment are not offered. K = 576 makes 18 (= 16 + 2) a divisor in the
// min(m,n)=32 window: it is offered for f16 but not for i8, while the
// multiple-of-4 tiles stay for both.
// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t f16 -g 1 -m 256 -k 576 -n 256 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-NARROW-SEG-F16
// CHECK-MFMA-NARROW-SEG-F16: gemm:{{.*kPerBlock=18,kpack=1,}}

// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t i8 -g 1 -m 256 -k 576 -n 256 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-NARROW-SEG-I8 \
// RUN:       --implicit-check-not='kPerBlock=18,'
// CHECK-MFMA-NARROW-SEG-I8: gemm:{{.*kPerBlock=24,kpack=1,}}

// A K that is a pure power of two (K = 128) must not introduce any non-pow2
// kPerBlock: only 32/64/128 appear on WMMA (256 is capped out by K = 128).
// RUN: rocmlir-gen --arch gfx1201 --operation=gemm -t f16 -g 1 -m 256 -k 128 -n 256 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-POW2K \
// RUN:       --implicit-check-not='{{kPerBlock=(36|48|72|96|144|192),}}'
// CHECK-WMMA-POW2K: gemm:{{.*kPerBlock=(32|64|128),kpack=1,}}

//===----------------------------------------------------------------------===//
// Widened kPerBlock range for a K no pow2 tile divides
//===----------------------------------------------------------------------===//

// When no power-of-two tile divides K, every reachable config masks a K
// remainder on each iteration, so the divisors of K in (16, 64] are offered as
// well -- bypassing the two-segment and window rules that would otherwise keep
// them out. A convolution's K is a single Merge of the input's channel and
// filter-spatial dims, and only a kPerBlock that is a multiple of the merge's
// trailing extents advances it without a coordinate carry -- which is what keeps
// the address arithmetic affine and the padding mask hoistable. So for a
// channels-first (ngc01) K = 3483 (C=387, 3x3), whose trailing product is 3*3,
// the range offers 27 = 3*(3*3) but not 43 (= 32+8+2+1). Every tile exhaustive
// tuning picked from the widened range on the AIROCMLIR-1182 convs was such a
// multiple.
// RUN: rocmlir-gen --arch gfx1101 --operation=conv -t f32 --groupsize=1 --batchsize=8 --in_channels=387 --in_h=32 --in_w=32 --out_channels=128 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-WIDEN-CONV \
// RUN:       --implicit-check-not='kPerBlock=43,'
// CHECK-NAVI-WIDEN-CONV: gemm:{{.*kPerBlock=27,kpack=1,}}

// A channels-last (nhwgc) input reorders that same merge to Merge(y, x, c), so
// the trailing product spans the channel dim: 3*387 = 1161, which no tile in the
// space can be a multiple of, so nothing is added. That layout needs no aligned
// tile anyway -- a K tile narrower than C leaves y/x uniform across the whole
// tile, so the mask is already loop-invariant. 27 is the witness, since it is
// reachable only through the widened range; 9 stays because rule (3)'s window
// admits it independently (9 = 8+1 lands in the 16-wide tile's window).
// RUN: rocmlir-gen --arch gfx1101 --operation=conv -t f32 --groupsize=1 --batchsize=8 --in_channels=387 --in_h=32 --in_w=32 --out_channels=128 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --fil_layout=gkyxc --in_layout=nhwgc --out_layout=nhwgk --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-WIDEN-CONV-NHWC \
// RUN:       --implicit-check-not='{{kPerBlock=(27|43),}}'
// CHECK-NAVI-WIDEN-CONV-NHWC: gemm:{{.*kPerBlock=9,kpack=1,}}

// A plain GEMM has no such merge, so there is no alignment to narrow the range
// and it stays shut for now, awaiting speedup numbers to weigh its growth
// against (see the TODO in computeKPerBlock). The same K = 3483 as a GEMM
// therefore offers neither the conv's aligned 27 nor the 43 that only the
// unfiltered range would reach; 9 stays, being window-reachable.
// RUN: rocmlir-gen --arch gfx1101 --operation=gemm -t f32 -g 1 -m 128 -k 3483 -n 128 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-NO-WIDEN-GEMM \
// RUN:       --implicit-check-not='{{kPerBlock=(27|43),}}'
// CHECK-NAVI-NO-WIDEN-GEMM: gemm:{{.*kPerBlock=9,kpack=1,}}

// The widened range runs further on an accelerated path, which stages its
// operands through LDS rather than registers and so is not held to the non-accel
// register ceiling of 64. It has to: WMMA's pow2 tiles are only {32,64}, so a
// bound of 64 would leave the range entirely inside the tiles it already has.
// In exchange, a tile there must fill the matrix instruction, so on WMMA the two
// requirements combine: a multiple of both the merge's trailing product and the
// instruction's 16, which for a channels-first 3x3 filter means multiples of
// lcm(9,16) = 144. That is past the accel bound of 128, and the range runs to it
// anyway, since stopping short would open the range and then admit nothing from
// it. So with f16 on gfx1101 (RDNA3), where neither 32 nor 64 divides K = 3024,
// 144 is the whole of the widened range: 108 and 126 divide K and land in the
// range but cannot fill instructions (126 peels into 64+32+16+8+4+2, whose last
// three segments would each be padded up to a full instruction), and the
// unaligned 112 is out even though it does fill them.
// RUN: rocmlir-gen --arch gfx1101 --operation=conv -t f16 --groupsize=1 --batchsize=1 --in_channels=336 --in_h=24 --in_w=24 --out_channels=336 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-WIDEN-CONV \
// RUN:       --implicit-check-not='kPerBlock=108,' \
// RUN:       --implicit-check-not='kPerBlock=112,' \
// RUN:       --implicit-check-not='kPerBlock=126,'
// CHECK-WMMA-WIDEN-CONV: gemm:{{.*kPerBlock=144,kpack=1,}}

// MFMA is held to the same rule, but its instruction is narrower: f16 consumes
// 8 at matrixInstrNonkdim=32, and since that knob is swept outside the tuning
// space's K axis a tile is kept when either setting could use it. So the bar
// here is 8, and lcm(9, 8) = 72 still fits under the bound where WMMA's 144 does
// not. For a 3x3 conv over C = 72 (K = 648) the widened range therefore keeps 72
// and drops 27, 54 and 108, which divide K and land in the range but cannot fill
// instructions.
// RUN: rocmlir-gen --arch gfx942 --operation=conv -t f16 --groupsize=1 --batchsize=1 --in_channels=72 --in_h=384 --in_w=384 --out_channels=72 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-WIDEN-CONV \
// RUN:       --implicit-check-not='{{kPerBlock=(27|54|108),}}'
// CHECK-MFMA-WIDEN-CONV: gemm:{{.*kPerBlock=72,kpack=1,}}

// The bar is read out of Triton's own instruction tables rather than mirrored
// here, which matters wherever they disagree with what the arch would suggest.
// gfx942 has native fp8 MFMA whose narrowest K is 16, so a 3x3 conv in the FNUZ
// spelling aligns to lcm(9,16) = 144, exactly as f16 does on WMMA. Triton has no
// intrinsic for the OCP spelling before CDNA4 though, and quietly substitutes
// f16 (composeMfmaKeyFor in MfmaGroup.cpp), so the bar there is f16's 8 and the
// same arch and shape align to lcm(9,8) = 72 instead. Over C = 32 (K = 288) both
// are divisors of K, and each spelling offers only its own: 144 is past the OCP
// alignment's bound of 128, and 72 is not a multiple of the FNUZ one. Neither is
// window-reachable, since M = out_channels = 64 caps every window at 64.
// RUN: rocmlir-gen --arch gfx942 --operation=conv -t f8E4M3FNUZ --groupsize=1 --batchsize=1 --in_channels=32 --in_h=20 --in_w=20 --out_channels=64 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-WIDEN-FP8 \
// RUN:       --implicit-check-not='kPerBlock=72,'
// CHECK-MFMA-WIDEN-FP8: gemm:{{.*kPerBlock=144,}}

// RUN: rocmlir-gen --arch gfx942 --operation=conv -t f8E4M3FN --groupsize=1 --batchsize=1 --in_channels=32 --in_h=20 --in_w=20 --out_channels=64 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-WIDEN-FP8-OCP \
// RUN:       --implicit-check-not='kPerBlock=144,'
// CHECK-MFMA-WIDEN-FP8-OCP: gemm:{{.*kPerBlock=72,}}

// Dividing K evenly is not on its own enough to call a tile good on a conv. An
// int8 3x3 conv over C=256 has K = 2304, which 32 divides exactly, so the pow2
// list does offer a remainder-free tile -- but 32 spans three and a half filter
// windows, so every iteration still straddles one. The gate therefore opens on
// the alignment as well as the division, and 144 = lcm(9,16) is offered. It
// cannot come from rule (3) here: M is out_channels = 64, so no tile is wide
// enough for a window that reaches 144.
// RUN: rocmlir-gen --arch gfx1101 --operation=conv -t i8 --groupsize=1 --batchsize=1 --in_channels=256 --in_h=20 --in_w=20 --out_channels=64 --fil_h=3 --fil_w=3 --padding_h=1 --padding_w=1 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-ALIGN-GATE \
// RUN:       --implicit-check-not='{{kPerBlock=(96|108|126|192),}}'
// CHECK-WMMA-ALIGN-GATE: gemm:{{.*kPerBlock=144,kpack=1,}}
