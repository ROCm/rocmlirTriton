//===----------------------------------------------------------------------===//
// MFMA tuning space
//===----------------------------------------------------------------------===//

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=104 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=16,splitKFactor=4,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=32,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-SPLITKFACTOR
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=3,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=4,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-NUMSTAGES
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=3,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1

// Attention emits the gemm+gemm (attn) perfConfig with the same seven default
// knob fields, plus the second-GEMM N tile nPerBlockG1 (0 == untiled).
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN
// CHECK-EXHAUSTIVE-ATTN: attn:mPerBlockG0=16,nPerBlockG0=16,nPerBlockG1=0,kPerBlock=16,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1

// The second-GEMM N tile (nPerBlockG1) is only tuned when the head dim
// (head_dim_v) is large; for head_dim_v=512 the space spans the untiled case
// (0) and the power-of-two tiles {64,128,256}. Pin both the untiled and a
// tiled (nPerBlockG1=64) config to prove the knob is swept.
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 512 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN-G1
// CHECK-EXHAUSTIVE-ATTN-G1: attn:mPerBlockG0=16,nPerBlockG0=16,nPerBlockG1=0,kPerBlock=32,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1
// CHECK-EXHAUSTIVE-ATTN-G1: attn:mPerBlockG0=16,nPerBlockG0=16,nPerBlockG1=64,kPerBlock=32,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=1,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1

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
// CHECK-LDS-NOBLACKLIST: gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,kpack=1,numCTAs=1,numWaves=1,matrixInstrNonkdim=16,splitKFactor=1,numStages=3,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1

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
// CHECK-NAVI: gemm:{{mPerBlock=(16|32|64|128|256),nPerBlock=(16|32|64|128|256),kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1}}

// Positive half: the alternation above is a superset of the pre-AIROCMLIR-938
// {32,64,128} space, so it alone can't prove the extension took effect. Assert
// the two newly added non-accel tiles (16 and 256) on *both* the M and N axes;
// these fail if computeDPerBlock's non-accel extension is reverted.
// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-TILES
// CHECK-NAVI-TILES-DAG: gemm:mPerBlock=16,{{nPerBlock=[0-9]+,kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1}}
// CHECK-NAVI-TILES-DAG: gemm:mPerBlock=256,{{nPerBlock=[0-9]+,kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1}}
// CHECK-NAVI-TILES-DAG: gemm:{{mPerBlock=[0-9]+}},nPerBlock=16,{{kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1}}
// CHECK-NAVI-TILES-DAG: gemm:{{mPerBlock=[0-9]+}},nPerBlock=256,{{kPerBlock=(4|8|16),kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=[0-9]+,numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1}}

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
// CHECK-NAVI-ATTN: attn:{{mPerBlockG0=(32|64|128),nPerBlockG0=(32|64|128),nPerBlockG1=0,kPerBlock=[0-9]+,kpack=1,numCTAs=[0-9]+,numWaves=[0-9]+,matrixInstrNonkdim=0,splitKFactor=(1|2),numStages=[0-9]+,wavesPerEU=0,gridGroupSize=0,useAsyncCopy=-1,useBlockPingpong=-1,useInThreadTranspose=-1,useBufferOps=-1,useBufferAtomics=-1,useReductionLayout=-1,useOptimizeEpilogue=-1}}

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
