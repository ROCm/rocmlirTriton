// Verifies that the rock-affix-params pass hard-fails (with a clear
// diagnostic) when a user-provided perf_config violates one of the per-field
// validators in `validatePerfConfig`. The valid baseline for gfx90a is
//
//   gemm:v1:128,128,128,1,1,4,16,1,2,0,0
//             ^   ^   ^  ^ ^ ^  ^ ^ ^ ^ ^
//   mPerBlock |   |   |  | | |  | | | | gridGroupSize
//             nPerBlock |  | | |  | | | wavesPerEU
//                 kPerBlock | | |  | | numStages
//                       kpack | |  | splitKFactor
//                         numCTAs|  matrixInstrNonkdim
//                             numWaves
//
// Each RUN line perturbs exactly one field to exercise a single validator
// branch; the field order also matches the order in which the validators run
// inside `validatePerfConfig`, so each error is unambiguously attributable.

// ---- mPerBlock / nPerBlock: positive (gemm) vs power-of-two (gemm+gemm) ----
// rock-decompose-nonpow2-tiles lets a plain gemm use non-power-of-two M/N, so
// gemm only requires them to be positive. gemm+gemm (attention) is not lowered
// through that pass, so it still requires powers of two.

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:0,128,128,1,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=MPERBLOCK-NON-POSITIVE

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,0,128,1,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NPERBLOCK-NON-POSITIVE

// RUN: rocmlir-gen --operation attention --arch gfx90a -t f16 \
// RUN:   -seq_len_q 256 -seq_len_k 256 -head_dim_qk 64 -head_dim_v 64 -p \
// RUN:   --perf_config "attn:v1:3,64,32,1,1,1,0,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=MPERBLOCK-NOT-POW2

// RUN: rocmlir-gen --operation attention --arch gfx90a -t f16 \
// RUN:   -seq_len_q 256 -seq_len_k 256 -head_dim_qk 64 -head_dim_v 64 -p \
// RUN:   --perf_config "attn:v1:64,3,32,1,1,1,0,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NPERBLOCK-NOT-POW2

// Scaled GEMMs are not lowered through rock-decompose-nonpow2-tiles either, so
// they keep the power-of-two M/N requirement.

// RUN: rocmlir-gen -t f4E2M1FN -m 80 -n 80 -k 256 -out_dtype f32 --scaledGemm \
// RUN:   --arch gfx950 --operation gemm -p \
// RUN:   --perf_config "gemm:v1:80,128,128,1,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=SCALED-MPERBLOCK-NOT-POW2

// RUN: rocmlir-gen -t f4E2M1FN -m 80 -n 80 -k 256 -out_dtype f32 --scaledGemm \
// RUN:   --arch gfx950 --operation gemm -p \
// RUN:   --perf_config "gemm:v1:128,80,128,1,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=SCALED-NPERBLOCK-NOT-POW2

// ---- kPerBlock: positive (gemm) vs power-of-two (gemm+gemm / scaled) -------

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,0,1,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=KPERBLOCK-NON-POSITIVE

// RUN: rocmlir-gen --operation attention --arch gfx90a -t f16 \
// RUN:   -seq_len_q 256 -seq_len_k 256 -head_dim_qk 64 -head_dim_v 64 -p \
// RUN:   --perf_config "attn:v1:64,64,3,1,1,1,0,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=KPERBLOCK-NOT-POW2

// RUN: rocmlir-gen -t f4E2M1FN -m 80 -n 80 -k 256 -out_dtype f32 --scaledGemm \
// RUN:   --arch gfx950 --operation gemm -p \
// RUN:   --perf_config "gemm:v1:128,128,3,1,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=SCALED-KPERBLOCK-NOT-POW2

// ---- validateKpack: both branches (non-positive, exceeds max) -------------

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,0,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=KPACK-NON-POSITIVE

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,16,1,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=KPACK-TOO-LARGE

// ---- validateNumCTAs: < 1, not pow2, and > maxNumCTAs ---------------------
// The pow2 check fires only when the bound check doesn't, so we exercise it
// on a multi-CTA arch (gfx1250, maxNumCTAs = 16) with numCTAs = 3.

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,0,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NUMCTAS-NON-POSITIVE

// RUN: rocmlir-gen --operation gemm --arch gfx1250 -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,3,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NUMCTAS-NOT-POW2

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,2,4,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NUMCTAS-TOO-LARGE

// ---- validateNumWaves: not pow2 and > maxNumWaves -------------------------

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,3,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NUMWAVES-NOT-POW2

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,32,16,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NUMWAVES-TOO-LARGE

// ---- validateMatrixInstrNonkdim: non-zero & not pow2 ----------------------

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,4,17,1,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=MATRIX-INSTR-BAD

// ---- validateSplitKFactor: < 1 (gemm) and != 1 (attention) ----------------

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,4,16,0,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=SPLITK-NON-POSITIVE

// RUN: rocmlir-gen --operation attention --arch gfx90a -t f16 \
// RUN:   -seq_len_q 256 -seq_len_k 256 -head_dim_qk 64 -head_dim_v 64 -p \
// RUN:   --perf_config "attn:v1:64,64,32,1,1,1,0,2,2,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=ATTN-SPLITK

// ---- validateNumStages: < 1 -----------------------------------------------

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,4,16,1,0,0,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=NUMSTAGES-NON-POSITIVE

// ---- validateWavesPerEU: negative and > maxWavesPerEU ---------------------

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,4,16,1,2,-1,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=WAVESPEREU-NEGATIVE

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,4,16,1,2,9,0" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=WAVESPEREU-TOO-LARGE

// ---- validateGridGroupSize: negative -------------------------------------
// gridGroupSize has no static upper bound (the heuristic in
// `makeGroupedGridLayout` itself can produce values larger than the
// per-arch chiplet count). The only structural check is non-negativity.

// RUN: rocmlir-gen --operation gemm --arch gfx90a -p -t f16 \
// RUN:   --perf_config "gemm:v1:128,128,128,1,1,4,16,1,2,0,-1" \
// RUN: | not rocmlir-opt -rock-affix-params 2>&1 \
// RUN: | FileCheck %s --check-prefix=GRIDGROUP-NEGATIVE

// MPERBLOCK-NON-POSITIVE: error: mPerBlock=0 must be positive
// NPERBLOCK-NON-POSITIVE: error: nPerBlock=0 must be positive
// MPERBLOCK-NOT-POW2:    error: mPerBlock=3 must be a positive power of two
// NPERBLOCK-NOT-POW2:    error: nPerBlock=3 must be a positive power of two
// SCALED-MPERBLOCK-NOT-POW2: error: mPerBlock=80 must be a positive power of two
// SCALED-NPERBLOCK-NOT-POW2: error: nPerBlock=80 must be a positive power of two
// KPERBLOCK-NON-POSITIVE: error: kPerBlock=0 must be positive
// KPERBLOCK-NOT-POW2:    error: kPerBlock=3 must be a positive power of two
// SCALED-KPERBLOCK-NOT-POW2: error: kPerBlock=3 must be a positive power of two
// KPACK-NON-POSITIVE:    error: kpack=0 must be positive
// KPACK-TOO-LARGE:       error: kpack=16 exceeds max (2) for amdgcn-amd-amdhsa:gfx90a
// NUMCTAS-NON-POSITIVE:  error: numCTAs=0 must be >= 1
// NUMCTAS-NOT-POW2:      error: numCTAs=3 must be a positive power of two
// NUMCTAS-TOO-LARGE:     error: numCTAs=2 exceeds max (1) for amdgcn-amd-amdhsa:gfx90a
// NUMWAVES-NOT-POW2:     error: numWaves=3 must be a positive power of two
// NUMWAVES-TOO-LARGE:    error: numWaves=32 * waveSize=64 exceeds max workgroup size (1024)
// MATRIX-INSTR-BAD:      error: matrixInstrNonkdim=17 must be 0 (heuristic) or a positive power of two
// SPLITK-NON-POSITIVE:   error: splitKFactor=0 must be >= 1
// ATTN-SPLITK:           error: splitKFactor=2 must be 1 for attention
// NUMSTAGES-NON-POSITIVE: error: numStages=0 must be >= 1
// WAVESPEREU-NEGATIVE:   error: wavesPerEU=-1 must be >= 0
// WAVESPEREU-TOO-LARGE:  error: wavesPerEU=9 exceeds max (8) for amdgcn-amd-amdhsa:gfx90a
// GRIDGROUP-NEGATIVE:    error: gridGroupSize=-1 must be >= 0
