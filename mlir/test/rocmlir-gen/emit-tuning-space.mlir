//===----------------------------------------------------------------------===//
// MFMA tuning space
//===----------------------------------------------------------------------===//

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=104 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: gemm:v4:64,64,128,1,1,4,16,4,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v4:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v4:16,16,16,1,1,1,32,1,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-SPLITKFACTOR
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v4:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v4:16,16,16,1,1,1,16,3,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v4:16,16,16,1,1,1,16,4,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-NUMSTAGES
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v4:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v4:16,16,16,1,1,1,16,1,2,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v4:16,16,16,1,1,1,16,1,3,0,0,-1,-1,-1,-1,-1,-1

// Attention emits the gemm+gemm (attn) perfConfig with the same five default
// knob fields; there is no longer a schedule-hint knob.
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN
// CHECK-EXHAUSTIVE-ATTN: attn:v4:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-GFX950-KPACK \
// RUN:       --implicit-check-not="gemm:v4:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-MFMA-GFX950-KPACK: gemm:v4:{{[0-9]+,[0-9]+,[0-9]+,1,}}

// A large-K GEMM must not emit tile combos whose lowered index/mask tensors
// exceed Triton's 2^20-element per-tensor cap: for k=16384 that means any tile
// with max(mPerBlock, nPerBlock) > 64 (64*16384 == 1048576, the exact cap).
// Such combos fail Triton's verifyTensorSize and, in exhaustive mode, abort the
// whole config. The PowerOf2Ceil(K) capping tile (16384) must still appear,
// paired only with in-cap M/N so K stays covered.
// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 256 -k 16384 -n 256 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-TENSOR-CAP-MFMA \
// RUN:       --implicit-check-not="gemm:v4:{{(128|256)}},{{[0-9]+}},16384," \
// RUN:       --implicit-check-not="gemm:v4:{{[0-9]+}},{{(128|256)}},16384,"
// CHECK-TENSOR-CAP-MFMA: gemm:v4:64,64,16384,

// The same per-tensor cap applies to attention (gemm+gemm): gemm0's K tile is
// PowerOf2Ceil(head_dim_qk), so a large head_dim can push kPerBlock high enough
// that kPerBlock * max(mPerBlock, nPerBlock) exceeds 2^20 and the lowered
// index/mask tensor fails Triton's verifyTensorSize. For head_dim_qk=8192 the
// capping tile is 8192, so max(m,n) must stay <= 128 (128*8192 == 1048576).
// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f16 -g 1 -head_dim_qk 8192 -head_dim_v 128 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-TENSOR-CAP-ATTN \
// RUN:       --implicit-check-not="attn:v4:256,{{[0-9]+}},8192," \
// RUN:       --implicit-check-not="attn:v4:{{[0-9]+}},256,8192,"
// CHECK-TENSOR-CAP-ATTN: attn:v4:128,128,8192,

//===----------------------------------------------------------------------===//
// WMMA tuning space
//===----------------------------------------------------------------------===//

// RUN: rocmlir-gen --arch gfx1100 --operation=gemm -t f16 -g 1 -m 256 -k 128 -n 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-GEMM-KPACK-F16 \
// RUN:       --implicit-check-not="gemm:v4:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-WMMA-GEMM-KPACK-F16: gemm:v4:{{[0-9]+,[0-9]+,[0-9]+,1,}}

// The WMMA attention tuning space applies the same hardcoding in
// `getAccelRangeGemmGemm`.
// RUN: rocmlir-gen --arch gfx1100 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-ATTN-KPACK \
// RUN:       --implicit-check-not="attn:v4:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-WMMA-ATTN-KPACK: attn:v4:{{[0-9]+,[0-9]+,[0-9]+,1,}}

//===----------------------------------------------------------------------===//
// Non-accel tuning space
//===----------------------------------------------------------------------===//

// f32 on gfx1100 (RDNA3) has no matrix-accel instruction (WMMA has no f32
// mode), so this exercises the non-accel (FMA) path.
// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI \
// RUN:       --implicit-check-not="gemm:v4:{{[0-9]+,[0-9]+,(32|64|128|256|512),}}" \
// RUN:       --implicit-check-not="gemm:v4:{{[0-9]+,[0-9]+,[0-9]+,2,}}" \
// RUN:       --implicit-check-not="gemm:v4:{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,(16|32),}}"
// CHECK-NAVI: gemm:v4:{{(32|64|128),(32|64|128),(4|8|16),1,[0-9]+,[0-9]+,0,[0-9]+,[0-9]+,0,0,-1,-1,-1,-1,-1,-1}}

// f32 attention on gfx1100 (RDNA3) has no matrix-accel instruction either,
// so this exercises the non-accel `getRangeGemmGemm` path.
// RUN: rocmlir-gen --arch gfx1100 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-ATTN \
// RUN:       --implicit-check-not="attn:v4:{{[0-9]+,[0-9]+,[0-9]+,2,}}" \
// RUN:       --implicit-check-not="attn:v4:{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,(16|32),}}" \
// RUN:       --implicit-check-not="attn:v4:{{(16|256),}}" \
// RUN:       --implicit-check-not="attn:v4:{{[0-9]+,(16|256),}}"
// CHECK-NAVI-ATTN: attn:v4:{{(32|64|128),(32|64|128),[0-9]+,1,[0-9]+,[0-9]+,0,(1|2),[0-9]+,0,0,-1,-1,-1,-1,-1,-1}}
