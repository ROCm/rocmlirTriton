//===----------------------------------------------------------------------===//
// MFMA tuning space
//===----------------------------------------------------------------------===//

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=104 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: gemm:v2:64,64,128,1,1,4,16,4,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v2:16,16,16,1,1,1,32,1,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-SPLITKFACTOR
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v2:16,16,16,1,1,1,16,3,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v2:16,16,16,1,1,1,16,4,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-NUMSTAGES
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v2:16,16,16,1,1,1,16,1,2,0,0,-1,-1,-1,-1,-1,-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v2:16,16,16,1,1,1,16,1,3,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN
// CHECK-EXHAUSTIVE-ATTN: attn:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-GFX950-KPACK \
// RUN:       --implicit-check-not="gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-MFMA-GFX950-KPACK: gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,1,}}

//===----------------------------------------------------------------------===//
// WMMA tuning space
//===----------------------------------------------------------------------===//

// RUN: rocmlir-gen --arch gfx1100 --operation=gemm -t f16 -g 1 -m 256 -k 128 -n 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-GEMM-KPACK-F16 \
// RUN:       --implicit-check-not="gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-WMMA-GEMM-KPACK-F16: gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,1,}}

// The WMMA attention tuning space applies the same hardcoding in
// `getAccelRangeGemmGemm`.
// RUN: rocmlir-gen --arch gfx1100 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-ATTN-KPACK \
// RUN:       --implicit-check-not="attn:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-WMMA-ATTN-KPACK: attn:v2:{{[0-9]+,[0-9]+,[0-9]+,1,}}

//===----------------------------------------------------------------------===//
// Non-accel tuning space
//===----------------------------------------------------------------------===//

// f32 on gfx1100 (RDNA3) has no matrix-accel instruction (WMMA has no f32
// mode), so this exercises the non-accel (FMA) path.
// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI \
// RUN:       --implicit-check-not="gemm:v2:{{[0-9]+,[0-9]+,(32|64|128|256|512),}}" \
// RUN:       --implicit-check-not="gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}" \
// RUN:       --implicit-check-not="gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,(16|32),}}"
// CHECK-NAVI: gemm:v2:{{(32|64|128),(32|64|128),(4|8|16),1,[0-9]+,[0-9]+,0,[0-9]+,[0-9]+,0,0,-1,-1,-1,-1,-1,-1}}

// f32 attention on gfx1100 (RDNA3) has no matrix-accel instruction either,
// so this exercises the non-accel `getRangeGemmGemm` path.
// RUN: rocmlir-gen --arch gfx1100 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NAVI-ATTN \
// RUN:       --implicit-check-not="attn:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}" \
// RUN:       --implicit-check-not="attn:v2:{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,(16|32),}}" \
// RUN:       --implicit-check-not="attn:v2:{{(16|256),}}" \
// RUN:       --implicit-check-not="attn:v2:{{[0-9]+,(16|256),}}"
// CHECK-NAVI-ATTN: attn:v2:{{(32|64|128),(32|64|128),[0-9]+,1,[0-9]+,[0-9]+,0,(1|2),[0-9]+,0,0,-1,-1,-1,-1,-1,-1}}
