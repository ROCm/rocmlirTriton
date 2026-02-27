// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-NAVI
// CHECK-NAVI: gemm:v1:64,64,32,1,1,8,0,1,2,0,0

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=104 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: gemm:v1:64,64,128,2,1,4,16,4,1,0,0

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v1:16,16,16,1,1,1,16,1,1,0,0
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v1:16,16,16,1,1,1,32,1,1,0,0

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-SPLITKFACTOR
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v1:16,16,16,1,1,1,16,1,1,0,0
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v1:16,16,16,1,1,1,16,3,1,0,0
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v1:16,16,16,1,1,1,16,4,1,0,0

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-NUMSTAGES
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v1:16,16,16,1,1,1,16,1,1,0,0
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v1:16,16,16,1,1,1,16,1,2,0,0
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v1:16,16,16,1,1,1,16,1,3,0,0

// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN
// CHECK-EXHAUSTIVE-ATTN: attn:v1:16,16,16,1,1,1,16,1,1,0,0
