// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-NAVI
// CHECK-NAVI: gemm:v2:64,64,32,1,1,8,0,1,2,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=104 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: gemm:v2:64,64,128,1,1,4,16,4,1,0,0,-1,-1,-1,-1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,0,-1,-1
// CHECK-EXHAUSTIVE-MATRIXINSTRNONKDIM: gemm:v2:16,16,16,1,1,1,32,1,1,0,0,-1,-1,-1,0,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-SPLITKFACTOR
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,0,-1,-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v2:16,16,16,1,1,1,16,3,1,0,0,-1,-1,-1,0,-1,-1
// CHECK-EXHAUSTIVE-SPLITKFACTOR: gemm:v2:16,16,16,1,1,1,16,4,1,0,0,-1,-1,-1,0,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-NUMSTAGES
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,0,-1,-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v2:16,16,16,1,1,1,16,1,2,0,0,-1,-1,-1,0,-1,-1
// CHECK-EXHAUSTIVE-NUMSTAGES: gemm:v2:16,16,16,1,1,1,16,1,3,0,0,-1,-1,-1,0,-1,-1

// Exhaustive tuning sweeps the use-buffer-ops knob (off=0 then on=1) for plain
// GEMM kernels only; the two variants are emitted back-to-back per config.
// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-BUFFEROPS
// CHECK-EXHAUSTIVE-BUFFEROPS: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,0,-1,-1
// CHECK-EXHAUSTIVE-BUFFEROPS-NEXT: gemm:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,1,-1,-1

// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN
// CHECK-EXHAUSTIVE-ATTN: attn:v2:16,16,16,1,1,1,16,1,1,0,0,-1,-1,-1,-1,-1,-1

// The WMMA gemm tuning space hardcodes `kPackList = {1}` in
// `getAccelRangeGemm`, so kpack=2 must never appear in the emitted
// configs -- including on RDNA archs like gfx1100 where `getMaxKpack`
// would otherwise return 2.
// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-GEMM-KPACK \
// RUN:       --implicit-check-not="gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-WMMA-GEMM-KPACK: gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,1,}}

// The WMMA attention tuning space applies the same hardcoding in
// `getAccelRangeGemmGemm`.
// RUN: rocmlir-gen --arch gfx1100 --operation=attention -t f16 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 1 -num_heads_kv 1 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-WMMA-ATTN-KPACK \
// RUN:       --implicit-check-not="attn:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-WMMA-ATTN-KPACK: attn:v2:{{[0-9]+,[0-9]+,[0-9]+,1,}}

// On gfx950 the MFMA `kPackList` is derived from `getMaxKpack(arch)`, which
// is 1 for CDNA4, so the gemm tuning space never emits kpack=2 either.
// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MFMA-GFX950-KPACK \
// RUN:       --implicit-check-not="gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,2,}}"
// CHECK-MFMA-GFX950-KPACK: gemm:v2:{{[0-9]+,[0-9]+,[0-9]+,1,}}
