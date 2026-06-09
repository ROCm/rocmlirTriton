// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GLOBAL_ATOMIC_ADD_F16
// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t bf16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GLOBAL_ATOMIC_CMPSWAP
// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t f32 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GLOBAL_ATOMIC_ADD_F32

// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_F16
// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t bf16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GLOBAL_ATOMIC_ADD_BF16
// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t f32 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_F32

// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_F16
// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t bf16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_BF16
// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t f32 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_F32

// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GLOBAL_ATOMIC_CMPSWAP
// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t bf16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GLOBAL_ATOMIC_CMPSWAP
// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t f32 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GLOBAL_ATOMIC_ADD_F32

// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_F16
// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t bf16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_BF16
// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t f32 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD_F32

// BUFFER_ATOMIC_ADD_F16: buffer_atomic_pk_add_f16
// BUFFER_ATOMIC_ADD_BF16: buffer_atomic_pk_add_bf16
// BUFFER_ATOMIC_ADD_F32: buffer_atomic_add_f32
// GLOBAL_ATOMIC_CMPSWAP: global_atomic_cmpswap
// GLOBAL_ATOMIC_ADD_F16: global_atomic_pk_add_f16
// GLOBAL_ATOMIC_ADD_F32: global_atomic_add_f32
// GLOBAL_ATOMIC_ADD_BF16: global_atomic_pk_add_bf16
