// GEMM: f16 output writeback truncates the f32 accumulator to f16.
// RUN: rocmlir-gen --arch gfx90a  --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX90A
// RUN: rocmlir-gen --arch gfx942  --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX942
// RUN: rocmlir-gen --arch gfx950  --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX950
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX1100
// RUN: rocmlir-gen --arch gfx1200 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX1200
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX1250

// Attention: the f16 output is likewise produced by truncating an f32 result.
// RUN: rocmlir-gen --arch gfx942  --operation attention -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX942
// RUN: rocmlir-gen --arch gfx950  --operation attention -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX950
// RUN: rocmlir-gen --arch gfx1100 --operation attention -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX1100
// RUN: rocmlir-gen --arch gfx1200 --operation attention -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX1200
// RUN: rocmlir-gen --arch gfx1250 --operation attention -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX1250

// A packed f16 GEMM/attention accumulates in f32 and truncates the result to
// f16 for the output. Only targets with the `cvt-pk-f16-f32-inst` subtarget
// feature (gfx950/CDNA4 and gfx1250) select the packed `v_cvt_pk_f16_f32`
// convert; every other target truncates one element at a time with the scalar
// `v_cvt_f16_f32` form.

// GFX90A-NOT:{{.*}}v_cvt_pk_f16_f32
// GFX90A:{{.*}}v_cvt_f16_f32
// GFX90A-NOT:{{.*}}v_cvt_pk_f16_f32
// GFX942-NOT:{{.*}}v_cvt_pk_f16_f32
// GFX942:{{.*}}v_cvt_f16_f32
// GFX942-NOT:{{.*}}v_cvt_pk_f16_f32
// GFX1100-NOT:{{.*}}v_cvt_pk_f16_f32
// GFX1100:{{.*}}v_cvt_f16_f32
// GFX1100-NOT:{{.*}}v_cvt_pk_f16_f32
// GFX1200-NOT:{{.*}}v_cvt_pk_f16_f32
// GFX1200:{{.*}}v_cvt_f16_f32
// GFX1200-NOT:{{.*}}v_cvt_pk_f16_f32

// GFX950:{{.*}}v_cvt_pk_f16_f32
// GFX1250:{{.*}}v_cvt_pk_f16_f32
