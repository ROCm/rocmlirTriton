// gfx1250 block-16 scaled WMMA: every legal operand / scale pair must lower to
// v_wmma_scale16_f32_16x16x128_f8f6f4 with the matching format modifiers.
//
// Each RUN line compiles one pair to a code object, and AMDGCN_ENABLE_DUMP
// prints its ISA. The 64x64x128 tile on 4 waves needs four 16x16x128
// instructions. B is transposed because gfx1250 packs scaled fp4 only along K.
//
// Instruction operands are D, A, B, C, scaleA, scaleB. One instruction covers
// K = 128, so at block 16 each scale operand holds 128 / 16 = 8 one-byte
// scales: a 64-bit register pair. (Block 32 would give a single 32-bit
// register and v_wmma_scale.)
//
// The disassembler omits format modifiers at their defaults (FP8 data, E8M0
// scales), so each pattern lists the full set of non-default formats and ends
// at the line end to reject any others. MATRIX_SCALE_ROW1 selects which half
// of a scale register an instruction reads; it varies with the tiling and is
// optional in every pattern.
//
// Comments name each case "Case T<n>" rather than "T<n>:", because FileCheck
// reads any "<prefix>:" in this file as a check directive.
//
// Triton swaps the operands: the instruction's A is the GEMM's B and the other
// way round. So a format on the GEMM's B side shows up as matrix_a_* (see T3
// and T4).

// Case T1. f8E4M3FN x f8E4M3FN, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E4M3FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T1

// Case T2. f8E5M2 x f8E5M2, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E5M2 -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T2

// Case T3. f8E4M3FN x f8E5M2, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E4M3FN_f8E5M2 -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T3

// Case T4. f8E5M2 x f8E4M3FN, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E5M2_f8E4M3FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T4

// Case T5. f4E2M1FN x f4E2M1FN, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f4E2M1FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T5

// Case T6. f8E4M3FN x f4E2M1FN, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E4M3FN_f4E2M1FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T6

// Case T7. f4E2M1FN x f8E4M3FN, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f4E2M1FN_f8E4M3FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T7

// Case T8. f8E5M2 x f4E2M1FN, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E5M2_f4E2M1FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T8

// Case T9. f4E2M1FN x f8E5M2, E8M0 x E8M0 scales.
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f4E2M1FN_f8E5M2 -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,T9

// CHECK: .amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
// CHECK-NOT: v_wmma

// Case T1. FP8 data and E8M0 scales are both defaults, so no format modifiers.
// T1-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Case T2. BF8 data on both sides; E8M0 scales stay default.
// T2-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_BF8{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Case T3. The GEMM's B is BF8. After the operand swap it is the
// instruction's A, so only matrix_a_fmt is set.
// T3-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_BF8{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Case T4. The mirror of case T3. The GEMM's A is BF8, which is the
// instruction's B.
// T4-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_b_fmt:MATRIX_FMT_BF8{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Case T5. FP4 data on both sides; E8M0 scales stay default.
// T5-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Mixed FP8 / FP4 (cases T6 to T9). The FP4 side uses half the data registers
// of the FP8 side (16x128 FP4 is 8 VGPRs, FP8 is 16), and the operand swap
// decides which matrix_*_fmt carries FP4.

// Case T6. The GEMM's B is FP4, so the instruction's A is FP4.
// T6-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_FP4{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Case T7. The GEMM's A is FP4, so the instruction's B is FP4.
// T7-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_b_fmt:MATRIX_FMT_FP4{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Case T8. GEMM B (FP4) is the instruction's A, and GEMM A (BF8) its B.
// T8-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_BF8{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// Case T9. GEMM B (BF8) is the instruction's A, and GEMM A (FP4) its B.
// T9-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP4{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// CHECK-NOT: v_wmma
// CHECK: s_endpgm
