// gfx1250 scaled WMMA. Block 32 selects v_wmma_scale and block 16 selects
// v_wmma_scale16. Both forms of the 16x16x128 f8f6f4 instruction accept mixed
// f8/f4; f8E4M3FN scales are legal on fp4 x fp4 and on the fp4 side of a
// mixed pair.

// gfx1250 cannot pack a scaled fp4 operand along M/N, so B is transposed and
// both fp4 operands pack along K. -p fills host buffers through a layout that
// drops that K vectorization.

// Each kernel is compiled to a code object and AMDGCN_ENABLE_DUMP prints its
// ISA. The 64x64 tile on 4 waves needs four 16x16x128 instructions. The
// disassembler omits format modifiers at their defaults (FP8 data, E8M0
// scales), so each pattern lists the full set of non-default formats and ends
// at the line end to reject any others.

// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E4M3FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SCALE32
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f4E2M1FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   -scale_a_dtype f8E4M3FN -scale_b_dtype f8E4M3FN \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,SCALE16
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f8E4M3FN_f4E2M1FN -m 64 -n 64 -k 128 \
// RUN:   -out_dtype f32 --scaledGemm -quantBlockSize 16 -transB=true \
// RUN:   -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E4M3FN \
// RUN:   --perf_config "gemm:v1:64,64,128,1,1,4,16,1,2,0,0" \
// RUN:   | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver --kernel-pipeline=full -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,MIXED16

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
// CHECK-NOT: v_wmma

// fp8 x fp8 with E8M0 scales: no format modifiers, one 32-bit register per scale.
// SCALE32-COUNT-4: v_wmma_scale_f32_16x16x128_f8f6f4 {{.*}}, 0, v{{[0-9]+}}, v{{[0-9]+}}{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*$}}

// fp4 x fp4 with E4M3 scales on both sides, a 64-bit register pair per scale.
// SCALE16-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*}} matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3{{$}}

// Triton swaps the operands, so the fp4 side with its E4M3 scales is the
// instruction's A and the fp8 side keeps the default FP8 data and E8M0 scales.
// MIXED16-COUNT-4: v_wmma_scale16_f32_16x16x128_f8f6f4 {{.*}}, 0, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] matrix_a_fmt:MATRIX_FMT_FP4{{( matrix_[ab]_scale:MATRIX_SCALE_ROW1)*}} matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3{{$}}

// CHECK-NOT: v_wmma
// CHECK: s_endpgm
