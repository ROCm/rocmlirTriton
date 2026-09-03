// Regression guard for triton-patches/patch-warp-id-readfirstlane.patch.
//
// Triton lowers `ttg.warp_id` either from a hardware wave id (RDNA4 and
// gfx1250, where `supportsWaveId()` holds) or from thread-id arithmetic. On the
// arithmetic path our patch appends a `v_readfirstlane_b32` on every target,
// where upstream restricts it to CDNA3/CDNA4.
//
// Compile-only: cross-compiles for each target, so no GPU is needed.

// RUN: rocmlir-gen --arch gfx1030 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARITH
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARITH
// RUN: rocmlir-gen --arch gfx1101 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARITH
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARITH

// v0 holds workitem.id.x at kernel entry, so this is the warp id being moved
// into an SGPR.
// ARITH-LABEL: rock_gemm:
// ARITH: v_readfirstlane_b32 s{{[0-9]+}}, v0

// gfx1200 reads the wave id out of ttmp8, which is already scalar, so it must
// not grow a readfirstlane.
// RUN: rocmlir-gen --arch gfx1200 --operation gemm -t f16 -p | AMDGCN_ENABLE_DUMP=1 rocmlir-driver -c -o /dev/null 2>&1 | FileCheck %s --check-prefix=HW-WAVE-ID --implicit-check-not=v_readfirstlane_b32

// HW-WAVE-ID-LABEL: rock_gemm:
// HW-WAVE-ID: s_bfe_u32 s{{[0-9]+}}, ttmp8
