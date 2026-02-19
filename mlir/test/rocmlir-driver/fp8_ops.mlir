// RUN: rocmlir-gen --arch gfx942 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=MFMA
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=MFMA

// MFMA: rocdl.mfma.f32.16x16x32.fp8.fp8
// MFMA-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ

// TODO(rocmlirTriton): Add meaningul tests to gfx1100 here

// XXX: rocmlir-gen --arch gfx1100 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX11
// XXX: rocmlir-gen --arch gfx1100 --operation gemm -t fp8 -p --force-f8-types=ocp | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX11_OCP

// GFX11: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// GFX11_OCP{LITERAL}: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FN(
