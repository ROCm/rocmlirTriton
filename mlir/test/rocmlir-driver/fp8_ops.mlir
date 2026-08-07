// RUN: rocmlir-gen --arch gfx942 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=MFMA
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=MFMA
// RUN: rocmlir-gen --arch gfx1170 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=WMMA
// COM: This runs the kernel pipeline so that we still get a good test with the
// COM: host pipeline off as in the static library build, using the fact that
// COM: the fp8 expander isn't limited to GPU code.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t fp8 -p -pv | rocmlir-driver -c | FileCheck %s --check-prefix=HOST
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8 -p -pv | rocmlir-driver -c | FileCheck %s --check-prefix=HOST_GFX950
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t fp8 -p -pv | rocmlir-driver -c | FileCheck %s --check-prefix=GFX11

// MFMA: rocdl.mfma
// MFMA-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// WMMA: llvm.call_intrinsic "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8"
// WMMA-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FN
// GFX11: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// HOST: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// HOST_GFX950: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FN
