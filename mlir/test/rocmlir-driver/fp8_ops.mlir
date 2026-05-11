// RUN: rocmlir-gen --arch gfx942 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=MFMA
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=MFMA
// COM: Run the full pipeline (kernel + host) so that EmulateFp8ExtTrunc, which
// COM: now lives in the host lowering pipeline, emits the fp8 conversion tables
// COM: we want to check below.
// RUN: rocmlir-gen --arch gfx942 --operation gemm -t fp8 -p -pv | rocmlir-driver -kernel-pipeline=full -host-pipeline=backend | FileCheck %s --check-prefix=HOST
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8 -p -pv | rocmlir-driver -kernel-pipeline=full -host-pipeline=backend | FileCheck %s --check-prefix=HOST_GFX950

// MFMA: rocdl.mfma
// MFMA-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ

// TODO(rocmlirTriton): Add meaningul tests to gfx1100 here

// XXX: rocmlir-gen --arch gfx1100 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX11
// XXX: rocmlir-gen --arch gfx1100 --operation gemm -t fp8 -p --force-f8-types=ocp | rocmlir-driver --kernel-pipeline=gpu,triton | FileCheck %s --check-prefix=GFX11_OCP

// GFX11: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// GFX11_OCP{LITERAL}: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FN(
// HOST: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// HOST_GFX950: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FN
