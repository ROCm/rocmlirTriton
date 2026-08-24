// TODO(rocmlirTriton): Route RDNA4 to the plain cvt instructions as well, and
// check it here, once LCOMPILER-2609 is fixed in LLVM.

// Check that OCP fp8/bf8 casts lower to hardware conversion instructions rather
// than to the software fallback. gfx1170 uses the plain cvt instructions;
// gfx950 and gfx1250 have those too, but are routed to the wider scaled ones
// instead.
//
// Both kernels use the quantized-GEMM shape MIGraphX emits: quantize both dot
// operands, run the dot in fp8, add an fp8 residual, requantize the output. The
// residual is what forces an elementwise upcast, since quant_dot consumes fp8
// natively. Keeping the dequantize and the output quantize on opposite sides of
// the add stops adjacent casts from folding away.
//
// Both functions are compiled, so each RUN line covers fp8 and bf8.

// Check gfx1170 lowering at the LLVM boundary.
// RUN: rocmlir-gen --clone-harness -arch gfx1170 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx1170 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=gfx1170 -kernel-pipeline=gpu,triton \
// RUN: | FileCheck %s --check-prefix=LLIR1170

// LLIR1170-DAG: rocdl.cvt.pk.f32.fp8
// LLIR1170-DAG: rocdl.cvt.pk.fp8.f32
// LLIR1170-DAG: rocdl.cvt.pk.f32.bf8
// LLIR1170-DAG: rocdl.cvt.pk.bf8.f32

// On gfx942 these mnemonics use FNUZ encoding, so OCP casts must remain on the
// software path. llvm.func prevents the negative check from passing vacuously.
// RUN: rocmlir-gen --clone-harness -arch gfx942 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx942 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=gfx942 -kernel-pipeline=gpu,triton \
// RUN: | FileCheck %s --check-prefix=LLIR942 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.fp8.f32 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.bf8.f32 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.f32.fp8 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.f32.bf8

// LLIR942: llvm.func

// Check that gfx1170 reaches the expected ISA instructions.
// RUN: rocmlir-gen --clone-harness -arch gfx1170 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx1170 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1170 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1170 2>&1
// RUN: FileCheck %s --check-prefix=ASM1170 < %t.gfx1170

// ASM1170-DAG: v_cvt_pk_f32_fp8
// ASM1170-DAG: v_cvt_pk_fp8_f32
// ASM1170-DAG: v_cvt_pk_f32_bf8
// ASM1170-DAG: v_cvt_pk_bf8_f32

// RDNA4 has the same OCP instructions, but must stay on the software path:
// LCOMPILER-2609 makes the packed upcast unassemblable in the real-true16 mode
// gfx12 defaults to. llvm.func prevents the negative check from passing
// vacuously.
// RUN: rocmlir-gen --clone-harness -arch gfx1200 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx1200 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=gfx1200 -kernel-pipeline=gpu,triton \
// RUN: | FileCheck %s --check-prefix=LLIR1200 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.fp8.f32 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.bf8.f32 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.f32.fp8 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.f32.bf8

// LLIR1200: llvm.func

// gfx950 has the plain cvt instructions as well, but CDNA4 is routed to the
// scaled cvt instructions, which convert the same OCP formats once the scale
// operand is pinned to 1.0. Unlike RDNA4 these assemble, so also check that the
// intrinsics survive to the ISA.
// RUN: rocmlir-gen --clone-harness -arch gfx950 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx950 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=gfx950 -kernel-pipeline=gpu,triton \
// RUN: | FileCheck %s --check-prefix=LLIR950 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.fp8.f32 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.bf8.f32

// LLIR950-DAG: rocdl.cvt.scalef32.pk.f32.fp8
// LLIR950-DAG: rocdl.cvt.scalef32.pk.fp8.f32
// LLIR950-DAG: rocdl.cvt.scalef32.pk.f32.bf8
// LLIR950-DAG: rocdl.cvt.scalef32.pk.bf8.f32

// RUN: rocmlir-gen --clone-harness -arch gfx950 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx950 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx950 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx950 2>&1
// RUN: FileCheck %s --check-prefix=ASM950 < %t.gfx950

// ASM950-DAG: v_cvt_scalef32_pk_f32_fp8
// ASM950-DAG: v_cvt_scalef32_pk_fp8_f32
// ASM950-DAG: v_cvt_scalef32_pk_f32_bf8
// ASM950-DAG: v_cvt_scalef32_pk_bf8_f32

// gfx1250 is routed to the packed-8 scaled cvt instructions, which handle four
// times as many elements per instruction as the packed-2 forms above.
// RUN: rocmlir-gen --clone-harness -arch gfx1250 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx1250 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton \
// RUN: | FileCheck %s --check-prefix=LLIR1250 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.fp8.f32 \
// RUN:   --implicit-check-not=rocdl.cvt.pk.bf8.f32

// LLIR1250-DAG: rocdl.cvt.scale.pk8.f32.fp8
// LLIR1250-DAG: rocdl.cvt.scalef32.pk8.fp8.f32
// LLIR1250-DAG: rocdl.cvt.scale.pk8.f32.bf8
// LLIR1250-DAG: rocdl.cvt.scalef32.pk8.bf8.f32

// RUN: rocmlir-gen --clone-harness -arch gfx1250 -fut mlir_fp8_cvt %s \
// RUN: | rocmlir-driver -arch=gfx1250 -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN: | env AMDGCN_ENABLE_DUMP=1 rocmlir-driver -arch=gfx1250 -kernel-pipeline=gpu,triton,binary -o /dev/null > %t.gfx1250 2>&1
// RUN: FileCheck %s --check-prefix=ASM1250 < %t.gfx1250

// ASM1250-DAG: v_cvt_scale_pk8_f32_fp8
// ASM1250-DAG: v_cvt_scalef32_pk8_fp8_f32
// ASM1250-DAG: v_cvt_scale_pk8_f32_bf8
// ASM1250-DAG: v_cvt_scalef32_pk8_bf8_f32

module {
  // Quantized GEMM over f8E4M3FN with an fp8 residual added in the epilogue.
  func.func @mlir_fp8_cvt(%a: !migraphx.shaped<1x64x64xf32, 4096x64x1>,
                          %b: !migraphx.shaped<1x64x64xf32, 4096x64x1>,
                          %residual: !migraphx.shaped<1x64x64xf8E4M3FN, 4096x64x1>)
      -> !migraphx.shaped<1x64x64xf8E4M3FN, 4096x64x1> attributes {rock.kernel} {
    %qs = migraphx.literal (dense<5.000000e-01> : tensor<1xf32>) : <1xf32, 0>
    %ds = migraphx.literal (dense<2.500000e-01> : tensor<1xf32>) : <1xf32, 0>
    %qsb = migraphx.multibroadcast %qs {out_dyn_dims = [], out_lens = [1, 64, 64]} : <1xf32, 0> -> <1x64x64xf32, 0x0x0>
    %dsb = migraphx.multibroadcast %ds {out_dyn_dims = [], out_lens = [1, 64, 64]} : <1xf32, 0> -> <1x64x64xf32, 0x0x0>
    %aq = migraphx.quantizelinear %a, %qsb : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf8E4M3FN, 4096x64x1>
    %bq = migraphx.quantizelinear %b, %qsb : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf8E4M3FN, 4096x64x1>
    %d = migraphx.quant_dot %aq, %bq : <1x64x64xf8E4M3FN, 4096x64x1>, <1x64x64xf8E4M3FN, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %up = migraphx.dequantizelinear %residual, %dsb : <1x64x64xf8E4M3FN, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf32, 4096x64x1>
    %sum = migraphx.add %d, %up : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %out = migraphx.quantizelinear %sum, %qsb : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf8E4M3FN, 4096x64x1>
    return %out : !migraphx.shaped<1x64x64xf8E4M3FN, 4096x64x1>
  }

  // Same pipeline over f8E5M2.
  func.func @mlir_bf8_cvt(%a: !migraphx.shaped<1x64x64xf32, 4096x64x1>,
                          %b: !migraphx.shaped<1x64x64xf32, 4096x64x1>,
                          %residual: !migraphx.shaped<1x64x64xf8E5M2, 4096x64x1>)
      -> !migraphx.shaped<1x64x64xf8E5M2, 4096x64x1> attributes {rock.kernel} {
    %qs = migraphx.literal (dense<5.000000e-01> : tensor<1xf32>) : <1xf32, 0>
    %ds = migraphx.literal (dense<2.500000e-01> : tensor<1xf32>) : <1xf32, 0>
    %qsb = migraphx.multibroadcast %qs {out_dyn_dims = [], out_lens = [1, 64, 64]} : <1xf32, 0> -> <1x64x64xf32, 0x0x0>
    %dsb = migraphx.multibroadcast %ds {out_dyn_dims = [], out_lens = [1, 64, 64]} : <1xf32, 0> -> <1x64x64xf32, 0x0x0>
    %aq = migraphx.quantizelinear %a, %qsb : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf8E5M2, 4096x64x1>
    %bq = migraphx.quantizelinear %b, %qsb : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf8E5M2, 4096x64x1>
    %d = migraphx.quant_dot %aq, %bq : <1x64x64xf8E5M2, 4096x64x1>, <1x64x64xf8E5M2, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %up = migraphx.dequantizelinear %residual, %dsb : <1x64x64xf8E5M2, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf32, 4096x64x1>
    %sum = migraphx.add %d, %up : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 4096x64x1> -> <1x64x64xf32, 4096x64x1>
    %out = migraphx.quantizelinear %sum, %qsb : <1x64x64xf32, 4096x64x1>, <1x64x64xf32, 0x0x0> -> <1x64x64xf8E5M2, 4096x64x1>
    return %out : !migraphx.shaped<1x64x64xf8E5M2, 4096x64x1>
  }
}
