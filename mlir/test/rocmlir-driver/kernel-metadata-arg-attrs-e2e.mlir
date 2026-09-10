// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN: | rocmlir-driver -kernel-pipeline=gpu,triton -arch=gfx1100 \
// RUN: | rocmlir-opt -triton-to-hsaco='arch=gfx1100' > /dev/null 2> %t.log
// RUN: FileCheck %s --input-file=%t.log --allow-empty

// Translating a real kernel to LLVM IR must not emit diagnostics. Front-end
// metadata on kernel parameters has no LLVM IR counterpart, and a dialect that
// does not claim it in an LLVMTranslationDialectInterface gets a warning per
// attribute per argument. Each of those prints the whole kernel into the
// diagnostic, which cost about a fifth of this compile before `rock` and `tt`
// were registered in registerKernelMetadataDialectTranslation().
//
// rocmlir-opt runs the translation here because it installs a diagnostic
// handler; rocmlir-driver installs none, so DiagnosticEngine drops everything
// below an error and the same warnings are paid for but never seen.

// CHECK-NOT: warning:

module {
  func.func @mlir_convolution_add_sigmoid_transpose(%arg0: !migraphx.shaped<1x16x512x512xf32, 4194304x1x8192x16>, %arg1: !migraphx.shaped<1x16x1x1xf32, 16x1x1x1>) -> !migraphx.shaped<1x512x512x1xf32, 262144x512x1x1> attributes {rock.arch = "gfx1100", rock.kernel = "mixr"} {
    %0 = migraphx.literal(dense<0.27341488> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x16x512x512xf32, 4194304x1x8192x16>, <1x16x1x1xf32, 16x1x1x1> -> <1x1x512x512xf32, 262144x1x512x1>
    %2 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 1, 512, 512]} : <1xf32, 0> -> <1x1x512x512xf32, 0x0x0x0>
    %3 = migraphx.add %1, %2 : <1x1x512x512xf32, 262144x1x512x1>, <1x1x512x512xf32, 0x0x0x0> -> <1x1x512x512xf32, 262144x1x512x1>
    %4 = migraphx.sigmoid %3 : <1x1x512x512xf32, 262144x1x512x1> -> <1x1x512x512xf32, 262144x1x512x1>
    %5 = migraphx.transpose %4 {permutation = [0, 2, 3, 1]} : <1x1x512x512xf32, 262144x1x512x1> -> <1x512x512x1xf32, 262144x512x1x1>
    return %5 : !migraphx.shaped<1x512x512x1xf32, 262144x512x1x1>
  }
}
