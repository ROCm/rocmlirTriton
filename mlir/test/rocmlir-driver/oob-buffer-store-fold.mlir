// Triton predicates stores branchlessly: a masked-off lane gets an offset past
// the buffer descriptor's NumRecords so the hardware drops the store. That
// hides the predicate from dead code elimination, and for a depthwise
// convolution whose GEMM M dimension is 1 but padded up to mPerBlock, it keeps
// the whole padded-tile epilogue alive.
//
// InstCombine's fold for llvm.amdgcn.raw.ptr.buffer.store (see
// AMDGPUInstCombineIntrinsic.cpp) erases those stores, after which the matrix
// multiplies and the sigmoid's exp2 that only fed padding rows die with them.
// Check the surviving counts, and that no store keeps an out-of-bounds offset.
//
// perf_config is pinned on the convolution so the counts below do not move when
// the default tuning heuristics change. The counts differ per target because
// the tile shapes and matrix instructions do. Without the fold gfx1100 emits 32
// stores, 16 matrix multiplies and 32 exp2 calls instead of 2, 8 and 2; the
// other targets roughly double.

// RUN: sed s/##TOKEN_ARCH##/gfx1100/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1100 | env LLVM_IR_ENABLE_DUMP=1 rocmlir-driver -kernel-pipeline=gpu,triton,binary -arch gfx1100 -o /dev/null 2>&1 | FileCheck %s --check-prefix=GFX1100 --implicit-check-not='buffer.store{{.*}}i32 -2147483648'
// RUN: sed s/##TOKEN_ARCH##/gfx1200/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1200 | env LLVM_IR_ENABLE_DUMP=1 rocmlir-driver -kernel-pipeline=gpu,triton,binary -arch gfx1200 -o /dev/null 2>&1 | FileCheck %s --check-prefix=GFX1200 --implicit-check-not='buffer.store{{.*}}i32 -2147483648'
// RUN: sed s/##TOKEN_ARCH##/gfx1250/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1250 | env LLVM_IR_ENABLE_DUMP=1 rocmlir-driver -kernel-pipeline=gpu,triton,binary -arch gfx1250 -o /dev/null 2>&1 | FileCheck %s --check-prefix=GFX1250 --implicit-check-not='buffer.store{{.*}}i32 -2147483648'
// RUN: sed s/##TOKEN_ARCH##/gfx942/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx942 | env LLVM_IR_ENABLE_DUMP=1 rocmlir-driver -kernel-pipeline=gpu,triton,binary -arch gfx942 -o /dev/null 2>&1 | FileCheck %s --check-prefix=GFX942 --implicit-check-not='buffer.store{{.*}}i32 -2147483648'
// RUN: sed s/##TOKEN_ARCH##/gfx950/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx950 | env LLVM_IR_ENABLE_DUMP=1 rocmlir-driver -kernel-pipeline=gpu,triton,binary -arch gfx950 -o /dev/null 2>&1 | FileCheck %s --check-prefix=GFX950 --implicit-check-not='buffer.store{{.*}}i32 -2147483648'

// GFX1100-COUNT-8: call{{.*}}@llvm.amdgcn.wmma
// GFX1100-NOT: call{{.*}}@llvm.amdgcn.wmma
// GFX1100-COUNT-2: call{{.*}}@llvm.exp2
// GFX1100-NOT: call{{.*}}@llvm.exp2
// GFX1100-COUNT-2: call void @llvm.amdgcn.raw.ptr.buffer.store
// GFX1100-NOT: call void @llvm.amdgcn.raw.ptr.buffer.store

// GFX1200-COUNT-8: call{{.*}}@llvm.amdgcn.wmma
// GFX1200-NOT: call{{.*}}@llvm.amdgcn.wmma
// GFX1200-COUNT-4: call{{.*}}@llvm.exp2
// GFX1200-NOT: call{{.*}}@llvm.exp2
// GFX1200-COUNT-2: call void @llvm.amdgcn.raw.ptr.buffer.store
// GFX1200-NOT: call void @llvm.amdgcn.raw.ptr.buffer.store

// GFX1250-COUNT-2: call{{.*}}@llvm.amdgcn.wmma
// GFX1250-NOT: call{{.*}}@llvm.amdgcn.wmma
// GFX1250-COUNT-8: call{{.*}}@llvm.exp2
// GFX1250-NOT: call{{.*}}@llvm.exp2
// GFX1250-COUNT-2: call void @llvm.amdgcn.raw.ptr.buffer.store
// GFX1250-NOT: call void @llvm.amdgcn.raw.ptr.buffer.store

// GFX942-COUNT-4: call{{.*}}@llvm.amdgcn.mfma
// GFX942-NOT: call{{.*}}@llvm.amdgcn.mfma
// GFX942-COUNT-2: call{{.*}}@llvm.exp2
// GFX942-NOT: call{{.*}}@llvm.exp2
// GFX942-COUNT-2: call void @llvm.amdgcn.raw.ptr.buffer.store
// GFX942-NOT: call void @llvm.amdgcn.raw.ptr.buffer.store

// GFX950-COUNT-2: call{{.*}}@llvm.amdgcn.mfma
// GFX950-NOT: call{{.*}}@llvm.amdgcn.mfma
// GFX950-COUNT-2: call{{.*}}@llvm.exp2
// GFX950-NOT: call{{.*}}@llvm.exp2
// GFX950-COUNT-2: call void @llvm.amdgcn.raw.ptr.buffer.store
// GFX950-NOT: call void @llvm.amdgcn.raw.ptr.buffer.store

module {
  func.func @mlir_quant_convolution_dequantizelinear_sigmoid_mul_quantizelinear(%arg0: !migraphx.shaped<1x80x80x80xsi8, 512000x6400x80x1>, %arg1: !migraphx.shaped<80x1x3x3xsi8, 9x9x3x1>) -> !migraphx.shaped<1x80x80x80xsi8, 512000x6400x80x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.enable_splitk_for_tuning, rock.kernel} {
    %0 = migraphx.literal(dense<3.29618277E-10> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.literal(dense<1.1920929E-7> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.quant_convolution %arg0, %arg1 {dilation = [1, 1], group = 80 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, perf_config = "gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1", stride = [1, 1]} : <1x80x80x80xsi8, 512000x6400x80x1>, <80x1x3x3xsi8, 9x9x3x1> -> <1x80x80x80xsi32, 512000x6400x80x1>
    %3 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 80, 80, 80]} : <1xf32, 0> -> <1x80x80x80xf32, 0x0x0x0>
    %4 = migraphx.dequantizelinear %2, %3 : <1x80x80x80xsi32, 512000x6400x80x1>, <1x80x80x80xf32, 0x0x0x0> -> <1x80x80x80xf32, 512000x6400x80x1>
    %5 = migraphx.sigmoid %4 : <1x80x80x80xf32, 512000x6400x80x1> -> <1x80x80x80xf32, 512000x6400x80x1>
    %6 = migraphx.mul %4, %5 : <1x80x80x80xf32, 512000x6400x80x1>, <1x80x80x80xf32, 512000x6400x80x1> -> <1x80x80x80xf32, 512000x6400x80x1>
    %7 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 80, 80, 80]} : <1xf32, 0> -> <1x80x80x80xf32, 0x0x0x0>
    %8 = migraphx.quantizelinear %6, %7 {out_type = 5 : i64} : <1x80x80x80xf32, 512000x6400x80x1>, <1x80x80x80xf32, 0x0x0x0> -> <1x80x80x80xsi8, 512000x6400x80x1>
    return %8 : !migraphx.shaped<1x80x80x80xsi8, 512000x6400x80x1>
  }
}
