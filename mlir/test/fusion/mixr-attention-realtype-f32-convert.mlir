// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -arch %arch | rocmlir-driver -kernel-pipeline gpu -arch %arch --mlir-print-ir-after=rock-gridwise-attn-to-blockwise -o /dev/null 2>&1 -debug-only=rock-gridwise-attn-to-blockwise | FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -arch %arch | rocmlir-driver -kernel-pipeline gpu,triton -arch %arch --mlir-print-ir-after=tritongpu-coalesce -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION


// CHECK: elemTypeQLoad: f16
// CHECK: elemTypeKLoad: f32
// CHECK: elemTypeVLoad: f16
// VECTORIZATION-DAG: #[[COALESCED:.*]] = #ttg.blocked<{sizePerThread = [1, 8]
// VECTORIZATION: tt.load {{.*}}#[[COALESCED]]>
module {
  func.func @mlir_attention_f32(%arg0: !migraphx.shaped<4096x4096xf32, 4096x1>, %arg1: !migraphx.shaped<4096x4096xf32, 4096x1>, %arg2: !migraphx.shaped<4096x4096xf16, 4096x1>, %arg3: !migraphx.shaped<4096x4096xf16, 4096x1>) -> !migraphx.shaped<4096x4096xf16, 4096x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.add %arg0, %arg1 : <4096x4096xf32, 4096x1>, <4096x4096xf32, 4096x1> -> <4096x4096xf32, 4096x1>
    %1 = migraphx.convert %0 {target_type = 0 : i64} : <4096x4096xf32, 4096x1> to <4096x4096xf16, 4096x1>
    %2 = migraphx.dot %arg2, %1 : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %3 = migraphx.softmax %2 {axis = 1 : i64} : <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %4 = migraphx.dot %3, %arg3 {perf_config = "attn:v1:128,128,128,1,1,8,0,1,1,0,0"} : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    return %4 : !migraphx.shaped<4096x4096xf16, 4096x1>
  }
}
