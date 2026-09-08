// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -arch %arch | rocmlir-driver -kernel-pipeline gpu -arch %arch --mlir-print-ir-after=rock-gridwise-attn-to-blockwise -o /dev/null 2>&1 -debug-only=rock-gridwise-attn-to-blockwise | FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/gfx942/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -arch gfx942 | rocmlir-driver -kernel-pipeline gpu,triton -arch gfx942 --mlir-print-ir-after=tritongpu-coalesce -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// CHECK: elemTypeQLoad: f16
// CHECK: elemTypeKLoad: i4
// CHECK: elemTypeVLoad: f16
// VECTORIZATION-DAG: #[[Q_LAYOUT:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 8], order = [0, 1]}>
// VECTORIZATION-DAG: #[[V_LAYOUT:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [2, 8], order = [1, 0]}>
// VECTORIZATION: tt.load {{.*}} : tensor<128x1x!tt.ptr<f16>, #[[Q_LAYOUT]]>
// VECTORIZATION: tt.load {{.*}} : tensor<64x4096x!tt.ptr<f16>, #[[V_LAYOUT]]>
module {
  func.func @mlir_attention_int4(%arg0: !migraphx.shaped<4096x4096xf16, 8192x1>, %arg1: !migraphx.shaped<4096xf16, 1>, %arg2: !migraphx.shaped<4096xf16, 1>, %arg3: !migraphx.shaped<4096x2048xui8, 2048x1>, %arg4: !migraphx.shaped<4096x4096xf16, 4096x1>) -> !migraphx.shaped<4096x4096xf16, 4096x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.unpack %arg3 {axis = 1 : i64} : <4096x2048xui8, 2048x1> -> <4096x4096xi8, 4096x1>
    %1 = migraphx.broadcast %arg1 {axis = 0 : i64, out_lens = [4096, 4096]} : <4096xf16, 1> -> <4096x4096xf16, 0x1>
    %2 = migraphx.broadcast %arg2 {axis = 0 : i64, out_lens = [4096, 4096]} : <4096xf16, 1> -> <4096x4096xf16, 0x1>
    %3 = migraphx.reshape %1 {dims = [4096, 4096]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
    %4 = migraphx.reshape %2 {dims = [4096, 4096]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
    %5 = migraphx.dequantizelinear %0, %3, %4 : <4096x4096xi8, 4096x1>, <4096x4096xf16, 16536x2>, !migraphx.shaped<4096x4096xf16, 16536x2> -> <4096x4096xf16, 4096x1>
    %6 = migraphx.dot %2, %5 : <4096x4096xf16, 0x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %7 = migraphx.softmax %6 {axis = 1 : i64} : <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %8 = migraphx.dot %7, %arg4 {perf_config = "attn:v1:128,64,128,1,1,16,0,1,1,0,0"} : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    return %8 : !migraphx.shaped<4096x4096xf16, 4096x1>
  }
}
