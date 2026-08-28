// Check that rock-narrow-redundant-loads fires on the pattern it was written
// for: a group-quantized int4 GEMM whose weights carry one f16 scale per
// 128-element group along K. Numerical correctness of this same kernel is
// covered by fusion/pr-e2e/mixr-int4-group-quant-dot-add.mlir; this test pins
// which loads the pass does and does not rewrite.
//
// The tuning parameters are fixed rather than tuned for, because whether a K
// tile fits inside one quantization group is what decides the rewrite:
// kPerBlock=32 lands inside a group of 128, while a tile spanning several
// groups reads a different scale per group and is left alone. The tile shapes
// below follow from those parameters alone, so the arch comes from the test
// environment as usual.

// RUN: sed -e 's/##TOKEN_ARCH##/%arch/g' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-driver -c --arch=%arch --perf-config=gemm:mPerBlock=80,nPerBlock=128,kPerBlock=32,kpack=1,numCTAs=1,numWaves=8,matrixInstrNonkdim=0,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0 --mlir-print-ir-after=rock-narrow-redundant-loads --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

// The packed weights and the activations read a distinct element per lane, so
// they keep the full tile.
// CHECK: tt.load %{{.*}} : tensor<16x128x!tt.ptr<i8>>

// The scale is constant across the tile's 32 K positions, so it is read once
// and broadcast back over them.
// CHECK: %[[SCALE:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
// CHECK: tt.broadcast %[[SCALE]] : tensor<1x128xf16> -> tensor<32x128xf16>

module {
  func.func @mlir_unpack_int4_reshape_dequantizelinear_transpose_reshape_unsqueeze_transpose_dot_add(%arg0: !migraphx.shaped<4096x2048xui8, 2048x1>, %arg1: !migraphx.shaped<4096x32x1xf16, 32x1x1>, %arg2: !migraphx.shaped<1x64x77x64xf16, 315392x4928x64x1>, %arg3: !migraphx.shaped<1x77x4096xf16, 315392x4096x1>) -> !migraphx.shaped<1x77x4096xf16, 315392x4096x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
    %0 = migraphx.literal(dense<8> : tensor<1xui8>) : <1xui8, 0>
    %1 = migraphx.unpack %arg0 {axis = 1 : i64} : <4096x2048xui8, 2048x1> -> <4096x4096xui8, 4096x1>
    %2 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [4096, 32, 128]} : <4096x32x1xf16, 32x1x1> -> <4096x32x128xf16, 32x1x0>
    %3 = migraphx.reshape %2 {dims = [4096, 4096]} : <4096x32x128xf16, 32x1x0> -> <4096x4096xf16, 4096x1>
    %4 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [4096, 4096]} : <1xui8, 0> -> <4096x4096xui8, 0x0>
    %5 = migraphx.dequantizelinear %1, %3, %4 : <4096x4096xui8, 4096x1>, <4096x4096xf16, 4096x1>, !migraphx.shaped<4096x4096xui8, 0x0> -> <4096x4096xf16, 4096x1>
    %6 = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <1x64x77x64xf16, 315392x4928x64x1> -> <1x77x64x64xf16, 315392x64x4928x1>
    %7 = migraphx.reshape %6 {dims = [1, 77, 4096]} : <1x77x64x64xf16, 315392x64x4928x1> -> <1x77x4096xf16, 315392x4096x1>
    %8 = migraphx.reshape %5 {dims = [1, 4096, 4096]} : <4096x4096xf16, 4096x1> -> <1x4096x4096xf16, 16777216x4096x1>
    %9 = migraphx.transpose %8 {permutation = [0, 2, 1]} : <1x4096x4096xf16, 16777216x4096x1> -> <1x4096x4096xf16, 16777216x1x4096>
    %10 = migraphx.dot %7, %9 : <1x77x4096xf16, 315392x4096x1>, <1x4096x4096xf16, 16777216x1x4096> -> <1x77x4096xf16, 315392x4096x1>
    %11 = migraphx.add %arg3, %10 : <1x77x4096xf16, 315392x4096x1>, <1x77x4096xf16, 315392x4096x1> -> <1x77x4096xf16, 315392x4096x1>
    return %11 : !migraphx.shaped<1x77x4096xf16, 315392x4096x1>
  }
}
