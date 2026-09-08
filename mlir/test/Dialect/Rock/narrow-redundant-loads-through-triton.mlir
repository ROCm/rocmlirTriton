// Check that a load narrowed by rock-narrow-redundant-loads stays narrow
// through Triton's layout passes, and that its layout does not spread into the
// epilogue it feeds. Whether the pass fires at all is covered by
// narrow-redundant-loads.mlir; what this test pins is the interaction, because
// a narrowed load is a small tensor with a coalesced layout of its own, and
// tritongpu-remove-layout-conversions used to hand that layout to every value
// downstream of the broadcast -- turning a cheaper load into a full-tile
// shared-memory round trip in the epilogue.
//
// The tuning parameters are fixed rather than tuned for, because the tile
// shape is what decides how much data the narrowed load covers: mPerBlock=128
// makes it a 128x1 read of the per-channel bias.
//
// The architecture is fixed for the same reason. gfx1100 has 32-lane warps, so
// with numWaves=4 that 128x1 read is exactly one element per thread, which is
// where Triton starts treating a load as worth preserving a layout for. On a
// 64-lane architecture the same read falls below that bar and never claims a
// layout in the first place, so the interaction this test is about does not
// arise there.

// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s \
// RUN: | rocmlir-driver -c --arch=gfx1100 --mlir-disable-threading -o /dev/null \
// RUN:   --perf-config=gemm:mPerBlock=128,nPerBlock=64,kPerBlock=32,kpack=1,numCTAs=1,numWaves=4,matrixInstrNonkdim=0,splitKFactor=1,numStages=2,wavesPerEU=0,gridGroupSize=0 \
// RUN:   --mlir-print-ir-after=tritonamdgpu-optimize-epilogue 2>&1 \
// RUN: | FileCheck %s --implicit-check-not='convert_layout {{.*}}tensor<128x64xf32'

// The accumulator layout of the dot, which the epilogue is expected to keep.
// CHECK-DAG: #[[MMA:.+]] = #ttg.amd_wmma<

// The bias is read once per row and broadcast back over the tile, rather than
// read as a full 128x64 tile.
// CHECK: %[[BIAS:.*]] = tt.load %{{.*}} : tensor<128x1x!tt.ptr<f16>, #[[NARROW:[a-z0-9]+]]>
// CHECK: %[[CVT:.*]] = ttg.convert_layout %[[BIAS]] : tensor<128x1xf16, #[[NARROW]]> -> tensor<128x1xf16, #[[MMA]]>
// CHECK: %[[BCAST:.*]] = tt.broadcast %[[CVT]] : tensor<128x1xf16, #[[MMA]]> -> tensor<128x64xf16, #[[MMA]]>

// Only those 128 values are converted: the epilogue stays in the accumulator
// layout all the way into both reductions, so the 128x64 tiles never go through
// shared memory. The implicit-check-not above is what enforces that.
// CHECK: arith.addf %{{.*}}, %[[BCAST]] {{.*}} : tensor<128x64xf16, #[[MMA]]>
// CHECK: "tt.reduce"
// CHECK: }) : (tensor<128x64xf32, #[[MMA]]>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[MMA]]}>>
// CHECK: "tt.reduce"
// CHECK: }) : (tensor<128x64xf32, #[[MMA]]>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #[[MMA]]}>>

module {
  func.func @mlir_reshape_convolution_reshape_broadcast_add_convert_mul_reshape_reduce_sum_reshape_mul_mul_reshape_reduce_sum_reshape(%arg0: !migraphx.shaped<1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1>, %arg1: !migraphx.shaped<128x128x3x3xf16, 1152x9x3x1>, %arg2: !migraphx.shaped<32x4xf16, 4x1>) -> (!migraphx.shaped<1x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<1x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1>) attributes {rock.arch = "gfx1100", rock.enable_splitk_for_tuning, rock.kernel} {
    %0 = migraphx.literal(dense<2.38418579E-7> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.literal(dense<2.38418579E-7> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.reshape %arg0 {dims = [1, 128, 1024, 1024]} : <1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1> -> <1x128x1024x1024xf16, 134217728x1048576x1024x1>
    %3 = migraphx.convolution %2, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x128x1024x1024xf16, 134217728x1048576x1024x1>, <128x128x3x3xf16, 1152x9x3x1> -> <1x128x1024x1024xf16, 134217728x1048576x1024x1>
    %4 = migraphx.reshape %3 {dims = [1, 32, 4, 1024, 1024]} : <1x128x1024x1024xf16, 134217728x1048576x1024x1> -> <1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1>
    %5 = migraphx.broadcast %arg2 {axis = 1 : i64, out_dyn_dims = [], out_lens = [1, 32, 4, 1024, 1024]} : <32x4xf16, 4x1> -> <1x32x4x1024x1024xf16, 0x4x1x0x0>
    %6 = migraphx.add %4, %5 : <1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1>, <1x32x4x1024x1024xf16, 0x4x1x0x0> -> <1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1>
    %7 = migraphx.convert %6 {target_type = 2 : i64} : <1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1> to <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1>
    %8 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 32, 4, 1024, 1024]} : <1xf32, 0> -> <1x32x4x1024x1024xf32, 0x0x0x0x0>
    %9 = migraphx.mul %7, %8 : <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1>, <1x32x4x1024x1024xf32, 0x0x0x0x0> -> <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1>
    %10 = migraphx.reshape %9 {dims = [1, 32, 4194304]} : <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1> -> <1x32x4194304xf32, 134217728x4194304x1>
    %11 = migraphx.reduce_sum %10 {axes = [2]} : <1x32x4194304xf32, 134217728x4194304x1> -> <1x32x1xf32, 32x1x1>
    %12 = migraphx.reshape %11 {dims = [1, 32, 1, 1, 1]} : <1x32x1xf32, 32x1x1> -> <1x32x1x1x1xf32, 32x1x1x1x1>
    %13 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 32, 4, 1024, 1024]} : <1xf32, 0> -> <1x32x4x1024x1024xf32, 0x0x0x0x0>
    %14 = migraphx.mul %7, %7 : <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1>, <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1> -> <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1>
    %15 = migraphx.mul %14, %13 : <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1>, <1x32x4x1024x1024xf32, 0x0x0x0x0> -> <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1>
    %16 = migraphx.reshape %15 {dims = [1, 32, 4194304]} : <1x32x4x1024x1024xf32, 134217728x4194304x1048576x1024x1> -> <1x32x4194304xf32, 134217728x4194304x1>
    %17 = migraphx.reduce_sum %16 {axes = [2]} : <1x32x4194304xf32, 134217728x4194304x1> -> <1x32x1xf32, 32x1x1>
    %18 = migraphx.reshape %17 {dims = [1, 32, 1, 1, 1]} : <1x32x1xf32, 32x1x1> -> <1x32x1x1x1xf32, 32x1x1x1x1>
    return %12, %18, %6 : !migraphx.shaped<1x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<1x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<1x32x4x1024x1024xf16, 134217728x4194304x1048576x1024x1>
  }
}
