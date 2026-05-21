// RUN: rocmlir-gen -fut mlir_test --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_test --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut mlir_test --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-driver -arch %arch -c -mlir-print-ir-after=rock-remove-redundant-casts 2>&1 | FileCheck %s --check-prefix=NO-EXTF

// CHECK: [1 1 1]

// NO-EXTF-LABEL: func.func @mlir_test(
// NO-EXTF-NOT: arith.extf
// NO-EXTF: return

module {
  func.func @mlir_test(%arg0: !migraphx.shaped<1x32x4x32xf16, 4096x128x32x1>,
                       %arg1: !migraphx.shaped<1x32x4x32xf16, 4096x128x32x1>,
                       %arg2: !migraphx.shaped<1x32x4x32xf16, 4096x128x32x1>)
      -> !migraphx.shaped<1x4x32x32xf16, 4096x1024x32x1> attributes {rock.kernel} {
    %scale = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 0>
    %q = migraphx.transpose %arg0 {permutation = [0, 2, 1, 3]} : <1x32x4x32xf16, 4096x128x32x1> -> <1x4x32x32xf16, 4096x32x128x1>
    %kt = migraphx.transpose %arg1 {permutation = [0, 2, 3, 1]} : <1x32x4x32xf16, 4096x128x32x1> -> <1x4x32x32xf16, 4096x32x1x128>
    %v = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <1x32x4x32xf16, 4096x128x32x1> -> <1x4x32x32xf16, 4096x32x128x1>
    %qkt_f16 = migraphx.dot %q, %kt : <1x4x32x32xf16, 4096x32x128x1>, <1x4x32x32xf16, 4096x32x1x128> -> <1x4x32x32xf16, 4096x1024x32x1>
    %scale_bcast_f16 = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [1, 4, 32, 32]} : <1xf16, 0> -> <1x4x32x32xf16, 0x0x0x0>
    %qkt_f32 = migraphx.convert %qkt_f16 {target_type = 2 : i64} : <1x4x32x32xf16, 4096x1024x32x1> to <1x4x32x32xf32, 4096x1024x32x1>
    %scale_bcast_f32 = migraphx.convert %scale_bcast_f16 {target_type = 2 : i64} : <1x4x32x32xf16, 0x0x0x0> to <1x4x32x32xf32, 0x0x0x0>
    %scaled = migraphx.mul %qkt_f32, %scale_bcast_f32 : <1x4x32x32xf32, 4096x1024x32x1>, <1x4x32x32xf32, 0x0x0x0> -> <1x4x32x32xf32, 4096x1024x32x1>
    %rmax = migraphx.reduce_max %scaled {axes = [3]} : <1x4x32x32xf32, 4096x1024x32x1> -> <1x4x32x1xf32, 128x32x1x1>
    %rmax_bcast = migraphx.multibroadcast %rmax {out_dyn_dims = [], out_lens = [1, 4, 32, 32]} : <1x4x32x1xf32, 128x32x1x1> -> <1x4x32x32xf32, 128x32x1x0>
    %centered = migraphx.sub %scaled, %rmax_bcast : <1x4x32x32xf32, 4096x1024x32x1>, <1x4x32x32xf32, 128x32x1x0> -> <1x4x32x32xf32, 4096x1024x32x1>
    %exp = migraphx.exp %centered : <1x4x32x32xf32, 4096x1024x32x1> -> <1x4x32x32xf32, 4096x1024x32x1>
    %rsum = migraphx.reduce_sum %exp {axes = [3]} : <1x4x32x32xf32, 4096x1024x32x1> -> <1x4x32x1xf32, 128x32x1x1>
    %rsum_bcast = migraphx.multibroadcast %rsum {out_dyn_dims = [], out_lens = [1, 4, 32, 32]} : <1x4x32x1xf32, 128x32x1x1> -> <1x4x32x32xf32, 128x32x1x0>
    %p_f32 = migraphx.div %exp, %rsum_bcast : <1x4x32x32xf32, 4096x1024x32x1>, <1x4x32x32xf32, 128x32x1x0> -> <1x4x32x32xf32, 4096x1024x32x1>
    %p_f16 = migraphx.convert %p_f32 {target_type = 1 : i64} : <1x4x32x32xf32, 4096x1024x32x1> to <1x4x32x32xf16, 4096x1024x32x1>
    %out = migraphx.dot %p_f16, %v : <1x4x32x32xf16, 4096x1024x32x1>, <1x4x32x32xf16, 4096x32x128x1> -> <1x4x32x32xf16, 4096x1024x32x1>
    return %out : !migraphx.shaped<1x4x32x32xf16, 4096x1024x32x1>
  }
}
