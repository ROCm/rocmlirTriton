// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// MIGraphX dot + add lowers to rock.gemm with arith.addf on the GEMM output,
// which testFusionLegalitySplitK allows for plain GEMM. Contrast with
// mixr-gemm-gemm-problem-key.mlir, where a pre-second-GEMM add blocks split-K.
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f32 -out_datatype f32 -transA false -transB false -transO false -g 1 -m 7 -n 7 -k 3 -supportsSplitK true
module
{
  func.func private @mlir_dot_add(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
                                %arg2: !migraphx.shaped<1x7x7xf32, 49x7x1>)
                                -> (!migraphx.shaped<1x7x7xf32, 49x7x1>) attributes {rock.kernel, rock.arch = "gfx942", rock.num_cu = 304 : i64} {
    %0 = migraphx.dot %arg0, %arg1: <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.add %0, %arg2 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    return %1 : !migraphx.shaped<1x7x7xf32, 49x7x1>
  }
}
