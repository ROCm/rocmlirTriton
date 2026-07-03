// RUN: rocmlir-opt -rock-split-cross-tile-fusion %s | FileCheck %s
// RUN: rocmlir-opt -rock-split-cross-tile-fusion -rock-elementwise-to-gridwise %s | FileCheck %s --check-prefix=PWROOT

// AIROCMLIR-709 Pattern 1: a fusion that adds two *different* slices of the same
// gemm output is a cross-tile dependency that cannot be expressed as a single
// fused gemm-writeback kernel (and is silently miscompiled by
// rock-regularize-output). rock-split-cross-tile-fusion detects the pattern and
// outlines it into a gemm kernel + an elementwise kernel joined by an
// intermediate buffer of the gemm result's (flattened) shape.

#map = affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 16 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 16 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1 + 16, d2)>
#map4 = affine_map<(d0) -> (d0 floordiv 256, (d0 mod 256) floordiv 16, d0 mod 16)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 16, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 16, 16] -> [1024]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{4, 32, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 32, 16] -> [2048]>
#transform_map2 = #rock.transform_map<#map2 by [<Slice{0, 4, 0, 16, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [4, 16, 16] -> [4, 32, 16]>
#transform_map3 = #rock.transform_map<#map3 by [<Slice{0, 4, 16, 32, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [4, 16, 16] -> [4, 32, 16]>
#transform_map4 = #rock.transform_map<#map4 by [<Merge{4, 16, 16} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [1024] -> [4, 16, 16]>

// The fused kernel is replaced by two kernels.
// CHECK-NOT: func.func @mlir_double_slice_add(
//
// Kernel A: the gemm whose result is flattened and stored to the intermediate
// 1-D buffer (tensor<2048xf16> = 4*32*16).
// CHECK-LABEL: func.func @mlir_double_slice_add_gemm
// CHECK-SAME:  (%{{.*}}: tensor<2048xf16>, %{{.*}}: tensor<1024xf16>, %[[BUF:.*]]: tensor<2048xf16>) -> tensor<2048xf16>
// CHECK:       %[[G:.*]] = rock.gemm
// CHECK:       %[[FLAT:.*]] = rock.transform %[[G]]
// CHECK-SAME:    to tensor<2048xf16>
// CHECK:       rock.store %[[FLAT]] to %[[BUF]]
//
// Kernel B: a native pure-elementwise kernel (no rock.store / output arg) that
// expands the 1-D buffer back to the gemm shape and adds the two slices.
// CHECK-LABEL: func.func @mlir_double_slice_add_elementwise
// CHECK-SAME:  (%[[IN:.*]]: tensor<2048xf16>) -> tensor<1024xf16>
// CHECK:       %[[EXP:.*]] = rock.transform %[[IN]]
// CHECK-SAME:    to tensor<4x32x16xf16>
// CHECK:       %[[S0:.*]] = rock.transform %[[EXP]]
// CHECK:       %[[S1:.*]] = rock.transform %[[EXP]]
// CHECK:       %[[ADD:.*]] = arith.addf %[[S0]], %[[S1]]
// CHECK:       return
// CHECK-NOT:   rock.store

// After the split, the elementwise kernel flows through the pure-elementwise
// path and gets a gridwise_elementwise root (the cross-tile dependency is gone
// because both slices now read a fully-materialized buffer).
// PWROOT-LABEL: func.func @mlir_double_slice_add_elementwise
// PWROOT:       rock.gridwise_elementwise
// PWROOT:       rock.store

module attributes {rock.arch = "gfx942"} {
  func.func @mlir_double_slice_add(%arg0: tensor<2048xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<1024xf16>) -> tensor<1024xf16> attributes {rock.kernel = "mixr"} {
    %0 = rock.transform %arg1 by #transform_map : tensor<1024xf16> to tensor<4x16x16xf16>
    %1 = rock.transform %arg0 by #transform_map1 : tensor<2048xf16> to tensor<4x32x16xf16>
    %2 = rock.gemm %1 * %0 : tensor<4x32x16xf16> * tensor<4x16x16xf16> -> tensor<4x32x16xf16>
    %3 = rock.transform %2 by #transform_map2 : tensor<4x32x16xf16> to tensor<4x16x16xf16>
    %4 = rock.transform %2 by #transform_map3 : tensor<4x32x16xf16> to tensor<4x16x16xf16>
    %5 = arith.addf %3, %4 : tensor<4x16x16xf16>
    %6 = rock.transform %5 by #transform_map4 : tensor<4x16x16xf16> to tensor<1024xf16>
    %7 = rock.store %6 to %arg2 by set : tensor<1024xf16> -> tensor<1024xf16> to tensor<1024xf16>
    return %7 : tensor<1024xf16>
  }
}
