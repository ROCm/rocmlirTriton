// RUN: rocmlir-opt -rock-elementwise-to-gridwise %s | FileCheck %s --check-prefix=ROOT
// RUN: rocmlir-opt -rock-elementwise-to-gridwise -rock-gridwise-elementwise-to-blockwise %s | FileCheck %s --check-prefix=BLOCK

// A pure elementwise kernel (slice + slice + add, no gemm/conv/attention) has no
// FusionRoot today, so it cannot be lowered. rock-elementwise-to-gridwise wraps
// the primary elementwise input in a rock.gridwise_elementwise root, inserts the
// output argument + rock.store, and sets the launch parameters;
// rock-gridwise-elementwise-to-blockwise then lowers it into the same
// load_marker + store_marker output tile a gemm tail uses.

#map = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 16 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1 + 16, d2)>
#map3 = affine_map<(d0) -> (d0 floordiv 256, (d0 mod 256) floordiv 16, d0 mod 16)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 32, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 32, 16] -> [2048]>
#transform_map1 = #rock.transform_map<#map1 by [<Slice{0, 4, 0, 16, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [4, 16, 16] -> [4, 32, 16]>
#transform_map2 = #rock.transform_map<#map2 by [<Slice{0, 4, 16, 32, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [4, 16, 16] -> [4, 32, 16]>
#transform_map3 = #rock.transform_map<#map3 by [<Merge{4, 16, 16} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [1024] -> [4, 16, 16]>

// ROOT-LABEL: func.func @mlir_pw_slice_add
// ROOT-SAME:  (%arg0: tensor<2048xf16>, %arg1: tensor<1024xf16>) -> tensor<1024xf16>
// ROOT-SAME:  rock.block_size = 64 : i32
// ROOT-SAME:  rock.grid_size = 4 : i32
// ROOT:       %[[ROOT:.*]] = rock.gridwise_elementwise(%{{.*}}) {mPerBlock = 16 : i64, nPerBlock = 16 : i64} : tensor<4x16x16xf16> -> tensor<4x16x16xf16>
// ROOT:       %[[ADD:.*]] = arith.addf %[[ROOT]], %{{.*}} : tensor<4x16x16xf16>
// ROOT:       %[[STORE:.*]] = rock.store {{.*}} to %arg1 by set
// ROOT:       return %[[STORE]]

// BLOCK-LABEL: func.func @mlir_pw_slice_add
// BLOCK-NOT:   rock.gridwise_elementwise
// BLOCK:       %[[TILE:.*]] = rock.load_marker %{{.*}} views {{.*}} : tensor<4x16x16xf16> -> tensor<16x16xf16>
// BLOCK:       %[[FULL:.*]] = rock.store_marker %[[TILE]] views {{.*}} : tensor<16x16xf16> -> tensor<4x16x16xf16>
// BLOCK:       arith.addf %[[FULL]], %{{.*}} : tensor<4x16x16xf16>
module {
  func.func @mlir_pw_slice_add(%arg0: tensor<2048xf16>) -> tensor<1024xf16> attributes {rock.arch = "gfx908:sramecc+:xnack-", rock.kernel = "mixr"} {
    %0 = rock.transform %arg0 by #transform_map : tensor<2048xf16> to tensor<4x32x16xf16>
    %1 = rock.transform %0 by #transform_map1 : tensor<4x32x16xf16> to tensor<4x16x16xf16>
    %2 = rock.transform %0 by #transform_map2 : tensor<4x32x16xf16> to tensor<4x16x16xf16>
    %3 = arith.addf %1, %2 : tensor<4x16x16xf16>
    %4 = rock.transform %3 by #transform_map3 : tensor<4x16x16xf16> to tensor<1024xf16>
    return %4 : tensor<1024xf16>
  }
}
