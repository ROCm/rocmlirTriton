// RUN: rocmlir-driver -c -mlir-print-ir-after=rock-conv-to-gemm -mlir-print-local-scope %s 2>&1 | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 7 + d2) * 7 + d3) * 32 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 9 + d2) * 9 + d3) * 32 + d4)>
#map2 = affine_map<(d0) -> (0, d0 floordiv 288, (d0 mod 288) floordiv 96, (d0 mod 96) floordiv 32, d0 mod 32)>
#transform_map = #rock.transform_map<#map by [<Unmerge{32, 7, 7, 32} ["ni", "0i", "1i", "ci"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [32, 1, 7, 7, 32] -> [50176]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{32, 9, 9, 32} ["no", "0o", "1o", "ko"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [32, 1, 9, 9, 32] -> [82944]>
#transform_map2 = #rock.transform_map<#map2 by [<Merge{32, 3, 3, 32} ["raw"] at [0] -> ["k", "0", "1", "c"] at [1, 2, 3, 4]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [9216] -> [1, 32, 3, 3, 32]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  func.func @rock_conv_bwd_weight_gkyxc_nghwc_nghwk_0(%arg0: tensor<50176xf32>, %arg1: tensor<82944xf32>, %arg2: tensor<9216xf32> {rock.prefill = 0.000000e+00 : f32}) -> tensor<9216xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel, rock.num_cu = 120 : i32} {
    %0 = rock.transform %arg0 by #transform_map : tensor<50176xf32> to tensor<32x1x7x7x32xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<82944xf32> to tensor<32x1x9x9x32xf32>
    %2 = rock.conv_bwd_weight(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "0i", "1i", "ci"], output_layout = ["no", "go", "0o", "1o", "ko"], padding = [2 : index, 2 : index, 2 : index, 2 : index], perf_config = "gemm:v1:32,32,32,1,1,4,0,1,2,0,0", strides = [1 : index, 1 : index]} : tensor<32x1x7x7x32xf32>, tensor<32x1x9x9x32xf32> -> tensor<1x32x3x3x32xf32>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x32x3x3x32xf32> to tensor<9216xf32>
    %4 = rock.store %3 to %arg2 by set : tensor<9216xf32> -> tensor<9216xf32> to tensor<9216xf32>
    return %4 : tensor<9216xf32>
  }
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight_gkyxc_nghwc_nghwk_0
// CHECK-DAG: <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <AddDim{1} ["kBlock"] at [1] -> [] at []>, <PassThrough ["k", "c", "0", "1"] at [2, 5, 3, 4] -> ["k", "c", "0", "1"] at [1, 4, 2, 3]>] bounds = [1, 1, 32, 3, 3, 32] -> [1, 32, 3, 3, 32]>
// CHECK-DAG: <affine_map<(d0, d1, d2) -> (0, 0, d1, d2 floordiv 96, (d2 mod 96) floordiv 32, d2 mod 32)> by [<Merge{1, 1} ["gemmG"] at [0] -> ["g", "kBlock"] at [0, 1]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [2]>, <Merge{3, 3, 32} ["gemmN"] at [2] -> ["0", "1", "c"] at [3, 4, 5]>] bounds = [1, 32, 288] -> [1, 1, 32, 3, 3, 32]>
// CHECK-DAG: <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d1, d2, d3 - 2, d4 - 2, d5)> by [<PassThrough ["gi"] at [2] -> ["gi"] at [1]>, <Unmerge{1, 32} ["n0", "n1"] at [0, 1] -> ["ni"] at [0]>, <PassThrough ["ci"] at [5] -> ["ci"] at [4]>, <Pad{2, 2, 2, 2} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [2, 3]>] bounds = [1, 32, 1, 11, 11, 32] -> [32, 1, 7, 7, 32]>
// CHECK-DAG: <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3 + d4, d5 + d6, d7)> by [<PassThrough ["gi", "n0", "n1", "ci"] at [2, 0, 1, 7] -> ["gi", "n0", "n1", "ci"] at [2, 0, 1, 5]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>] bounds = [1, 32, 1, 3, 9, 3, 9, 32] -> [1, 32, 1, 11, 11, 32]>
// CHECK-DAG: <affine_map<(d0, d1, d2) -> (0, d1 floordiv 81, 0, d2 floordiv 96, (d1 mod 81) floordiv 9, (d2 mod 96) floordiv 32, d1 mod 9, d2 mod 32)> by [<Merge{1, 1} ["gemmG"] at [0] -> ["gi", "n0"] at [2, 0]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["n1", "0o", "1o"] at [1, 4, 6]>, <Merge{3, 3, 32} ["gemmN"] at [2] -> ["0", "1", "ci"] at [3, 5, 7]>] bounds = [1, 2592, 288] -> [1, 32, 1, 3, 9, 3, 9, 32]>
// CHECK-DAG: <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d1, d2, d3, d4, d5)> by [<PassThrough ["go"] at [2] -> ["go"] at [1]>, <Unmerge{1, 32} ["n0", "n1"] at [0, 1] -> ["no"] at [0]>, <PassThrough ["ko", "0o", "1o"] at [5, 3, 4] -> ["ko", "0o", "1o"] at [4, 2, 3]>] bounds = [1, 32, 1, 9, 9, 32] -> [32, 1, 9, 9, 32]>
// CHECK-DAG: <affine_map<(d0, d1, d2) -> (0, d1 floordiv 81, 0, (d1 mod 81) floordiv 9, d1 mod 9, d2)> by [<Merge{1, 1} ["gemmG"] at [0] -> ["go", "n0"] at [2, 0]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["n1", "0o", "1o"] at [1, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [5]>] bounds = [1, 2592, 32] -> [1, 32, 1, 9, 9, 32]>

// CHECK:       %[[gemm:.*]] = rock.gemm tr %{{.*}} * %{{.*}}
// CHECK:       rock.store %[[gemm]] to %{{.*}} by atomic_add
