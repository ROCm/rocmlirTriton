// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=gpu --mlir-print-ir-after=rock-regularize-output 2>&1 | FileCheck %s

// CHECK: rock.transform {{.*}} by <affine_map<(d0, d1, d2, d3, d4) -> (d4)> by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>, <AddDim{30} ["exp2"] at [2] -> [] at []>, <AddDim{30} ["exp3"] at [3] -> [] at []>, <PassThrough ["dim0"] at [4] -> ["dim0"] at [0]>] bounds = [1, 1, 30, 30, 16] -> [16]>

#map_in = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 32 + d2) * 32 + d3) * 8 + d4)>
#tf_in = #rock.transform_map<#map_in by [<AddDim{1} ["gi"] at [0] -> [] at []>, <Unmerge{1, 32, 32, 8} ["ni", "hi", "wi", "ci"] at [1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 1, 32, 32, 8] -> [8192]>
#map_fil = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 3 + d2) * 3 + d3) * 8 + d4)>
#tf_fil = #rock.transform_map<#map_fil by [<AddDim{1} ["g"] at [0] -> [] at []>, <Unmerge{16, 3, 3, 8} ["k", "y", "x", "c"] at [1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 16, 3, 3, 8] -> [1152]>
#map_bias = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
#tf_bias = #rock.transform_map<#map_bias by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>, <AddDim{30} ["exp2"] at [2] -> [] at []>, <AddDim{30} ["exp3"] at [3] -> [] at []>, <PassThrough ["dim0"] at [4] -> ["dim0"] at [0]>] bounds = [1, 1, 30, 30, 16] -> [16]>
#map_out = affine_map<(d0) -> (0, 0, d0 floordiv 480, (d0 floordiv 16) mod 30, d0 mod 16)>
#tf_out = #rock.transform_map<#map_out by [<Merge{1, 1, 30, 30, 16} ["raw"] at [0] -> ["go", "no", "ho", "wo", "ko"] at [0, 1, 2, 3, 4]>] bounds = [14400] -> [1, 1, 30, 30, 16]>
module {
  func.func @test_fusion(%arg0: tensor<8192xf32>, %arg1: tensor<1152xf32>, %arg2: tensor<16xf32>, %arg3: tensor<14400xf32>) -> tensor<14400xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
    %input = rock.transform %arg0 by #tf_in : tensor<8192xf32> to tensor<1x1x32x32x8xf32>
    %filter = rock.transform %arg1 by #tf_fil : tensor<1152xf32> to tensor<1x16x3x3x8xf32>
    %bias = rock.transform %arg2 by #tf_bias : tensor<16xf32> to tensor<1x1x30x30x16xf32>
    %conv = rock.conv(%filter, %input) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["gi", "ni", "hi", "wi", "ci"], output_layout = ["go", "no", "ho", "wo", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : tensor<1x16x3x3x8xf32>, tensor<1x1x32x32x8xf32> -> tensor<1x1x30x30x16xf32>
    %fused = arith.addf %conv, %bias : tensor<1x1x30x30x16xf32>
    %flat = rock.transform %fused by #tf_out : tensor<1x1x30x30x16xf32> to tensor<14400xf32>
    %stored = rock.store %flat to %arg3 by set : tensor<14400xf32> -> tensor<14400xf32> to tensor<14400xf32>
    return %stored : tensor<14400xf32>
  }
}
