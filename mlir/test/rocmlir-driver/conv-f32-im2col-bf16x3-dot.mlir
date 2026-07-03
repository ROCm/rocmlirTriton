// RUN: rocmlir-driver -kernel-pipeline=gpu --arch=gfx1201 %s | FileCheck %s
// RUN: rocmlir-driver -kernel-pipeline=gpu --arch=gfx1201 --disable-fast-math %s | FileCheck %s --check-prefix=NOFAST

// CHECK-LABEL: tt.func @rock_conv_gkc01_ngc01_ngk01
// CHECK: scf.for
// CHECK: tt.dot {{.*}}, inputPrecision = bf16x3 {{.*}} tensor<64x32xf32> * tensor<32x64xf32> -> tensor<64x64xf32>

// NOFAST-LABEL: tt.func @rock_conv_gkc01_ngc01_ngk01
// NOFAST: scf.for
// NOFAST: tt.dot
// NOFAST-NOT: inputPrecision = bf16x3
// NOFAST-SAME: tensor<64x32xf32> * tensor<32x64xf32> -> tensor<64x64xf32>

#map = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 128 + d2) * 3 + d3) * 3 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 256 + d3) * 256 + d4)>
#map2 = affine_map<(d0) -> (0, 0, d0 floordiv 65536, (d0 mod 65536) floordiv 256, d0 mod 256)>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 128, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 128, 3, 3] -> [73728]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{128, 256, 256} ["ci", "0i", "1i"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 128, 256, 256] -> [8388608]>
#transform_map2 = #rock.transform_map<#map2 by [<Merge{64, 256, 256} ["raw"] at [0] -> ["ko", "0o", "1o"] at [2, 3, 4]>, <ConstDim{0, 1} [] at [] -> ["no"] at [0]>, <ConstDim{0, 1} [] at [] -> ["go"] at [1]>] bounds = [4194304] -> [1, 1, 64, 256, 256]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1201"} {
  func.func @rock_conv_gkc01_ngc01_ngk01(%arg0: tensor<73728xf32>, %arg1: tensor<8388608xf32>, %arg2: tensor<4194304xf32>) -> tensor<4194304xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1201", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 64 : i32} {
    %0 = rock.transform %arg0 by #transform_map : tensor<73728xf32> to tensor<1x64x128x3x3xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<8388608xf32> to tensor<1x1x128x256x256xf32>
    %2 = rock.conv(%0, %1) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], perf_config = "gemm:v2:64,64,32,1,1,4,0,1,2,0,0,-1,-1,-1,-1,-1,-1", strides = [1 : index, 1 : index]} : tensor<1x64x128x3x3xf32>, tensor<1x1x128x256x256xf32> -> tensor<1x1x64x256x256xf32>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x1x64x256x256xf32> to tensor<4194304xf32>
    %4 = rock.store %3 to %arg2 by set : tensor<4194304xf32> -> tensor<4194304xf32> to tensor<4194304xf32>
    return %4 : tensor<4194304xf32>
  }
}
