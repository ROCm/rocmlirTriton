// Verifies that the tuning driver recognizes an i4 kernel argument and can
// allocate its packed buffer instead of rejecting the element type.
//
// RUN: sed s/##ARCH##/%arch/g %s \
// RUN: | rocmlir-tuning-driver \
// RUN:     --benchmark-config="gemm:v1:64,64,64,1,1,4,16,1,2,0,0" \
// RUN: | FileCheck %s
//
// CHECK: gemm:v1:64,64,64,1,1,4,16,1,2,0,0{{[[:space:]]+}}[0-9]

#map = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map1 = affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]>
#transform_map1 = #rock.transform_map<#map1 by [<Merge{1, 64, 64} ["raw"] at [0] -> ["g", "m", "n"] at [0, 1, 2]>] bounds = [4096] -> [1, 64, 64]>

module attributes {rock.arch = "amdgcn-amd-amdhsa:##ARCH##"} {
  func.func @gemmi4f16(%arg0: tensor<4096xi4>, %arg1: tensor<4096xf16>, %arg2: tensor<4096xf16>) -> tensor<4096xf16> attributes {rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<4096xi4> to tensor<1x64x64xi4>
    %1 = rock.transform %arg1 by #transform_map : tensor<4096xf16> to tensor<1x64x64xf16>
    %2 = arith.sitofp %0 : tensor<1x64x64xi4> to tensor<1x64x64xf16>
    %3 = rock.gemm %2 * %1 : tensor<1x64x64xf16> * tensor<1x64x64xf16> -> tensor<1x64x64xf16>
    %4 = rock.transform %3 by #transform_map1 : tensor<1x64x64xf16> to tensor<4096xf16>
    %5 = rock.store %4 to %arg2 by set : tensor<4096xf16> -> tensor<4096xf16> to tensor<4096xf16>
    return %5 : tensor<4096xf16>
  }
}
