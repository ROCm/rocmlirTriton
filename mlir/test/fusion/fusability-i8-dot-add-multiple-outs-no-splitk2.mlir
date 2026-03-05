// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,5,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg0: tensor<1x2x320xf16>, %arg1: tensor<1x2x1280xi8>, %arg2: tensor<1x1280x320xi8>, %arg3: tensor<1x2x320xf16>, %arg4: tensor<1x2x1xf16>) -> (tensor<1x2x320xf16>, tensor<1x2x1xf16>) attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %0 = rock.gemm %arg1 * %arg2 : tensor<1x2x1280xi8> * tensor<1x1280x320xi8> -> tensor<1x2x320xi8>
    %1 = arith.sitofp %0 : tensor<1x2x320xi8> to tensor<1x2x320xf16>
    %2 = arith.addf %1, %arg0 : tensor<1x2x320xf16>
    %3 = arith.fptoui %arg0 : tensor<1x2x320xf16> to tensor<1x2x320xi8>
    %4 = arith.addi %3, %0 : tensor<1x2x320xi8>
    %5 = arith.sitofp %4 : tensor<1x2x320xi8> to tensor<1x2x320xf16>
    %6 = rock.reduce sum %2 {axis = 2 : index} : tensor<1x2x320xf16> -> tensor<1x2x1xf16>
    %7 = rock.store %5 to %arg3 by  set : tensor<1x2x320xf16> -> tensor<1x2x320xf16> to tensor<1x2x320xf16>
    %8 = rock.store %6 to %arg4 by  set : tensor<1x2x1xf16> -> tensor<1x2x1xf16> to tensor<1x2x1xf16>
    return %7, %8 : tensor<1x2x320xf16>, tensor<1x2x1xf16>
  }
}
