// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,5,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg0: tensor<1x2x320xf32>, %arg1: tensor<1x2x1280xf32>, %arg2: tensor<1x1280x320xf32>, %arg3: tensor<1x2x320xf16>) -> tensor<1x2x320xf16> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
    %0 = rock.gemm %arg1 * %arg2 : tensor<1x2x1280xf32> * tensor<1x1280x320xf32> -> tensor<1x2x320xf32>
    %1 = arith.addf %0, %arg0 : tensor<1x2x320xf32>
    %2 = arith.truncf %1 : tensor<1x2x320xf32> to tensor<1x2x320xf16>
    %3 = rock.store %2 to %arg3 by  set : tensor<1x2x320xf16> -> tensor<1x2x320xf16> to tensor<1x2x320xf16>
    return %3 : tensor<1x2x320xf16>
  }
}
