// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,5,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg0: tensor<1x2x320xbf16>, %arg1: tensor<1x2x1280xbf16>, %arg2: tensor<1x1280x320xbf16>, %arg3: tensor<1x2x320xbf16>) -> tensor<1x2x320xbf16> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %0 = rock.gemm %arg1 * %arg2 : tensor<1x2x1280xbf16> * tensor<1x1280x320xbf16> -> tensor<1x2x320xbf16>
    %1 = arith.addf %0, %arg0 : tensor<1x2x320xbf16>
    %2 = rock.store %1 to %arg3 by  set : tensor<1x2x320xbf16> -> tensor<1x2x320xbf16> to tensor<1x2x320xbf16>
    return %2 : tensor<1x2x320xbf16>
  }
}
