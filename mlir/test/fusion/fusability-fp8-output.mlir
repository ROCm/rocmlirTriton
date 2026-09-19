// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,5,2,0,0 - < %s | FileCheck %s --check-prefix=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefix=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot(%arg0: tensor<1x2x1280xf8E4M3FNUZ>, %arg1: tensor<1x1280x320xf8E4M3FNUZ>, %arg2: tensor<1x2x320xf8E4M3FNUZ>) -> tensor<1x2x320xf8E4M3FNUZ> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
    %0 = rock.gemm %arg0 * %arg1 : tensor<1x2x1280xf8E4M3FNUZ> * tensor<1x1280x320xf8E4M3FNUZ> -> tensor<1x2x320xf8E4M3FNUZ>
    %1 = rock.store %0 to %arg2 by set : tensor<1x2x320xf8E4M3FNUZ> -> tensor<1x2x320xf8E4M3FNUZ> to tensor<1x2x320xf8E4M3FNUZ>
    return %1 : tensor<1x2x320xf8E4M3FNUZ>
  }
}
