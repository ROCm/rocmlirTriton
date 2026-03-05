// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,5,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=gemm:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg1: tensor<1x2x1280xi8>, %arg2: tensor<1x1280x320xi8>, %arg3: tensor<1x2x320xi8>) -> tensor<1x2x320xi8> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %0 = rock.gemm %arg1 * %arg2 : tensor<1x2x1280xi8> * tensor<1x1280x320xi8> -> tensor<1x2x320xi8>
    %1 = rock.store %0 to %arg3 by  set : tensor<1x2x320xi8> -> tensor<1x2x320xi8> to tensor<1x2x320xi8>
    return %1 : tensor<1x2x320xi8>
  }
}
