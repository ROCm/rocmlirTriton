// RUN: rocmlir-gen -fut ew_add --arch %arch --clone-harness %s | FileCheck %s

module {
  // CHECK-LABEL: module attributes {rock.arch = "{{.*}}"}
  // CHECK: func.func @ew_add_cpu_host(
  // CHECK-SAME: attributes {rock.arch = "{{.*}}", rock.block_size = 64 : i32, rock.elementwise}
  // CHECK: func.func @ew_add(
  // CHECK-SAME: attributes {rock.arch = "{{.*}}", rock.block_size = 64 : i32, rock.elementwise, rock.kernel}
  func.func @ew_add(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel, rock.elementwise, rock.block_size = 64 : i32, rock.arch = "##TOKEN_ARCH##"} {
    %0 = arith.addf %arg0, %arg1 : tensor<64xf32>
    %1 = rock.store %0 to %arg2 by set : tensor<64xf32> -> tensor<64xf32> to tensor<64xf32>
    return %1 : tensor<64xf32>
  }
}
