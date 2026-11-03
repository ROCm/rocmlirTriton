// RUN: rocmlir-opt --rock-elementwise-pipeline -mlir-print-local-scope %s | FileCheck %s --check-prefix=FAST
// RUN: rocmlir-opt --rock-elementwise-pipeline='disable-fast-math=true' -mlir-print-local-scope %s | FileCheck %s --check-prefix=NOFAST --implicit-check-not='fastmath<'

module {
  // FAST-LABEL: module attributes {rock.grid_size.ew_add = 1 : i32}
  // FAST: tt.func @ew_add(
  // FAST: tt.get_program_id x
  // FAST: tt.load
  // FAST: tt.load
  // FAST: arith.addf {{.*}} fastmath<{{.*}}contract{{.*}}> : tensor<256xf32>
  // FAST: tt.store
  // FAST: tt.return
  // FAST-NOT: rock.load_marker
  // FAST-NOT: rock.store_marker

  // NOFAST-LABEL: module attributes {rock.grid_size.ew_add = 1 : i32}
  // NOFAST: tt.func @ew_add(
  // NOFAST: arith.addf {{.*}} : tensor<256xf32>
  func.func @ew_add(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, rock.elementwise, rock.block_size = 256 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    %1 = rock.store %0 to %arg2 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
    return %1 : tensor<256xf32>
  }
}
