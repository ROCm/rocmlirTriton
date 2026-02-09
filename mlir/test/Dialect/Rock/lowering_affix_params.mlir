// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s
// RUN: rocmlir-gen --arch gfx908 -p --operation=conv | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func.func @{{.*}}(%{{.*}}: tensor<{{.*}}>, %{{.*}}: tensor<{{.*}}>, %arg2: tensor<{{.*}}>) -> tensor<{{.*}}> attributes {rock.arch = {{.*}}, rock.block_size = {{.*}} : i32, rock.enable_splitk_for_tuning, rock.kernel = 0 : i32, rock.num_chiplets = {{.*}} : i64, rock.num_cu = {{.*}} : i32}
