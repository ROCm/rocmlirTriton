// RUN: rocmlir-gen --arch gfx90a -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,triton,binary --verify-passes | FileCheck %s 

// CHECK: rock.grid_size{{.*}} = {{.*}} : i32, triton.hsaco