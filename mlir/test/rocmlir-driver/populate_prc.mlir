// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc | FileCheck %s --check-prefix=F32

// F32: func.func @main()
// F32:  %[[RES:.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// F32-NEXT:    call @printMemrefF32(%[[RES]]) : (memref<*xf32>) -> ()

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc -t f16 | FileCheck %s --check-prefixes=CHECK '-D$TYPE=f16'
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc -t bf16 | FileCheck %s --check-prefix=CHECK '-D$TYPE=bf16'

// CHECK-LABEL: func.func @conv_cpu
// CHECK-SAME: (%{{.*}}: tensor<9216x[[$TYPE]]>, %{{.*}}: tensor<1048576x[[$TYPE]]>, %{{.*}}: tensor<14745600x[[$TYPE]]>)
// CHECK: tensor.expand_shape
// CHECK: arith.extf {{.*}} : [[$TYPE]] to f32
// CHECK-LABEL: func.func @main
// CHECK: bufferization.to_tensor
// CHECK: call @conv_cpu(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<9216x[[$TYPE]]>, tensor<{{.*}}>, tensor<{{.*}}>) -> tensor<{{.*}}>
// CHECK: bufferization.to_buffer
// CHECK: call @_memcpy_[[$TYPE]]_f32_{{[0-9]+}}(%{{.*}}, %{{.*}}) : (memref<{{.*}}x[[$TYPE]]>, memref<{{.*}}xf32>) -> ()
// CHECK:  %[[RES2:.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// CHECK:  call @printMemrefF32(%[[RES2]]) : (memref<*xf32>) -> ()

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc -t i8 | FileCheck %s --check-prefix=INT8

// INT8: func.func @main()
// INT8:  [[RES:%.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xi32>
// INT8-NEXT:    call @printMemrefI32([[RES]]) : (memref<*xi32>) -> ()
