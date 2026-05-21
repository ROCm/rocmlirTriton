// RUN: sed -e 's/##ARCH##/%arch/g' %s | rocmlir-gen -ph -rand 1 -rand_type float -fut mm_fut --verifier clone - | FileCheck %s

module attributes {rock.arch = "##ARCH##"} {
  func.func private @mm_fut_cpu_host(%arg0: tensor<1x256x768xf32>, %arg1: tensor<1x768x768xf32>, %arg2: tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf16>) {
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x768xf32>, tensor<1x768x768xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x768xf32>
    %1 = tosa.add %0, %arg2 : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %2 = tosa.cast %1 : (tensor<1x256x768xf32>) -> tensor<1x256x768xf16>
    return %0, %2 : tensor<1x256x768xf32>, tensor<1x256x768xf16>
  }
  func.func private @mm_fut(%arg0: tensor<1x256x768xf32>, %arg1: tensor<1x768x768xf32>, %arg2: tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf16>) attributes {rock.kernel} {
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x768xf32>, tensor<1x768x768xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x768xf32>
    %1 = tosa.add %0, %arg2 : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %2 = tosa.cast %1 : (tensor<1x256x768xf32>) -> tensor<1x256x768xf16>
    return %0, %2 : tensor<1x256x768xf32>, tensor<1x256x768xf16>
  }
}

// CHECK-LABEL: func.func @main()
// CHECK: call @mm_fut_gpu({{.*}}) : (memref<1x256x768xf32>, memref<1x768x768xf32>, memref<1x256x768xf32>, memref<1x256x768xf32>, memref<1x256x768xf16>) -> ()
// CHECK: call @mm_fut_cpu_host({{.*}}) : (tensor<1x256x768xf32>, tensor<1x768x768xf32>, tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf16>)
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<1x256x768xf32> to memref<1x256x768xf32>
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<1x256x768xf16> to memref<1x256x768xf16>
// CHECK-DAG: call @mm_fut_verify{{[0-9]+}}({{.*}}) : (memref<1x256x768xf32>, memref<1x256x768xf32>) -> ()
// CHECK-DAG: call @mm_fut_verify{{[0-9]+}}({{.*}}) : (memref<1x256x768xf16>, memref<1x256x768xf16>) -> ()

// CHECK: func.func @mm_fut_gpu(%{{.*}}: memref<1x256x768xf32>, %{{.*}}: memref<1x768x768xf32>, %{{.*}}: memref<1x256x768xf32>, %{{.*}}: memref<1x256x768xf32>, %{{.*}}: memref<1x256x768xf16>)
// CHECK: bufferization.to_buffer {{.*}} : tensor<1x256x768xf32> to memref<1x256x768xf32>
// CHECK: bufferization.to_buffer {{.*}} : tensor<1x256x768xf16> to memref<1x256x768xf16>
