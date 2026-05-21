// Defensive coverage for a kernel whose trailing parameters happen to be
// shape/dtype-compatible with the results, but in a different order than
// the result tuple.

// RUN: sed -e 's/##ARCH##/%arch/g' %s | rocmlir-gen -ph -rand 1 -rand_type float -fut permuted_fut --verifier clone - | FileCheck %s

module attributes {rock.arch = "##ARCH##"} {
  // Kernel arg order is (in_f32, out_f16, out_f32) but the result tuple
  // is (f32, f16) - the trailing two args are reversed relative to the
  // result tuple. We synthesise the result values from the kernel inputs
  // so the body just parses.
  func.func private @permuted_fut_cpu_host(%arg0: tensor<256xf32>, %arg1: tensor<128xf16>, %arg2: tensor<256xf32>) -> (tensor<256xf32>, tensor<128xf16>) {
    %0 = tosa.add %arg0, %arg2 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %1 = tosa.add %arg1, %arg1 : (tensor<128xf16>, tensor<128xf16>) -> tensor<128xf16>
    return %0, %1 : tensor<256xf32>, tensor<128xf16>
  }
  func.func private @permuted_fut(%arg0: tensor<256xf32>, %arg1: tensor<128xf16>, %arg2: tensor<256xf32>) -> (tensor<256xf32>, tensor<128xf16>) attributes {rock.kernel} {
    %0 = tosa.add %arg0, %arg2 : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    %1 = tosa.add %arg1, %arg1 : (tensor<128xf16>, tensor<128xf16>) -> tensor<128xf16>
    return %0, %1 : tensor<256xf32>, tensor<128xf16>
  }
}

// CHECK-LABEL: func.func @main()
// CHECK: call @permuted_fut_gpu({{.*}}) : (memref<256xf32>, memref<128xf16>, memref<256xf32>, memref<256xf32>, memref<128xf16>) -> ()
// CHECK: call @permuted_fut_cpu_host({{.*}}) : (tensor<256xf32>, tensor<128xf16>, tensor<256xf32>) -> (tensor<256xf32>, tensor<128xf16>)
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<256xf32> to memref<256xf32>
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<128xf16> to memref<128xf16>
// CHECK-DAG: call @permuted_fut_verify{{[0-9]+}}({{.*}}) : (memref<256xf32>, memref<256xf32>) -> ()
// CHECK-DAG: call @permuted_fut_verify{{[0-9]+}}({{.*}}) : (memref<128xf16>, memref<128xf16>) -> ()

// CHECK: func.func @permuted_fut_gpu(%{{.*}}: memref<256xf32>, %{{.*}}: memref<128xf16>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<128xf16>)
// CHECK: call @permuted_fut({{.*}}) : (tensor<256xf32>, tensor<128xf16>, tensor<256xf32>) -> (tensor<256xf32>, tensor<128xf16>)
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<256xf32> to memref<256xf32>
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<128xf16> to memref<128xf16>
