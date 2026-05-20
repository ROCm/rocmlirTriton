// Check that rocmlir-gen -ph works when building a harness around a kernel
// whose outputs are returned purely as tensor func.return values.

// RUN: sed -e 's/##ARCH##/%arch/g' %s | rocmlir-gen -ph -rand 1 -rand_type float -fut multi_out_fut --verifier clone - | FileCheck %s

// The signature deliberately has no trailing output args, mimicking what
// comes out of --clone-harness kernel without fusion: pure tensor inputs
// returning multiple tensor results of dtypes that intentionally do not
// alias any input arg dtype/shape. The kernel attribute is set so the
// harness emits a GPU wrapper for it.
module attributes {rock.arch = "##ARCH##"} {
  func.func private @multi_out_fut_cpu_host(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>,
                                            %arg2: tensor<128xf16>, %arg3: tensor<128xf16>)
      -> (tensor<256xf32>, tensor<128xf16>) {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    %1 = arith.addf %arg2, %arg3 : tensor<128xf16>
    return %0, %1 : tensor<256xf32>, tensor<128xf16>
  }
  func.func private @multi_out_fut(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>,
                                   %arg2: tensor<128xf16>, %arg3: tensor<128xf16>)
      -> (tensor<256xf32>, tensor<128xf16>) attributes {rock.kernel} {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    %1 = arith.addf %arg2, %arg3 : tensor<128xf16>
    return %0, %1 : tensor<256xf32>, tensor<128xf16>
  }
}

// CHECK-LABEL: func.func @main()
// CHECK: call @multi_out_fut_gpu({{.*}}) : (memref<256xf32>, memref<256xf32>, memref<128xf16>, memref<128xf16>, memref<256xf32>, memref<128xf16>) -> ()
// CHECK: call @multi_out_fut_cpu_host({{.*}}) : (tensor<256xf32>, tensor<256xf32>, tensor<128xf16>, tensor<128xf16>) -> (tensor<256xf32>, tensor<128xf16>)
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<256xf32> to memref<256xf32>
// CHECK-DAG: bufferization.to_buffer {{.*}} : tensor<128xf16> to memref<128xf16>
// CHECK-DAG: call @multi_out_fut_verify{{[0-9]+}}({{.*}}) : (memref<256xf32>, memref<256xf32>) -> ()
// CHECK-DAG: call @multi_out_fut_verify{{[0-9]+}}({{.*}}) : (memref<128xf16>, memref<128xf16>) -> ()

// CHECK: func.func @multi_out_fut_gpu(%{{.*}}: memref<256xf32>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<128xf16>, %{{.*}}: memref<128xf16>, %{{.*}}: memref<256xf32>, %{{.*}}: memref<128xf16>)
// CHECK: bufferization.to_buffer {{.*}} : tensor<256xf32> to memref<256xf32>
// CHECK: bufferization.to_buffer {{.*}} : tensor<128xf16> to memref<128xf16>
