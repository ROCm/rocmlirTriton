// RUN: rocmlir-driver --host-pipeline=migraphx,highlevel %s | \
// RUN: rocmlir-opt -empty-tensor-to-alloc-tensor \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:   -buffer-results-to-out-params="add-result-attr=false hoist-dynamic-allocs=false hoist-static-allocs=false modify-public-functions=true" | \
// RUN: rocmlir-gen -ph -print-results -rand none -fut test - | \
// RUN: rocmlir-driver --host-pipeline=backend | \
// RUN: mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s

module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// CHECK-NEXT: 0.0320586,  0.236883,  0.643914,  0.0871443

  func.func @create_test_tensor() -> !migraphx.shaped<4xf32, 1> {
    %0 = migraphx.literal (dense<[0.0, 2.0, 3.0, 1.0]> : tensor<4xf32>) : <4xf32, 1>
    return %0 : !migraphx.shaped<4xf32, 1>
  }

  func.func @softmax(%arg0: !migraphx.shaped<4xf32, 1>) -> !migraphx.shaped<4xf32, 1> {
    %0 = migraphx.softmax %arg0 {axis = 0 : i64} : <4xf32, 1> -> <4xf32, 1>
     return %0 : !migraphx.shaped<4xf32, 1>
  }

  func.func @test() -> !migraphx.shaped<4xf32, 1> {
    %0 = call @create_test_tensor() : () -> (!migraphx.shaped<4xf32, 1>)
    %1 = call @softmax(%0) : (!migraphx.shaped<4xf32, 1>) -> (!migraphx.shaped<4xf32, 1>)
     return %1 : !migraphx.shaped<4xf32, 1>
  }
}
