// RUN: sed s/##TOKEN_ARCH##/%arch/g %s \
// RUN:   | rocmlir-gen -fut test_reduce_i8 --arch %arch --clone-harness - \
// RUN:   | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel \
// RUN:   | rocmlir-gen -ph -print-results -fut test_reduce_i8 --verifier clone -rand none - \
// RUN:   | rocmlir-driver -c -arch %arch \
// RUN:   | rocm-run | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base

// Same as rock-reduce-i32, but the gemm result is truncated to i8 before the
// reduction, so the atomic add is 8-bit. No target has a native 8-bit atomic
// add; the backend must expand it to a masked compare-and-swap loop.
func.func @test_reduce_i8(%arg0: tensor<1x6x1xi8>, %arg1: tensor<1x1x40xi8>) -> tensor<1x6x1xi8> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %a_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %b_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %gemm = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = i32} : (tensor<1x6x1xi8>, tensor<1x1x40xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x40xi32>
  %trunc = "tosa.cast"(%gemm) : (tensor<1x6x40xi32>) -> tensor<1x6x40xi8>
  %reduced = "tosa.reduce_sum"(%trunc) {axis = 2 : i32} : (tensor<1x6x40xi8>) -> tensor<1x6x1xi8>
  return %reduced : tensor<1x6x1xi8>
}
