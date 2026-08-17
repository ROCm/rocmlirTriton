// RUN: sed s/##TOKEN_ARCH##/%arch/g %s \
// RUN:   | rocmlir-gen -fut test_reduce_i32 --arch %arch --clone-harness - \
// RUN:   | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel \
// RUN:   | rocmlir-gen -ph -print-results -fut test_reduce_i32 --verifier clone -rand none - \
// RUN:   | rocmlir-driver -c -arch %arch \
// RUN:   | rocm-run | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base

// An i8 gemm accumulating into i32 followed by reduce-sum, which lowers to an
// i32 atomic add. Atomic add has no arch gate: where the hardware lacks a
// native instruction the backend expands it to a compare-and-swap loop, so
// this must work on every target.
func.func @test_reduce_i32(%arg0: tensor<1x6x1xi8>, %arg1: tensor<1x1x40xi8>) -> tensor<1x6x1xi32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %a_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %b_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %gemm = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = i32} : (tensor<1x6x1xi8>, tensor<1x1x40xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x40xi32>
  %reduced = "tosa.reduce_sum"(%gemm) {axis = 2 : i32} : (tensor<1x6x40xi32>) -> tensor<1x6x1xi32>
  return %reduced : tensor<1x6x1xi32>
}
