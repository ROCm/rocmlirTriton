// TODO(rocmlirTriton): error: 'rock.transform' op failed to verify that all of {input, output} have same element type
// UNSUPPORTED: true
// RUN: rocmlir-gen --clone-harness -arch %arch -fut gemmi4f16 %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -fut gemmi4f16 --verifier clone - | rocmlir-driver -c -arch %arch | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// ALLOW-RETRIES: 2

!aCompressedFlat = tensor<4096xi4>
!bFlat = tensor<4096xf16>
!cFlat = tensor<4096xf16>
!aCompressed = tensor<1x64x64xi4>
!aExpanded = tensor<1x64x64xi8>
!a = tensor<1x64x64xf16>
!b = tensor<1x64x64xf16>
!c = tensor<1x64x64xf16>

func.func @gemmi4f16(%arg0: !aCompressedFlat, %arg1: !bFlat) -> !cFlat attributes {rock.kernel} {
  %const_shape = "tosa.const_shape"() { values = dense<[1, 64, 64]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = tosa.reshape %arg0, %const_shape : (!aCompressedFlat, !tosa.shape<3>) -> !aCompressed
  %1 = tosa.reshape %arg1, %const_shape : (!bFlat, !tosa.shape<3>) -> !b
  %2 = arith.extui %0 : !aCompressed to !aExpanded
  %3 = arith.uitofp %2 : !aExpanded to !a
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %6 = tosa.matmul %3, %1, %a_zp, %b_zp : (!a, !b, tensor<1xf16>, tensor<1xf16>) -> !c
  %const_shape2 = "tosa.const_shape"() { values = dense<[4096]> : tensor<1xindex> } : () -> !tosa.shape<1>
  %7 = tosa.reshape %6, %const_shape2 : (!c, !tosa.shape<1>) -> !cFlat
  func.return %7 : !cFlat
}
