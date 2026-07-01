// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK:  73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73

module {
  func.func @test_fusion(%arg0: !migraphx.shaped<1x8x32x32xf32, 8192x1024x32x1>, %arg1: !migraphx.shaped<16x8x3x3xf32, 72x9x3x1>, %arg2: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<1x16x30x30xf32, 14400x900x30x1> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x32x32xf32, 8192x1024x32x1>, <16x8x3x3xf32, 72x9x3x1> -> <1x16x30x30xf32, 14400x900x30x1>
    %1 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = [1 : i64, 16 : i64, 30 : i64, 30 : i64]} : <16xf32, 1> -> <1x16x30x30xf32, 0x1x0x0>
    %2 = migraphx.add %0, %1 : <1x16x30x30xf32, 14400x900x30x1>, <1x16x30x30xf32, 0x1x0x0> -> <1x16x30x30xf32, 14400x900x30x1>
    return %2 : !migraphx.shaped<1x16x30x30xf32, 14400x900x30x1>
  }
}
