// RUN: rocmlir-opt --tosa-to-rock --verify-diagnostics %s

func.func @test_fp8_splitk(
    %a: tensor<2x128x64xf8E4M3FNUZ>,
    %b: tensor<2x64x256xf8E4M3FNUZ>) -> tensor<2x128x256xf8E4M3FNUZ>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FNUZ>}> : () -> tensor<1xf8E4M3FNUZ>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf8E4M3FNUZ>}> : () -> tensor<1xf8E4M3FNUZ>
  // expected-error @+2 {{'tosa.matmul' op split-K output element type 'f8E4M3FNUZ' does not support atomic add}}
  // expected-error @+1 {{failed to legalize operation 'tosa.matmul' that was explicitly marked illegal}}
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config = "gemm:v1:16,32,4,16,16,4,4,2,1,1,1"} : (tensor<2x128x64xf8E4M3FNUZ>, tensor<2x64x256xf8E4M3FNUZ>, tensor<1xf8E4M3FNUZ>, tensor<1xf8E4M3FNUZ>) -> tensor<2x128x256xf8E4M3FNUZ>
  return %c : tensor<2x128x256xf8E4M3FNUZ>
}
