// RUN: sed s/##TOKEN_ARCH##/gfx1201/g %s | rocmlir-opt -rock-to-ttir | FileCheck %s --check-prefix=BF16X3
// RUN: sed s/##TOKEN_ARCH##/gfx1250/g %s | rocmlir-opt -rock-to-ttir | FileCheck %s --check-prefix=NATIVE
// RUN: sed s/##TOKEN_ARCH##/gfx942/g %s | rocmlir-opt -rock-to-ttir | FileCheck %s --check-prefix=NATIVE

// BF16X3-LABEL: @f32_dot
// BF16X3: tt.dot {{.*}}, inputPrecision = bf16x3 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>

// NATIVE-LABEL: @f32_dot
// NATIVE: tt.dot
// NATIVE-NOT: inputPrecision = bf16x3
// NATIVE-SAME: : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
func.func @f32_dot(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
                   %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf32>, tensor<64x64xf32>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}
