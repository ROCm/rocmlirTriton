// truncf on the attention output must NOT be fused with splitKV > 1:
// narrowing does not commute with the LSE combine.
// Regression guard for the extf carve-out, ported from rocMLIR.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0

module {
  func.func @attn_splitkv_output_truncf_not_fusible(%q: tensor<1x1024x64xf32>, %k: tensor<1x64x1024xf32>, %v: tensor<1x1024x64xf32>, %out: tensor<4x1024x64xf16>) -> tensor<4x1024x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
    %result, %lse = rock.attention{
     qk = %q * %k : tensor<1x1024x64xf32>, tensor<1x64x1024xf32>
     softmax(qk) * %v : tensor<1x1024x64xf32>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 4 : i32} -> tensor<4x1024x64xf32>, tensor<4x1024xf32>
    %narrowed = arith.truncf %result : tensor<4x1024x64xf32> to tensor<4x1024x64xf16>
    %stored = rock.store %narrowed to %out by set : tensor<4x1024x64xf16> -> tensor<4x1024x64xf16> to tensor<4x1024x64xf16>
    return %stored : tensor<4x1024x64xf16>
  }
}
