// Output fusions are not allowed for attention ops with splitKV > 1.
// With flash decoding, partial results need LSE-based corrections in a
// subsequent stage, so output fusions should not be applied to the attention
// kernel. Input fusions and fusions between the two GEMMs are still allowed.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0

module {
  func.func @attn_splitkv_output_fusion_not_fusible(%q: tensor<1x1024x64xf16>, %k: tensor<1x64x1024xf16>, %v: tensor<1x1024x64xf16>, %bias: tensor<4x1024x64xf16>, %out: tensor<4x1024x64xf16>) -> tensor<4x1024x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
    %result, %lse = rock.attention{
     qk = %q * %k : tensor<1x1024x64xf16>, tensor<1x64x1024xf16>
     softmax(qk) * %v : tensor<1x1024x64xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 4 : i32} -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
    %fused = arith.mulf %result, %bias : tensor<4x1024x64xf16>
    %stored = rock.store %fused to %out by set : tensor<4x1024x64xf16> -> tensor<4x1024x64xf16> to tensor<4x1024x64xf16>
    return %stored : tensor<4x1024x64xf16>
  }
}
