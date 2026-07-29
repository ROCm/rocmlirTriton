// An extf chained with another op must NOT be fused with splitKV > 1: the
// carve-out only allows a lossless widening on its own.
// Regression guard for the extf carve-out, ported from rocMLIR.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0

module {
  func.func @attn_splitkv_output_extf_chain_not_fusible(%q: tensor<1x1024x64xf16>, %k: tensor<1x64x1024xf16>, %v: tensor<1x1024x64xf16>, %bias: tensor<4x1024x64xf32>, %out: tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
    %result, %lse = rock.attention{
     qk = %q * %k : tensor<1x1024x64xf16>, tensor<1x64x1024xf16>
     softmax(qk) * %v : tensor<1x1024x64xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 4 : i32} -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
    %widened = arith.extf %result : tensor<4x1024x64xf16> to tensor<4x1024x64xf32>
    %fused = arith.mulf %widened, %bias : tensor<4x1024x64xf32>
    %stored = rock.store %fused to %out by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
    return %stored : tensor<4x1024x64xf32>
  }
}
