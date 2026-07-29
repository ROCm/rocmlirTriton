// Input fusions and fusions between the first and second gemm are allowed
// for attention ops with splitKV > 1. Only output fusions are disallowed.
// Ported from rocMLIR.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:1

module {
  func.func @attn_splitkv_input_intergemm_fusible(%q: tensor<1x1024x64xf16>, %qscale: tensor<1x1024x64xf16>, %k: tensor<1x64x1024xf16>, %v: tensor<1x1024x64xf16>, %mask: tensor<4x1024x1024xf16>, %out: tensor<4x1024x64xf16>) -> tensor<4x1024x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
    %scaledQ = arith.mulf %q, %qscale : tensor<1x1024x64xf16>
    %result, %lse = rock.attention{
     qk = %scaledQ * %k : tensor<1x1024x64xf16>, tensor<1x64x1024xf16>
     qk = elementwise otherIns(%mask : tensor<4x1024x1024xf16>) {
    ^bb0(%qkIn: tensor<4x1024x1024xf16>, %maskIn: tensor<4x1024x1024xf16>):
      %masked = arith.addf %qkIn, %maskIn : tensor<4x1024x1024xf16>
      rock.yield %masked : tensor<4x1024x1024xf16>
    }
     softmax(qk) * %v : tensor<1x1024x64xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 4 : i32} -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
    %stored = rock.store %result to %out by set : tensor<4x1024x64xf16> -> tensor<4x1024x64xf16> to tensor<4x1024x64xf16>
    return %stored : tensor<4x1024x64xf16>
  }
}
