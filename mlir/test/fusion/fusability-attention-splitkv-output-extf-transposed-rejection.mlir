// An extf reaching the output through a transposing view must NOT be fused with
// splitKV > 1: the LSE combine assumes the output's natural layout, so the
// carve-out only holds while the chain leaves that layout alone. Flatten and
// unflatten views are fine, since they keep every element at the same linear
// offset.
// Regression guard for the extf carve-out, ported from rocMLIR.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:0

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#transpose = #rock.transform_map<#map by [<PassThrough ["dim0", "dim1", "dim2"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 2, 1]>] bounds = [4, 64, 1024] -> [4, 1024, 64]>

module {
  func.func @attn_splitkv_output_extf_transposed_not_fusible(%q: tensor<1x1024x64xf16>, %k: tensor<1x64x1024xf16>, %v: tensor<1x1024x64xf16>, %out: tensor<4x64x1024xf32>) -> tensor<4x64x1024xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
    %result, %lse = rock.attention{
     qk = %q * %k : tensor<1x1024x64xf16>, tensor<1x64x1024xf16>
     softmax(qk) * %v : tensor<1x1024x64xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 4 : i32} -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
    %widened = arith.extf %result : tensor<4x1024x64xf16> to tensor<4x1024x64xf32>
    %transposed = rock.transform %widened by #transpose : tensor<4x1024x64xf32> to tensor<4x64x1024xf32>
    %stored = rock.store %transposed to %out by set : tensor<4x64x1024xf32> -> tensor<4x64x1024xf32> to tensor<4x64x1024xf32>
    return %stored : tensor<4x64x1024xf32>
  }
}
