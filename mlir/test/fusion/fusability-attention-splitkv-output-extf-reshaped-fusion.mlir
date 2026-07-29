// Companion to the transposed-extf rejection: a flattening view on the extf
// output keeps every element at the same linear offset, so the LSE combine
// still finds the partial results where it expects them and the extf carve-out
// continues to apply.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s
// CHECK: fusible:1

#map = affine_map<(d0) -> (d0 floordiv 65536, (d0 mod 65536) floordiv 64, d0 mod 64)>
#flatten = #rock.transform_map<#map by [<Merge{4, 1024, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [262144] -> [4, 1024, 64]>

module {
  func.func @attn_splitkv_output_extf_reshaped_fusible(%q: tensor<1x1024x64xf16>, %k: tensor<1x64x1024xf16>, %v: tensor<1x1024x64xf16>, %out: tensor<262144xf32>) -> tensor<262144xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
    %result, %lse = rock.attention{
     qk = %q * %k : tensor<1x1024x64xf16>, tensor<1x64x1024xf16>
     softmax(qk) * %v : tensor<1x1024x64xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 4 : i32} -> tensor<4x1024x64xf16>, tensor<4x1024xf32>
    %widened = arith.extf %result : tensor<4x1024x64xf16> to tensor<4x1024x64xf32>
    %flat = rock.transform %widened by #flatten : tensor<4x1024x64xf32> to tensor<262144xf32>
    %stored = rock.store %flat to %out by set : tensor<262144xf32> -> tensor<262144xf32> to tensor<262144xf32>
    return %stored : tensor<262144xf32>
  }
}
