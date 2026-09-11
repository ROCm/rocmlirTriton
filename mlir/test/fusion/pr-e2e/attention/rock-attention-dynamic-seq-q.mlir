// Cross-attention where seq_q is dynamic: the
// kernel takes seq_q as an argument and the launch derives its grid from that,
// while the key/value sequence length stays a compile-time constant.

// RUN: rocmlir-gen -operation attention -t f32 --arch %arch -g 1 -seq_len_q 64 -seq_len_k 128 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 32 -head_dim_v 32 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=False -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 -ph -pr -rand 1 -rand_type float \
// RUN: | rocmlir-driver -c | rocm-run | tail -1 > %t.static64
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -ph -pr -rand 1 -rand_type float -fut rock_attention -m 64 - \
// RUN: | rocmlir-driver -c | rocm-run | tail -1 > %t.dynamic64
// RUN: diff %t.static64 %t.dynamic64

// RUN: rocmlir-gen -operation attention -t f32 --arch %arch -g 1 -seq_len_q 256 -seq_len_k 128 -num_heads_q 1 -num_heads_kv 1 -head_dim_qk 32 -head_dim_v 32 -with-attn-scale=False -with-attn-bias=False -transQ=False -transK=False -transV=False -transO=False -causal=False -return_lse=False -split_kv=1 -ph -pr -rand 1 -rand_type float \
// RUN: | rocmlir-driver -c | rocm-run | tail -1 > %t.static256
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -ph -pr -rand 1 -rand_type float -fut rock_attention -m 256 - \
// RUN: | rocmlir-driver -c | rocm-run | tail -1 > %t.dynamic256
// RUN: diff %t.static256 %t.dynamic256

#map = affine_map<(d0, d1, d2) -> (d1 * 32 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map2 = affine_map<(d0) -> (0, d0 floordiv 32, d0 mod 32)>
#transform_map = #rock.transform_map<#map by [<Unmerge{?, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, ?, 32] -> [?]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{32, 128} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 128] -> [4096]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{128, 32} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 32] -> [4096]>
#transform_map3 = #rock.transform_map<#map2 by [<Merge{?, 32} ["raw"] at [0] -> ["seq_q", "head_v"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [?] -> [1, ?, 32]>
module attributes {rock.arch = "##TOKEN_ARCH##"} {
  func.func @rock_attention(%arg0: tensor<?xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096xf32>, %arg3: tensor<?xf32>) -> tensor<?xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<?xf32> to tensor<1x?x32xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<4096xf32> to tensor<1x32x128xf32>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<4096xf32> to tensor<1x128x32xf32>
    %result = rock.attention{
     qk = %0 * %1 : tensor<1x?x32xf32>, tensor<1x32x128xf32>
     qk = elementwise {
    ^bb0(%arg4: tensor<1x?x128xf32>):
      rock.yield %arg4 : tensor<1x?x128xf32>
    }
     softmax(qk) * %2 : tensor<1x128x32xf32>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<1x?x32xf32>
    %3 = rock.transform %result by #transform_map3 : tensor<1x?x32xf32> to tensor<?xf32>
    %4 = rock.store %3 to %arg3 by set : tensor<?xf32> -> tensor<?xf32> to tensor<?xf32>
    return %4 : tensor<?xf32>
  }
}
