// RUN: rocmlir-gen --arch gfx1100 --operation attention -t f16 -g 2 -seq_len_q 100 -seq_len_k 70 -head_dim_qk 64 -head_dim_v 32 --dynamic-dims g,seq_q,seq_k -return_lse --with-attn-scale | FileCheck %s --check-prefix=ALL
// RUN: rocmlir-gen --arch gfx1100 --operation attention -t f16 -g 2 -seq_len_q 100 -seq_len_k 70 -head_dim_qk 64 -head_dim_v 32 --dynamic-dims seq_k --transK=true --with-attn-bias --transBias=true | FileCheck %s --check-prefix=SEQ-K
// RUN: not rocmlir-gen --arch gfx1100 --operation attention -t f16 -seq_len_q 64 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 --dynamic-dims head_qk 2>&1 | FileCheck %s --check-prefix=BAD-NAME
// RUN: rocmlir-gen --arch gfx1100 --operation attention -t f16 -g 2 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 300 -head_dim_qk 32 -head_dim_v 32 --dynamic-dims g,seq_k --last_valid_kv_index 40,250 --prefix_offset 7,200 --split_kv 2 -return_lse | FileCheck %s --check-prefix=DECODE
// RUN: not rocmlir-gen --arch gfx1100 --operation attention -t f16 -seq_len_q 64 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 --dynamic-dims seq_k --sliding_window_look_back 16 2>&1 | FileCheck %s --check-prefix=SLIDING

// The kernel takes logical-rank arguments, with the scale, LSE and output
// following the dynamic sequence lengths, and stores the results straight
// into them.
// ALL: func.func @rock_attention(%{{.*}}: tensor<?x?x64xf16>, %{{.*}}: tensor<?x64x?xf16>, %{{.*}}: tensor<?x?x32xf16>, %{{.*}}: tensor<?x?x?xf16>, %{{.*}}: tensor<?x?xf16>, %{{.*}}: tensor<?x?x32xf16>) -> (tensor<?x?x32xf16>, tensor<?x?xf16>)
// ALL: ^bb0(%{{.*}}: tensor<?x?x?xf16>, %{{.*}}: tensor<?x?x?xf16>):
// ALL: rock.store {{.*}} : tensor<?x?x32xf16> -> tensor<?x?x32xf16> to tensor<?x?x32xf16>
// ALL: rock.store {{.*}} : tensor<?x?xf16> -> tensor<?x?xf16> to tensor<?x?xf16>

// A transposed bias is transposed by a symbolic map.
// SEQ-K: #[[TRANSPOSE:.*]] = #rock.transform_map<{{.*}} symbols = [arg(3, 1)] bounds = [2, 100, s0] -> [2, s0, 100]>
// SEQ-K: func.func @rock_attention(%{{.*}}: tensor<2x100x64xf16>, %{{.*}}: tensor<2x?x64xf16>, %{{.*}}: tensor<2x?x32xf16>, %[[BIAS:.*]]: tensor<2x?x100xf16>, %{{.*}}: tensor<2x100x32xf16>)
// SEQ-K: rock.transform %[[BIAS]] by #[[TRANSPOSE]] : tensor<2x?x100xf16> to tensor<2x100x?xf16>

// BAD-NAME: invalid --dynamic-dims entry 'head_qk'; expected one of: g,seq_q,seq_k

// With GQA, split-KV, a KV cache and a prefix, the batch of every argument is
// dynamic, including the per-group KV cache and prefix, which are broadcast
// over the query heads by symbolic maps.
// DECODE: #[[MERGE:.*]] = #rock.transform_map<{{.*}}<Merge{s0, 4} ["gemmG"] at [0] -> ["gemmG", "numHeadsQ"] at [0, 1]>] symbols = [arg(3, 0)] bounds = [s0 * 4] -> [s0, 4]>
// DECODE: func.func @rock_attention(%{{.*}}: tensor<?x1x32xf16>, %{{.*}}: tensor<?x32x?xf16>, %{{.*}}: tensor<?x?x32xf16>, %[[KV:.*]]: tensor<?xi32>, %{{.*}}: tensor<?xi32>, %{{.*}}: tensor<?x1xf16>, %{{.*}}: tensor<?x1x32xf16>) -> (tensor<?x1x32xf16>, tensor<?x1xf16>)
// DECODE: rock.transform %{{.*}} by #[[MERGE]] : tensor<?x4xi32> to tensor<?xi32>
// DECODE: rock.attention
// DECODE: {numHeadsKV = 2 : i32, numHeadsQ = 4 : i32, softmaxType = f32, splitKV = 2 : i32} -> tensor<?x1x32xf16>, tensor<?x1xf16>

// SLIDING: --dynamic-dims is not supported for attention with a sliding window or i8 inputs
