//===- mixr-attention-shared-scale-bias-problem-key.mlir ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f32 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 7 -seq_len_k 7 -head_dim_qk 3 -head_dim_v 3 -with-attn-scale true -with-attn-bias true -transBias false -share-attn-scale-bias true -num_dequant_inputs 0

module {
  func.func private @mlir_attention(
      %arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>,
      %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
      %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>,
      %scale_bias: !migraphx.shaped<1x7x7xf32, 49x7x1>)
      -> (!migraphx.shaped<1x7x3xf32, 21x3x1>)
      attributes {
        rock.kernel,
        rock.arch = "gfx942",
        rock.num_cu = 304 : i64
      } {
    %qk = migraphx.dot %arg0, %arg1
        : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1>
          -> <1x7x7xf32, 49x7x1>
    %scaled = migraphx.mul %qk, %scale_bias
        : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
    %biased = migraphx.add %scaled, %scale_bias
        : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1>
          -> <1x7x7xf32, 49x7x1>
    %softmax = migraphx.softmax %biased {axis = 2 : i64}
        : <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %result = migraphx.dot %softmax, %arg2
        : <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1>
          -> <1x7x3xf32, 21x3x1>
    return %result : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
