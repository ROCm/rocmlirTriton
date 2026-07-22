// RUN: rocmlir-opt --rock-set-attention-bias-load-layout="num-stages=1" --mlir-print-local-scope --split-input-file %s | FileCheck %s --check-prefix=SAFE
// RUN: rocmlir-opt --rock-set-attention-bias-load-layout="num-stages=2" --mlir-print-local-scope --split-input-file %s | FileCheck %s --check-prefix=SAFE
// RUN: rocmlir-opt --rock-set-attention-bias-load-layout="num-stages=3" --mlir-print-local-scope --split-input-file %s | FileCheck %s --check-prefix=STAGE3

#bias_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 2], order = [1, 0]}>
#score_blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @safe_transposed_bias
  // SAFE: tt.load
  // SAFE-SAME: amdg.bypass_lds_load = true
  // SAFE-SAME: rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
  // SAFE-SAME: tensor<128x256x!tt.ptr<f32>, #ttg.amd_mfma
  // A connected non-candidate input is rematerialized in its actual post-dot
  // consumer layout, not in a hard-coded layout.
  // SAFE: tt.load
  // SAFE-SAME: rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 1, role = bias, orientation = unknown>
  // SAFE-SAME: tensor<128x256x!tt.ptr<f32>, #ttg.blocked
  // SAFE-SAME: warpsPerCTA = [2, 1]
  // STAGE3-LABEL: tt.func @safe_transposed_bias
  // STAGE3: tt.load
  // STAGE3-SAME: tensor<128x256x!tt.ptr<f32>, #ttg.blocked
  // STAGE3-SAME: warpsPerCTA = [1, 2]
  // STAGE3: tt.load
  // STAGE3-SAME: tensor<128x256x!tt.ptr<f32>, #ttg.blocked
  // STAGE3-SAME: warpsPerCTA = [1, 2]
  tt.func @safe_transposed_bias(
      %ptr: tensor<128x256x!tt.ptr<f32>, #bias_blocked>,
      %other_ptr: tensor<128x256x!tt.ptr<f32>, #bias_blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<128x256xf32, #score_blocked> {
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x256x!tt.ptr<f32>, #bias_blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<128x256xf32, #bias_blocked> -> tensor<128x256xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x256xf32, #mma>
    %other_bias = tt.load %other_ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 1, role = bias, orientation = unknown>
    } : tensor<128x256x!tt.ptr<f32>, #bias_blocked>
    %score_dist = ttg.convert_layout %score
        : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #score_blocked>
    %other_dist = ttg.convert_layout %other_bias
        : tensor<128x256xf32, #bias_blocked> -> tensor<128x256xf32, #score_blocked>
    %result = arith.addf %score_dist, %other_dist
        : tensor<128x256xf32, #score_blocked>
    tt.return %result : tensor<128x256xf32, #score_blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @safe_mfma16
  // SAFE: tt.load
  // SAFE-SAME: tensor<64x16x!tt.ptr<f32>, #ttg.amd_mfma
  // STAGE3-LABEL: tt.func @safe_mfma16
  // STAGE3: tt.load
  // STAGE3-SAME: tensor<64x16x!tt.ptr<f32>, #ttg.blocked
  tt.func @safe_mfma16(
      %ptr: tensor<64x16x!tt.ptr<f32>, #blocked>,
      %a: tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<64x16xf32, #mma> {
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<64x16x!tt.ptr<f32>, #blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<64x16xf32, #mma>
    tt.return %score : tensor<64x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @remaining_mask_rejected
  // SAFE: tt.load
  // SAFE-SAME: tensor<128x256x!tt.ptr<f32>, #ttg.blocked
  tt.func @remaining_mask_rejected(
      %ptr: tensor<128x256x!tt.ptr<f32>, #blocked>,
      %mask: tensor<128x256xi1, #blocked>,
      %other: tensor<128x256xf32, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<128x256xf32, #mma> {
    %bias = tt.load %ptr, %mask, %other {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x256x!tt.ptr<f32>, #blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<128x256xf32, #blocked> -> tensor<128x256xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x256xf32, #mma>
    tt.return %score : tensor<128x256xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @unsupported_arch_rejected
  // SAFE: tt.load
  // SAFE-SAME: tensor<128x256x!tt.ptr<f32>, #ttg.blocked
  tt.func @unsupported_arch_rejected(
      %ptr: tensor<128x256x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<128x256xf32, #mma> {
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x256x!tt.ptr<f32>, #blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<128x256xf32, #blocked> -> tensor<128x256xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x256xf32, #mma>
    tt.return %score : tensor<128x256xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @cross_warp_layout_rejected
  // SAFE: tt.load
  // SAFE-SAME: tensor<128x32x!tt.ptr<f32>, #ttg.blocked
  tt.func @cross_warp_layout_rejected(
      %ptr: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>,
      %c: tensor<128x32xf32, #mma>)
      -> (tensor<128x32xf32, #mma>, tensor<128x32xf32, #blocked>) {
    %score = tt.dot %a, %b, %c {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    tt.return %score, %bias
        : tensor<128x32xf32, #mma>, tensor<128x32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @non_candidate_metadata_rejected
  // SAFE: tt.load
  // SAFE-SAME: role = bias, orientation = natural
  // SAFE-SAME: #ttg.blocked
  // SAFE: tt.load
  // SAFE-SAME: role = scale, orientation = transposed
  // SAFE-SAME: #ttg.blocked
  // SAFE: tt.load
  // SAFE-SAME: role = bias, orientation = unknown
  // SAFE-SAME: #ttg.blocked
  tt.func @non_candidate_metadata_rejected(
      %ptr0: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %ptr1: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %ptr2: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>,
      %c: tensor<128x32xf32, #mma>)
      -> (tensor<128x32xf32, #mma>, tensor<128x32xf32, #blocked>,
          tensor<128x32xf32, #blocked>, tensor<128x32xf32, #blocked>) {
    %score = tt.dot %a, %b, %c {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    %natural_bias = tt.load %ptr0 {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = natural>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    %transposed_scale = tt.load %ptr1 {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 1, role = scale, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    %unknown_bias = tt.load %ptr2 {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 2, role = bias, orientation = unknown>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    tt.return %score, %natural_bias, %transposed_scale, %unknown_bias
        : tensor<128x32xf32, #mma>, tensor<128x32xf32, #blocked>,
          tensor<128x32xf32, #blocked>, tensor<128x32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @duplicate_input_leaf_rejected
  // SAFE: tt.load
  // SAFE-SAME: #ttg.blocked
  // SAFE: tt.load
  // SAFE-SAME: #ttg.blocked
  tt.func @duplicate_input_leaf_rejected(
      %ptr0: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %ptr1: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>,
      %c: tensor<128x32xf32, #mma>)
      -> (tensor<128x32xf32, #mma>, tensor<128x32xf32, #blocked>,
          tensor<128x32xf32, #blocked>) {
    %score = tt.dot %a, %b, %c {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    %bias0 = tt.load %ptr0 {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    %bias1 = tt.load %ptr1 {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    tt.return %score, %bias0, %bias1
        : tensor<128x32xf32, #mma>, tensor<128x32xf32, #blocked>,
          tensor<128x32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @duplicate_score_group_rejected
  // SAFE: tt.load
  // SAFE-SAME: #ttg.blocked
  tt.func @duplicate_score_group_rejected(
      %ptr: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>,
      %c: tensor<128x32xf32, #mma>)
      -> (tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma>,
          tensor<128x32xf32, #blocked>) {
    %score0 = tt.dot %a, %b, %c {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    %score1 = tt.dot %a, %b, %c {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    tt.return %score0, %score1, %bias
        : tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma>,
          tensor<128x32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @two_equal_shape_attention_groups
  // SAFE: tt.load
  // SAFE-SAME: groupId = 0
  // SAFE-SAME: #ttg.amd_mfma
  // SAFE: tt.load
  // SAFE-SAME: groupId = 1
  // SAFE-SAME: #ttg.amd_mfma
  tt.func @two_equal_shape_attention_groups(
      %ptr0: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %ptr1: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> (tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma>) {
    %bias0 = tt.load %ptr0 {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    %bias0_acc = ttg.convert_layout %bias0
        : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #mma>
    %score0 = tt.dot %a, %b, %bias0_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    %bias1 = tt.load %ptr1 {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 1, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    %bias1_acc = ttg.convert_layout %bias1
        : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #mma>
    %score1 = tt.dot %a, %b, %bias1_acc {
      rock.attention_group = #rock.attention_group<1>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    tt.return %score0, %score1
        : tensor<128x32xf32, #mma>, tensor<128x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @one_warp_rejected
  // SAFE: tt.load
  // SAFE-SAME: #ttg.blocked
  tt.func @one_warp_rejected(
      %ptr: tensor<32x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<32x32xf32, #mma> {
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<32x32x!tt.ptr<f32>, #blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<32x32xf32, #mma>
    tt.return %score : tensor<32x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [16, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [16, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @sixteen_warps_rejected
  // SAFE: tt.load
  // SAFE-SAME: #ttg.blocked
  tt.func @sixteen_warps_rejected(
      %ptr: tensor<512x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<512x32xf32, #mma> {
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<512x32x!tt.ptr<f32>, #blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<512x32xf32, #blocked> -> tensor<512x32xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<512x32xf32, #mma>
    tt.return %score : tensor<512x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @mfma_v4_rejected
  // SAFE: tt.load
  // SAFE-SAME: tensor<128x32x!tt.ptr<f32>, #ttg.blocked
  tt.func @mfma_v4_rejected(
      %ptr: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<128x32xf32, #mma> {
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 0, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    tt.return %score : tensor<128x32xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // SAFE-LABEL: tt.func @mismatched_group_rejected
  // SAFE: tt.load
  // SAFE-SAME: tensor<128x32x!tt.ptr<f32>, #ttg.blocked
  tt.func @mismatched_group_rejected(
      %ptr: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %a: tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
      %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)
      -> tensor<128x32xf32, #mma> {
    %bias = tt.load %ptr {
      rock.pre_softmax_input = #rock.pre_softmax_input<groupId = 1, inputIndex = 0, role = bias, orientation = transposed>
    } : tensor<128x32x!tt.ptr<f32>, #blocked>
    %bias_acc = ttg.convert_layout %bias
        : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #mma>
    %score = tt.dot %a, %b, %bias_acc {
      rock.attention_group = #rock.attention_group<0>
    } : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      -> tensor<128x32xf32, #mma>
    tt.return %score : tensor<128x32xf32, #mma>
  }
}
