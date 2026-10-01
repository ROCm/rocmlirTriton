// A decode-shaped attention with a static seq_q of 1, dynamic G and seq_k,
// GQA, split-KV, a KV cache and a prefix offset through the kernel pipeline.

// RUN: rocmlir-driver --kernel-pipeline=gpu --arch=gfx1100 %s -o /dev/null \
// RUN:   --mlir-print-ir-after=rock-attn-to-gridwise \
// RUN:   --mlir-print-ir-after=rock-gridwise-attn-to-blockwise 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=GRIDWISE,BLOCKWISE
// RUN: rocmlir-driver --kernel-pipeline=gpu --arch=gfx1100 %s | FileCheck %s --check-prefix=TTIR

// GQA folds the 2 query heads of each KV head into seq_q, so Q, the output
// and the LSE are viewed over the batch of K, the output and LSE with the
// splits folded into it, and the KV cache and prefix of KV head g are those of
// query head 2 * g. The batch of Q is twice that of K, so the only assumes
// are between K and V.
// GRIDWISE-LABEL: IR Dump After RockAttnToGridwisePass
// GRIDWISE: func.func @rock_attention
// GRIDWISE-SAME: rock.grid_size = #rock.arg_expr<s0 * 2, [arg(1, 0)]>
// GRIDWISE: tensor.dim %arg1, %c0
// GRIDWISE: tensor.dim %arg2, %c0
// GRIDWISE: llvm.intr.assume
// GRIDWISE: tensor.dim %arg1, %c2
// GRIDWISE: tensor.dim %arg2, %c1
// GRIDWISE: llvm.intr.assume
// GRIDWISE-NOT: llvm.intr.assume
// GRIDWISE: rock.transform %arg0 by {{.*}}<Unmerge{s0, 2, 1} ["gemmG", "numRepeats", "splitKV"] at [0, 3, 1] -> ["gemmG"] at [0]>{{.*}} symbols = [arg(1, 0)] bounds = [s0, 1, 1, 2, 32] -> [s0 * 2, 1, 32]>
// GRIDWISE: rock.transform %arg3 by {{.*}}<Embed{2} ["gemmG"] at [0] -> ["gemmG"] at [0]>] symbols = [arg(1, 0), arg(3, 0)] bounds = [s0] -> [s1]>
// GRIDWISE: rock.transform %arg4 by {{.*}}<Embed{2} ["gemmG"] at [0] -> ["gemmG"] at [0]>] symbols = [arg(1, 0), arg(4, 0)] bounds = [s0] -> [s1]>
// GRIDWISE: rock.transform %arg6 by {{.*}}<Unmerge{s0, 2, 2} ["gemmG", "numRepeats", "splitKV"] at [0, 3, 1] -> ["gemmG"] at [0]>{{.*}} bounds = [s0, 2, 1, 2, 32] -> [s0 * 4, 1, 32]>
// GRIDWISE: rock.transform %{{.*}} by {{.*}}<Merge{s0, 2} ["gemmG"] at [0] -> ["gemmG", "splitKV"] at [0, 1]>{{.*}} bounds = [s0 * 2, 2, 32] -> [s0, 2, 1, 2, 32]>
// GRIDWISE: rock.transform %arg5 by {{.*}}<Unmerge{s0, 2, 2} ["gemmG", "numRepeats", "splitKV"] at [0, 3, 1] -> ["gemmG"] at [0]>{{.*}} bounds = [s0, 2, 1, 2] -> [s0 * 4, 1]>
// GRIDWISE: rock.gridwise_attention
// GRIDWISE: {causal, numRepeatsGQA = 2 : index
// GRIDWISE-SAME: prePadG0M = 2 : index, prePadG0N = #rock.arg_expr<s0, [arg(1, 2)]>
// GRIDWISE-SAME: splitKV = 2 : i32

// Each split runs its share of the key blocks up to the KV-cache and prefix
// bounds and the number of key blocks, and is skipped when that is empty.
// Inside, the columns past seq_k, past the KV cache and past the query (row
// / 2) plus the prefix are set to -inf.
// BLOCKWISE-LABEL: IR Dump After RockGridwiseAttnToBlockwisePass
// BLOCKWISE: %[[KV:.*]] = tt.unsplat %{{.*}} : tensor<1xi32>
// BLOCKWISE: %[[PREFIX:.*]] = tt.unsplat %{{.*}} : tensor<1xi32>
// BLOCKWISE: arith.minui %[[KV]], %{{.*}} : i32
// BLOCKWISE: %[[WORK:.*]] = arith.cmpi ugt, %[[END:.*]], %[[START:.*]] : i32
// BLOCKWISE: scf.if %[[WORK]]
// BLOCKWISE: scf.for %{{.*}} = %[[START]] to %[[END]]
// BLOCKWISE: %[[PAD:.*]] = arith.cmpi uge, %[[COLS:.*]], %{{.*}} : tensor<64x32xi32>
// BLOCKWISE: %[[KVSPLAT:.*]] = tt.splat %[[KV]] : i32 -> tensor<64x32xi32>
// BLOCKWISE: %[[PASTKV:.*]] = arith.cmpi ugt, %[[COLS]], %[[KVSPLAT]] : tensor<64x32xi32>
// BLOCKWISE: arith.ori %[[PAD]], %[[PASTKV]] : tensor<64x32xi1>
// BLOCKWISE: %[[QUERY:.*]] = arith.divui %{{.*}}, %{{.*}} : tensor<64x32xi32>
// BLOCKWISE: %[[PREFIXSPLAT:.*]] = tt.splat %[[PREFIX]] : i32 -> tensor<64x32xi32>
// BLOCKWISE: %[[BOUND:.*]] = arith.addi %[[QUERY]], %[[PREFIXSPLAT]] : tensor<64x32xi32>
// BLOCKWISE: arith.cmpi ugt, %[[COLS]], %[[BOUND]] : tensor<64x32xi32>

// TTIR: tt.func @rock_attention(
// TTIR-NOT: tensor.dim
// TTIR: tt.return

module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_attention(%arg0: tensor<?x1x32xf16>, %arg1: tensor<?x32x?xf16>, %arg2: tensor<?x?x32xf16>, %arg3: tensor<?xi32>, %arg4: tensor<?xi32>, %arg5: tensor<?x1xf16>, %arg6: tensor<?x1x32xf16>) -> (tensor<?x1x32xf16>, tensor<?x1xf16>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 48 : i64} {
    %result, %lse = rock.attention{
      qk = %arg0 * %arg1 : tensor<?x1x32xf16>, tensor<?x32x?xf16>
      lastValidKVIndex = (%arg3 : tensor<?xi32>)
      prefixOffset = (%arg4 : tensor<?xi32>)
      causal
      softmax(qk) * %arg2 : tensor<?x?x32xf16>
    } {numHeadsKV = 2 : i32, numHeadsQ = 4 : i32, softmaxType = f32, splitKV = 2 : i32, perf_config = "attn:v1:64,32,32,1,1,4,0,1,1,0,0"} -> tensor<?x1x32xf16>, tensor<?x1xf16>
    %0 = rock.store %result to %arg6 by set : tensor<?x1x32xf16> -> tensor<?x1x32xf16> to tensor<?x1x32xf16>
    %1 = rock.store %lse to %arg5 by set : tensor<?x1xf16> -> tensor<?x1xf16> to tensor<?x1xf16>
    return %0, %1 : tensor<?x1x32xf16>, tensor<?x1xf16>
  }
}
