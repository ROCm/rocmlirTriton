// A causal attention with dynamic G and sequence lengths through the kernel
// pipeline: the sequence lengths are padded by symbolic maps and recorded as
// #rock.arg_expr pre-padding lengths, the grid size is an expression over the
// dimensions, and the padding and causal masks compare the tile indices with
// the dimension values.

// RUN: rocmlir-driver --kernel-pipeline=gpu --arch=gfx1100 %s -o /dev/null \
// RUN:   --mlir-print-ir-after=rock-attn-to-gridwise \
// RUN:   --mlir-print-ir-after=rock-gridwise-attn-to-blockwise 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=GRIDWISE,BLOCKWISE
// RUN: rocmlir-driver --kernel-pipeline=gpu --arch=gfx1100 %s | FileCheck %s --check-prefix=TTIR

// GRIDWISE-LABEL: IR Dump After RockAttnToGridwisePass
// GRIDWISE: func.func @rock_attention(%[[Q:.*]]: tensor<?x?x32xf16>, %[[K:.*]]: tensor<?x32x?xf16>, %[[V:.*]]: tensor<?x?x32xf16>, %[[O:.*]]: tensor<?x?x32xf16>)
// GRIDWISE-SAME: rock.grid_size = #rock.arg_expr<(s1 ceildiv 64) * s0, [arg(0, 0), arg(0, 1)]>
// GRIDWISE: %[[SEQKK:.*]] = tensor.dim %[[K]], %{{.*}} : tensor<?x32x?xf16>
// GRIDWISE: %[[SEQKV:.*]] = tensor.dim %[[V]], %{{.*}} : tensor<?x?x32xf16>
// GRIDWISE: llvm.intr.assume
// GRIDWISE: rock.transform %[[Q]] {{.*}}<Pad{0, -s1 + (s1 ceildiv 64) * 64} ["gemm0MPad"] at [1] -> ["gemm0M"] at [1]>{{.*}} symbols = [arg(0, 0), arg(0, 1)]
// GRIDWISE: rock.transform %[[K]] {{.*}}<Pad{0, -s1 + (s1 ceildiv 32) * 32} ["gemm0NPad"] at [2] -> ["gemm0N"] at [2]>{{.*}} symbols = [arg(1, 0), arg(1, 2)]
// GRIDWISE: rock.transform %[[V]] {{.*}}<Pad{0, -s1 + (s1 ceildiv 32) * 32} ["gemm1KPad"] at [1] -> ["gemm1K"] at [1]>
// GRIDWISE: rock.transform %[[O]] {{.*}}<Pad{0, -s1 + (s1 ceildiv 64) * 64} ["gemm1MPad"] at [1] -> ["gemm1M"] at [1]>
// GRIDWISE: rock.gridwise_attention
// GRIDWISE: {causal
// GRIDWISE-SAME: prePadG0M = #rock.arg_expr<s0, [arg(0, 1)]>, prePadG0N = #rock.arg_expr<s0, [arg(1, 2)]>

// The N loop runs to min(causal bound, number of N blocks), and inside it the
// columns past seq_k and above the diagonal are set to -inf.
// BLOCKWISE-LABEL: IR Dump After RockGridwiseAttnToBlockwisePass
// BLOCKWISE: %[[KDIM:.*]] = tensor.dim %{{.*}}, %c2 : tensor<?x32x?xf16>
// BLOCKWISE-NEXT: %[[SEQK:.*]] = arith.index_cast %[[KDIM]] : index to i32
// BLOCKWISE: %[[NBLOCKSUM:.*]] = arith.addi %[[SEQK]], %c31_i32 : i32
// BLOCKWISE: %[[NBLOCKS:.*]] = arith.divui %[[NBLOCKSUM]], %{{.*}} : i32
// BLOCKWISE: %[[END:.*]] = arith.minui %{{.*}}, %[[NBLOCKS]] : i32
// BLOCKWISE: scf.for %[[NITER:.*]] = %{{.*}} to %[[END]]
// BLOCKWISE: rock.blockwise_gemm
// BLOCKWISE: tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// BLOCKWISE: arith.muli %[[NITER]], %{{.*}} : i32
// BLOCKWISE: %[[COLS:.*]] = tt.broadcast %{{.*}} : tensor<1x32xi32> -> tensor<64x32xi32>
// BLOCKWISE: %[[SEQKSPLAT:.*]] = tt.splat %[[SEQK]] : i32 -> tensor<64x32xi32>
// BLOCKWISE: %[[PAD:.*]] = arith.cmpi uge, %[[COLS]], %[[SEQKSPLAT]] : tensor<64x32xi32>
// BLOCKWISE: tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// BLOCKWISE: %[[ROWS:.*]] = tt.broadcast %{{.*}} : tensor<64x1xi32> -> tensor<64x32xi32>
// BLOCKWISE: %[[CAUSAL:.*]] = arith.cmpi ugt, %[[COLS]], %[[ROWS]] : tensor<64x32xi32>
// BLOCKWISE: %[[MASK:.*]] = arith.ori %[[PAD]], %[[CAUSAL]] : tensor<64x32xi1>
// BLOCKWISE: arith.select %[[MASK]], %{{.*}}, %{{.*}} : tensor<64x32xi1>, tensor<64x32xf32>
// BLOCKWISE: rock.blockwise_reduce max
// BLOCKWISE: rock.store_marker {{.*}} symbols = [arg(0, 0), arg(0, 1)]

// TTIR: module attributes {{{.*}}rock.dim_args.rock_attention = [#rock.arg_dim<0, 0>, #rock.arg_dim<0, 1>, #rock.arg_dim<1, 0>, #rock.arg_dim<1, 2>, #rock.arg_dim<2, 0>, #rock.arg_dim<2, 1>, #rock.arg_dim<3, 0>, #rock.arg_dim<3, 1>]
// TTIR-SAME: rock.grid_size.rock_attention = #rock.arg_expr<(s1 ceildiv 64) * s0, [arg(0, 0), arg(0, 1)]>
// TTIR: tt.func @rock_attention({{.*}}: !tt.ptr<f16> {{{.*}}}, {{.*}}: !tt.ptr<f16> {{{.*}}}, {{.*}}: !tt.ptr<f16> {{{.*}}}, {{.*}}: !tt.ptr<f16> {{{.*}}}, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32)
// TTIR-NOT: tensor.dim
// TTIR: tt.return

module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_attention(%arg0: tensor<?x?x32xf16>, %arg1: tensor<?x32x?xf16>, %arg2: tensor<?x?x32xf16>, %arg3: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 48 : i64} {
    %0 = rock.attention{
      qk = %arg0 * %arg1 : tensor<?x?x32xf16>, tensor<?x32x?xf16>
      softmax(qk) * %arg2 : tensor<?x?x32xf16>
    } {causal, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, perf_config = "attn:v1:64,32,32,1,1,4,0,1,1,0,0"} -> tensor<?x?x32xf16>
    %1 = rock.store %0 to %arg3 by set : tensor<?x?x32xf16> -> tensor<?x?x32xf16> to tensor<?x?x32xf16>
    return %1 : tensor<?x?x32xf16>
  }
}
