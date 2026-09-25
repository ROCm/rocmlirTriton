// A GEMM with dynamic M and N through the kernel pipeline: the padding maps
// and tile views are symbolic, the equalities between the dimensions of A, B
// and C are assumed, the grid size is an expression over the dimensions, and
// the dimensions end up as i32 kernel arguments.

// RUN: rocmlir-driver --kernel-pipeline=gpu --arch=gfx1100 %s -o /dev/null \
// RUN:   --mlir-print-ir-after=rock-gemm-to-gridwise \
// RUN:   --mlir-print-ir-after=rock-gridwise-gemm-to-blockwise 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=GRIDWISE,BLOCKWISE
// RUN: rocmlir-driver --kernel-pipeline=gpu --arch=gfx1100 %s | FileCheck %s --check-prefix=TTIR

// GRIDWISE-LABEL: IR Dump After RockGemmToGridwisePass
// GRIDWISE: func.func @rock_gemm(%[[A:.*]]: tensor<1x?x64xf16>, %[[B:.*]]: tensor<1x64x?xf16>, %[[C:.*]]: tensor<1x?x?xf16>)
// GRIDWISE-SAME: rock.grid_size = #rock.arg_expr<(s0 ceildiv 128) * (s1 ceildiv 64), [arg(0, 1), arg(1, 2)]>
// GRIDWISE: %[[MA:.*]] = tensor.dim %[[A]], %{{.*}} : tensor<1x?x64xf16>
// GRIDWISE: %[[MAI:.*]] = arith.index_cast %[[MA]] : index to i32
// GRIDWISE: %[[MC:.*]] = tensor.dim %[[C]], %{{.*}} : tensor<1x?x?xf16>
// GRIDWISE: %[[MCI:.*]] = arith.index_cast %[[MC]] : index to i32
// GRIDWISE: %[[EQM:.*]] = arith.cmpi eq, %[[MAI]], %[[MCI]] : i32
// GRIDWISE: llvm.intr.assume %[[EQM]] : i1
// GRIDWISE: %[[NB:.*]] = tensor.dim %[[B]], %{{.*}} : tensor<1x64x?xf16>
// GRIDWISE: %[[NC:.*]] = tensor.dim %[[C]], %{{.*}} : tensor<1x?x?xf16>
// GRIDWISE: llvm.intr.assume
// GRIDWISE: rock.transform %[[A]] {{.*}}<Pad{0, -s0 + (s0 ceildiv 128) * 128} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>{{.*}} symbols = [arg(0, 1)] bounds = [1, (s0 ceildiv 128) * 128, 64] -> [1, s0, 64]>
// GRIDWISE: rock.transform %[[B]] {{.*}}<Pad{0, -s0 + (s0 ceildiv 64) * 64} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>{{.*}} symbols = [arg(1, 2)]
// GRIDWISE: rock.transform %[[C]] {{.*}} symbols = [arg(2, 1), arg(2, 2)] bounds = [1, (s0 ceildiv 128) * 128, (s1 ceildiv 64) * 64] -> [1, s0, s1]>
// GRIDWISE: rock.gridwise_gemm
// GRIDWISE-SAME: splitKFactor = 1

// The tile views unmerge the padded dimensions into symbolic block counts, and
// the program id is decomposed with arithmetic on the dimension values.
// BLOCKWISE-LABEL: IR Dump After RockGridwiseGemmToBlockwisePass
// BLOCKWISE: %[[M:.*]] = arith.index_cast %{{.*}} : index to i32
// BLOCKWISE: llvm.intr.assume
// BLOCKWISE: %[[N:.*]] = arith.index_cast %{{.*}} : index to i32
// BLOCKWISE: llvm.intr.assume
// BLOCKWISE: arith.addi %[[M]], %c127_i32
// BLOCKWISE: arith.divui %{{.*}}, %c128_i32
// BLOCKWISE: arith.addi %[[N]], %c63_i32
// BLOCKWISE: arith.divui %{{.*}}, %c64_i32
// BLOCKWISE: <Unmerge{s0 ceildiv 128, 128} ["m_block", "m_iter"]
// BLOCKWISE-SAME: symbols = [arg(0, 1), arg(1, 2)]

// TTIR: module attributes {{{.*}}rock.dim_args.rock_gemm = [#rock.arg_dim<0, 1>, #rock.arg_dim<1, 2>, #rock.arg_dim<2, 1>, #rock.arg_dim<2, 2>]
// TTIR-SAME: rock.grid_size.rock_gemm = #rock.arg_expr<(s0 ceildiv 128) * (s1 ceildiv 64), [arg(0, 1), arg(1, 2)]>
// TTIR: tt.func @rock_gemm(%{{.*}}: !tt.ptr<f16> {{{.*}}}, %{{.*}}: !tt.ptr<f16> {{{.*}}}, %{{.*}}: !tt.ptr<f16> {{{.*}}}, %[[M:.*]]: i32, %[[N:.*]]: i32, %[[MC:.*]]: i32, %[[NC:.*]]: i32)
// TTIR-NOT: tensor.dim
// TTIR: arith.cmpi eq, %[[M]], %[[MC]] : i32
// TTIR: llvm.intr.assume
// TTIR: arith.cmpi eq, %[[N]], %[[NC]] : i32
// TTIR: llvm.intr.assume
// TTIR: tt.return

module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<1x?x64xf16>, %arg1: tensor<1x64x?xf16>, %arg2: tensor<1x?x?xf16>) -> tensor<1x?x?xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 48 : i64} {
    %0 = rock.gemm %arg0 * %arg1 : tensor<1x?x64xf16> * tensor<1x64x?xf16> -> tensor<1x?x?xf16>
    %1 = rock.store %0 to %arg2 by set : tensor<1x?x?xf16> -> tensor<1x?x?xf16> to tensor<1x?x?xf16>
    return %1 : tensor<1x?x?xf16>
  }
}
