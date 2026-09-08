// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-gridwise-gemm-to-blockwise | FileCheck %s

// Lowering a gemm whose M is only known at run time. The number of m blocks is
// then unknown too, so it has to survive as `?` in every view derived from it
// and must not appear in the block id mapping.
//
// G = 2 and N = 128 with nPerBlock = 64 give gBlocks = 2 and nBlocks = 2, so
// the two divisors of the mapping below are distinct.

#params = #rock.gemm_params<
  mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numCTAs = 1,
  numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0>

// The tile views keep the m block count unknown: A splits m into an unknown
// number of blocks of 64, B does not depend on m and so ignores an unknown
// number of them, and the output is written back through the same split.
// CHECK-DAG: #[[$A_VIEW:.*]] = #rock.transform_map<{{.*}}<Unmerge{?, 64} ["m_block", "m_iter"]{{.*}}bounds = [8, 2, ?, 2, 64, 16] -> [2, ?, 128]>
// CHECK-DAG: #[[$B_VIEW:.*]] = #rock.transform_map<{{.*}}<AddDim{?} ["m_block"]{{.*}}bounds = [8, 2, ?, 2, 16, 64] -> [2, 128, 128]>
// CHECK-DAG: #[[$C_VIEW:.*]] = #rock.transform_map<{{.*}}<Unmerge{?, 64} ["m_block", "m_iter"]{{.*}}bounds = [2, ?, 2, 64, 64] -> [2, ?, 128]>

// CHECK-LABEL: func.func @gridwise_gemm_dynamic_m
// CHECK-SAME: (%[[A:.*]]: tensor<2x?x128xf32>, %[[B:.*]]: tensor<2x128x128xf32>, %[[C:.*]]: tensor<2x?x128xf32>)

// m is the most significant grid coordinate, so it is peeled off with the
// static gBlocks * nBlocks = 4 and the unknown m block count never appears.
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: %[[MBLOCK:.*]] = arith.divui %[[BID]], %[[C4]]
// CHECK: %[[GN:.*]] = arith.remui %[[BID]], %[[C4]]
// CHECK: %[[GBLOCK:.*]] = arith.divui %[[GN]], %[[C2]]
// CHECK: %[[NBLOCK:.*]] = arith.remui %[[GN]], %[[C2]]

// K stays static, so the loop trip count is still 128 / 16 = 8.
// CHECK: %[[KITERS:.*]] = arith.constant 8 : i32
// CHECK: scf.for %[[KLOOP:.*]] = %{{.*}} to %[[KITERS]] step
// CHECK-DAG: rock.load_marker %[[B]] views [#[[$B_VIEW]]][%[[KLOOP]], %[[GBLOCK]], %[[MBLOCK]], %[[NBLOCK]]]
// CHECK-DAG: rock.load_marker %[[A]] views [#[[$A_VIEW]]][%[[KLOOP]], %[[GBLOCK]], %[[MBLOCK]], %[[NBLOCK]]]
// CHECK: rock.blockwise_gemm
// CHECK: rock.store_marker %{{.*}} views [#[[$C_VIEW]]][%[[GBLOCK]], %[[MBLOCK]], %[[NBLOCK]]]
func.func @gridwise_gemm_dynamic_m(%A: tensor<2x?x128xf32>, %B: tensor<2x128x128xf32>,
                                   %C: tensor<2x?x128xf32>)
    attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 256 : i32,
                rock.kernel} {
  %0 = rock.gridwise_gemm(%A, %B) {params = #params}
    : tensor<2x?x128xf32>, tensor<2x128x128xf32> -> tensor<2x?x128xf32>
  %1 = rock.store %0 to %C by set
    : tensor<2x?x128xf32> -> tensor<2x?x128xf32> to tensor<2x?x128xf32>
  func.return
}
