// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics \
// RUN:   --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable

// Verifies that a scaled gemm whose `kPerBlock` is not a multiple of
// `quantBlockSize` is classified as not-applicable (rather than a real
// compilation bug): the pass marks the enclosing module with
// `rock.not_applicable` before signalling failure so the tuning driver can
// distinguish "config doesn't fit" from a real compiler error.
//
// kPerBlock=4, quantBlockSize=8, so 4 % 8 != 0 triggers the divisibility check.
#bad_qb_params = #rock.gemm_params<
  kPerBlock = 4, kpack = 1, mPerBlock = 8, nPerBlock = 16,
  numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// NA-LABEL: 'func.func' operation: @gridwise_gemm_kperblock_not_multiple_of_quantblock
// NA: module attributes {rock.not_applicable
func.func @gridwise_gemm_kperblock_not_multiple_of_quantblock(
    %A: tensor<1x8x64xf4E2M1FN>, %B: tensor<1x64x16xf4E2M1FN>,
    %scaleA: tensor<1x8x8xf8E8M0FNU>, %scaleB: tensor<1x16x8xf8E8M0FNU>,
    %C: tensor<1x8x16xf32>) attributes {rock.arch = "##TOKEN_ARCH##",
                                        rock.block_size = 64 : i32,
                                        rock.grid_size = 1 : i32,
                                        rock.kernel} {
  // expected-error @+2 {{kPerBlock is not a multiple of quantBlockSize}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) {
    quantBlockSize = 8 : i64,
    params = #bad_qb_params
  } : tensor<1x8x64xf4E2M1FN>, tensor<1x64x16xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>, tensor<1x16x8xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}
