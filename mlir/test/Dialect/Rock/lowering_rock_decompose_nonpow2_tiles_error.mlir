// Error tests for the rock-decompose-nonpow2-tiles pass.
//
// These are minimal, hand-written reductions of the real (pre-pass) IR that
// exercise each diagnostic the pass emits.

// RUN: rocmlir-opt -rock-decompose-nonpow2-tiles -verify-diagnostics -split-input-file -mlir-print-local-scope %s

// ============================================================
// Error: non-power-of-two K. M/N are non-power-of-two (so the
// GEMM is selected for splitting), but the contraction dim is
// never split, and a non-power-of-two K cannot be handled.
// ============================================================

func.func @test_nonpow2_k(%a: tensor<80x48xf16>, %b: tensor<48x80xf16>) attributes {rock.kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  // expected-error @+1 {{non-power-of-two K is not supported}}
  %0 = rock.blockwise_gemm(%a, %b, %cst) : tensor<80x48xf16>, tensor<48x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
  return
}

// -----

// ============================================================
// Error: scaled GEMMs are not supported.
// ============================================================

func.func @test_scaled_gemm(%a: tensor<80x16xf4E2M1FN>, %b: tensor<16x80xf4E2M1FN>,
    %sa: tensor<80x1xf8E8M0FNU>, %sb: tensor<80x1xf8E8M0FNU>) attributes {rock.kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  // expected-error @+1 {{scaled blockwise_gemm not supported}}
  %0 = rock.blockwise_gemm(%a scaled by %sa, %b scaled by %sb, %cst) {quantBlockSize = 16 : i64} : tensor<80x16xf4E2M1FN> scaled by tensor<80x1xf8E8M0FNU>, tensor<16x80xf4E2M1FN> scaled by tensor<80x1xf8E8M0FNU>, tensor<80x80xf32> -> tensor<80x80xf32>
  return
}

// -----

// ============================================================
// Error: the GEMM is inside an scf.for that is not its
// accumulator K-loop (matrixC is not the loop iter_arg).
// ============================================================

func.func @test_loop_not_accumulator(%a: tensor<80x16xf16>, %b: tensor<16x80xf16>) attributes {rock.kernel} {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> (tensor<80x80xf32>) : i32 {
    // expected-error @+1 {{GEMM is inside a loop that is not its accumulator K-loop}}
    %0 = rock.blockwise_gemm(%a, %b, %cst) : tensor<80x16xf16>, tensor<16x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
    scf.yield %acc : tensor<80x80xf32>
  }
  return
}

// -----

// ============================================================
// Error: the GEMM accumulates over the loop iter_arg, but the
// loop does not yield the GEMM result.
// ============================================================

func.func @test_loop_bad_yield(%a: tensor<80x16xf16>, %b: tensor<16x80xf16>) attributes {rock.kernel} {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> (tensor<80x80xf32>) : i32 {
    // expected-error @+1 {{loop does not yield the GEMM result}}
    %0 = rock.blockwise_gemm(%a, %b, %acc) : tensor<80x16xf16>, tensor<16x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
    scf.yield %acc : tensor<80x80xf32>
  }
  return
}

// -----

// ============================================================
// Error: the GEMM result has no blockwise_store sink.
// ============================================================

func.func @test_no_store(%a: tensor<80x16xf16>, %b: tensor<16x80xf16>) attributes {rock.kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  // The GEMM result is unused: no blockwise_store sink is reachable.
  // expected-error @+1 {{no blockwise_store for GEMM result}}
  %0 = rock.blockwise_gemm(%a, %b, %cst) : tensor<80x16xf16>, tensor<16x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
  return
}

// -----

// ============================================================
// Error: the GEMM result reaches an op that is neither a fusion
// op nor a blockwise_store (here, func.return).
// ============================================================

func.func @test_unsupported_use(%a: tensor<80x16xf16>, %b: tensor<16x80xf16>) -> tensor<80x80xf32> attributes {rock.kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  // expected-error @+1 {{unsupported use of GEMM result}}
  %0 = rock.blockwise_gemm(%a, %b, %cst) : tensor<80x16xf16>, tensor<16x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
  return %0 : tensor<80x80xf32>
}
