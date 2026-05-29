// RUN: rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: tensor<1x384x64xf32>, %[[K:.+]]: tensor<1x64x384xf32>, %[[V:.+]]: tensor<1x384x64xf32>)
// CHECK-DAG: %[[ln2Recip:.+]] = arith.constant dense<1.44269502>
// CHECK-DAG: %[[negInf:.+]] = arith.constant dense<0xFF800000>
// CHECK-DAG: %[[zeroF32:.+]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG: %[[zeroVec:.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
// CHECK-DAG: %[[zeroAttn:.+]] = arith.constant dense<0.000000e+00> : tensor<16x64xf32>

// Loop-bound constants derived from tile params + input shapes:
//   c12 = gemmN(=384) / nPerBlock(=32)  -> outer N-tile loop count
//   c4  = gemmK(=64)  / kPerBlock(=16)  -> inner KpacksPerBlock loop count
//   c24 = grid_size attribute            -> used for block-coord divui/remui
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : i32
// CHECK-DAG: %[[c12:.+]] = arith.constant 12 : i32
// CHECK-DAG: %[[c24:.+]] = arith.constant 24 : i32

// Block-id derivation
// (replaces old rock.workgroup_id / rock.workitem_id + arith.remui/divui chain)
// CHECK: %[[PID:.+]] = tt.get_program_id x : i32

// init maxRow buffer / init sumRow buffer / init attentionAcc buffer
// (in new IR these are iter_args carried by the outer scf.for; the rock.fills
//  are replaced by initializing iter_args with the corresponding dense<> constants)

// Outer N-tile loop
// CHECK: %[[OUT:.+]]:3 = scf.for %{{.*}} = %{{.*}} to %[[c12]] step %{{.*}} iter_args(%{{.*}} = %[[zeroAttn]], %{{.*}} = %[[negInf]], %{{.*}} = %[[zeroF32]])

  // Per-iteration block-M / block-N coordinates from PID
  // CHECK: arith.divui %[[PID]], %[[c24]] : i32
  // CHECK: arith.remui %[[PID]], %[[c24]] : i32

  // init gemm0AccBuf (inner iter_args carries gemm0 partial sum)
  // (LDS allocations / lds_barrier / rock.stage / pipeline attribute are deferred to later passes)

  // Inner gemm0 KpacksPerBlock loop (bound = c4)
  // CHECK: %[[GEMM0:.+]] = scf.for %{{.*}} = %{{.*}} to %[[c4]] step %{{.*}} iter_args(%[[gemm0AccArg:.+]] = %[[zeroVec]])

    // Emit blockwise gemm0
    // (rock.transform on Q / K is encoded in the load_marker's `views` attribute instead
    //  of being an explicit rock.transform SSA value)
    // CHECK: rock.load_marker %[[Q]] views [#{{.*}}]
    // CHECK: rock.load_marker %[[K]] views [#{{.*}}]
    // CHECK: rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[gemm0AccArg]])

  // Scale gemm0 by 1/ln2 (was linalg.generic with arith.mulf %in, %ln2Recip)
  // CHECK: arith.mulf %[[GEMM0]], %[[ln2Recip]] : tensor<16x32xf32>

  // CHECK: %[[gemm0Max:.+]] = rock.blockwise_reduce max

  // Compute exp(gemm0 - rowmax_j)
  // *****************************
  // (broadcasting of newmax to the 2D gemm0 tile is now done with explicit
  //  tt.expand_dims + tt.broadcast; old blockwise_broadcast_reduce did it implicitly)
  // CHECK-DAG: %[[newmax:.+]] = arith.maximumf %{{.*}}, %[[gemm0Max]] : tensor<16xf32>
  // CHECK-DAG: tt.expand_dims %[[newmax]]
  // CHECK-DAG: tt.broadcast %{{.*}} : tensor<16x1xf32> -> tensor<16x32xf32>
  // CHECK-DAG: arith.subf %{{.*}}, %{{.*}} : tensor<16x32xf32>
  // CHECK-DAG: %[[gemm0NormExp:.+]] = math.exp2 %{{.*}} : tensor<16x32xf32>

  // CHECK: %[[gemm0NormExpSum:.+]] = rock.blockwise_reduce sum %[[gemm0NormExp]]

  // li = exp(m_{j-1} - m_{j}) * l_{j-1} + rowsum(Pij)
  // where
  // l is the rowsum accumulator
  // m is the rowmax accumulator
  // P is exp(gemm0 - rowmax_j)
  // *************************************************
  // CHECK-DAG: %[[maxdiff:.+]] = arith.subf %{{.*}}, %{{.*}} : tensor<16xf32>
  // CHECK-DAG: %[[maxdiffexp:.+]] = math.exp2 %[[maxdiff]] : tensor<16xf32>
  // CHECK-DAG: %[[rowsummul:.+]] = arith.mulf %[[maxdiffexp]], %{{.*}} : tensor<16xf32>
  // CHECK-DAG: %[[newrowsum:.+]] = arith.addf %[[rowsummul]], %[[gemm0NormExpSum]] : tensor<16xf32>

  // Gemm1 (V is also transformed via load_marker views; gemm1AccBuf zero init
  //  is the %[[zeroAttn]] constant fed directly as the gemm1 accumulator operand,
  //  replacing the old rock.fill(gemm1AccBuf, zeroVecF32))
  // CHECK: rock.load_marker %[[V]] views [#{{.*}}]
  // CHECK: %[[GEMM1:.+]] = rock.blockwise_gemm(%[[gemm0NormExp]], %{{.*}}, %[[zeroAttn]])

  // Reduction corrections
  // (broadcast of maxdiffexp to the 2D attnOut tile via tt.expand_dims/tt.broadcast)
  // CHECK-DAG: tt.expand_dims %[[maxdiffexp]]
  // CHECK-DAG: tt.broadcast %{{.*}} : tensor<16x1xf32> -> tensor<16x64xf32>
  // CHECK-DAG: %[[attnOutMul:.+]] = arith.mulf %{{.*}}, %{{.*}} : tensor<16x64xf32>
  // CHECK-DAG: %[[newAttnOut:.+]] = arith.addf %[[attnOutMul]], %[[GEMM1]] : tensor<16x64xf32>

  // Carry updated accumulators to the next N-tile iteration
  // CHECK: scf.yield %[[newAttnOut]], %{{.*}}, %[[newrowsum]] : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>

// Final softmax normalization (divide accumulated O by row sum; this completes
// the softmax which the old pipeline left to a separate post-process pass).
// The 1D final row-sum (%[[OUT]]#2) is broadcast back to 2D before the divf.
// CHECK: %[[finalSumExp:.+]] = tt.expand_dims %[[OUT]]#2
// CHECK: %[[finalSumBcast:.+]] = tt.broadcast %[[finalSumExp]] : tensor<16x1xf32> -> tensor<16x64xf32>
// CHECK: arith.divf %[[OUT]]#0, %[[finalSumBcast]] : tensor<16x64xf32>
// CHECK: rock.store_marker %{{.*}} views [#{{.*}}]
func.func @gridwise_attn_simple(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

// -----

// CHECK-LABEL: func @gridwise_attn_schedulev2

// Outer N-tile loop + inner KpacksPerBlock loop
// (lds_barrier / loadType=DoubleBuffer / rock.stage / {name = "MMA"} / CHECK-NOT loadAfromLDS|loadBfromLDS
//  are not emitted by this pass; pipelining/scheduling is deferred and driven by `numStages`)
// CHECK: %{{.*}}:3 = scf.for {{.*}} iter_args
  // CHECK: scf.for {{.*}} iter_args
  // CHECK: rock.load_marker %{{.*}} views [#{{.*}}]
  // CHECK: rock.load_marker %{{.*}} views [#{{.*}}]
  // CHECK: rock.blockwise_gemm

  // CHECK: rock.load_marker %{{.*}} views [#{{.*}}]
  // CHECK: rock.blockwise_gemm

// CHECK: rock.store_marker
func.func @gridwise_attn_schedulev2(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}
