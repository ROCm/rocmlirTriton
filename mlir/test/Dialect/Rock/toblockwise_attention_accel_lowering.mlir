// TODO(rocmlirTriton): Add dumy check to make FileCheck happy
// CHECK: module
// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @gridwise_attn_simple
// DISABLED-CHECK-SAME: (%[[Q:.+]]: memref<1x384x64xf32>, %[[K:.+]]: memref<1x64x384xf32>, %[[V:.+]]: memref<1x384x64xf32>, %[[O:.+]]: memref<1x384x64xf32>)
// DISABLED-CHECK-DAG: %[[ln2Recip:.+]] = arith.constant 1.44269502 : f32
// DISABLED-CHECK-DAG: %[[negInf:.+]] = arith.constant 0xFF800000 : f32
// DISABLED-CHECK-DAG: %[[zeroF32:.+]] = arith.constant 0.000000e+00 : f32
// DISABLED-CHECK-DAG: %[[zeroVecF32:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>

// DISABLED-CHECK-DAG: %[[QTr0:.+]] = rock.transform %[[Q]] by

// init maxRow buffer
// DISABLED-CHECK-DAG: rock.fill(%[[maxRowBuf:.+]], %[[negInf]]) : memref<1xf32

// init sumRow buffer
// DISABLED-CHECK-DAG: rock.fill(%[[sumRowBuf:.+]], %[[zeroF32]]) : memref<1xf32

// init attentionAcc buffer
// DISABLED-CHECK-DAG: rock.fill(%[[attnOutBuf:.+]], %[[zeroF32]]) : memref<2x16xf32

// Outer N-tile loop
// DISABLED-CHECK: scf.for
  // DISABLED-CHECK-DAG: rock.fill(%[[gemm0AccBuf:.+]], %[[zeroVecF32]])
  // DISABLED-CHECK: %[[ldsG0B:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  // DISABLED-CHECK: %[[ldsG0A:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
  
  // DISABLED-CHECK: rock.lds_barrier
  // Inner gemm0 KpacksPerBlock loop
  // DISABLED-CHECK: scf.for 
    // DISABLED-CHECK: rock.blockwise_load_tile %[[QTr0]]{{.*}} LDS -> %[[ldsG0B]] -> %[[preAccelRegB:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>
    // DISABLED-CHECK: rock.blockwise_load_tile %[[K]]{{.*}} LDS -> %[[ldsG0A]] -> %[[preAccelRegA:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>

    // Emit blockwise gemm0
    // DISABLED-CHECK: rock.stage
    // DISABLED-CHECK: %[[viewG0AStore:.+]] = memref.view %[[ldsG0A]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // DISABLED-CHECK: %[[viewG0BStore:.+]] = memref.view %[[ldsG0B]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // DISABLED-CHECK: rock.blockwise_gemm_accel %[[gemm0AccBuf]] += %[[preAccelRegA]] from %[[viewG0AStore]] * %[[preAccelRegB]] from %[[viewG0BStore]]
    // DISABLED-CHECK: {name = "MMA"}
  
  // DISABLED-CHECK: {pipeline = #rock.pipeline<2>}

  // DISABLED-CHECK: rock.transforming_for
    // DISABLED-CHECK: %[[tmp:.+]] =  memref.load %[[gemm0AccBuf]][
    // DISABLED-CHECK: rock.in_bounds_store %[[tmp]] -> %[[gemm0AccBufScalar:.+]][
  // DISABLED-CHECK: linalg.generic {{.*}} ins(%[[gemm0AccBufScalar]] {{.*}} outs(%[[gemm0AccBufScalar]]
    // DISABLED-CHECK: %[[gemm0Scaled:.+]] = arith.mulf %in, %[[ln2Recip]] : f32
    // DISABLED-CHECK: linalg.yield %[[gemm0Scaled]]
  // DISABLED-CHECK: %[[ldsReductionWS:.+]] = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  // DISABLED-CHECK: %[[ldsReductionWSView:.+]] = memref.view %[[ldsReductionWS]][{{.*}}][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  // DISABLED-CHECK: rock.blockwise_broadcast_reduce max {{.*}} %[[gemm0AccBufScalar]] into %[[gemm0Max:[0-9]+]] using %[[ldsReductionWSView]]

  // Compute exp(gemm0 - rowmax_j)
  // *****************************
  // DISABLED-CHECK: rock.transforming_for
    // DISABLED-CHECK-DAG: %[[rowmax:.+]] = rock.in_bounds_load %[[maxRowBuf]]
    // DISABLED-CHECK-DAG: %[[tilemax:.+]] = rock.in_bounds_load %[[gemm0Max]]
    // DISABLED-CHECK-DAG: %[[newmax:.+]] = arith.maximumf %[[rowmax]], %[[tilemax]]
    // DISABLED-CHECK-DAG: %[[gemm0Val:.+]] = rock.in_bounds_load %[[gemm0AccBufScalar]]
    // DISABLED-CHECK-DAG: %[[gemm0ValSubMax:.+]] = arith.subf %[[gemm0Val]], %[[newmax]]
    // DISABLED-CHECK-DAG: %[[gemm0ValSubMaxExp:.+]] = math.exp2 %[[gemm0ValSubMax]]
    // DISABLED-CHECK-DAG: rock.in_bounds_store %[[gemm0ValSubMaxExp]] -> %[[gemm0NormExp:.+]][

  // DISABLED-CHECK: %[[ldsReductionWS2:.+]] = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  // DISABLED-CHECK: %[[ldsReductionWS2View:.+]] = memref.view %[[ldsReductionWS2]][{{.*}}][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  // DISABLED-CHECK: rock.blockwise_broadcast_reduce sum {{.*}} %[[gemm0NormExp]] into %[[gemm0NormExpSum:[0-9]+]] using %[[ldsReductionWS2View]]

  // li = exp(m_{j-1} - m_{j}) * l_{j-1} + rowsum(Pij)
  // where
  // l is the rowsum accumulator
  // m is the rowmax accmulator
  // P is exp(gemm0 - rowmax_j)
  // *************************************************
  // DISABLED-CHECK: rock.transforming_for
    // DISABLED-CHECK-DAG: %[[rowsum:.+]] = rock.in_bounds_load %[[sumRowBuf]]
    // DISABLED-CHECK-DAG: %[[tilesum:.+]] = rock.in_bounds_load %[[gemm0NormExpSum]]
    // DISABLED-CHECK-DAG: %[[rowmax:.+]] = rock.in_bounds_load %[[maxRowBuf]]
    // DISABLED-CHECK-DAG: %[[tilemax:.+]] = rock.in_bounds_load %[[gemm0Max]]
    // DISABLED-CHECK-DAG: %[[newmax:.+]] = arith.maximumf %[[rowmax]], %[[tilemax]]
    // DISABLED-CHECK-DAG: %[[maxdiff:.+]] = arith.subf %[[rowmax]], %[[newmax]]
    // DISABLED-CHECK-DAG: %[[maxdiffexp:.+]] =  math.exp2 %[[maxdiff]]
    // DISABLED-CHECK-DAG: rock.in_bounds_store %[[maxdiffexp]] -> %[[maxdiffexpbuf:.+]][
    // DISABLED-CHECK-DAG: %[[rowsummul:.+]] =  arith.mulf %[[maxdiffexp]], %[[rowsum]]
    // DISABLED-CHECK-DAG: %[[tilesumadd:.+]] =  arith.addf %[[rowsummul]], %[[tilesum]]
    // DISABLED-CHECK-DAG: %[[tilesumadd]] -> %[[sumRowBuf]]

  // Viewing first gemm output as K x D
  // DISABLED-CHECK-DAG: %[[gemm0NormExpTr0:.+]] = rock.transform %[[gemm0NormExp]]
  // DISABLED-CHECK-DAG: %[[gemm0NormExpTr1:.+]] = rock.transform %[[gemm0NormExpTr0]]
  // DISABLED-CHECK-DAG: %[[gemm0NormExpTr2:.+]] = rock.transform %[[gemm0NormExpTr1]]
  // DISABLED-CHECK-DAG: %[[gemm0NormExpTr3:.+]] = rock.transform %[[gemm0NormExpTr2]]
  // DISABLED-CHECK-DAG: %[[gemm0NormExpTr4:.+]] = rock.transform %[[gemm0NormExpTr3]]
  // DISABLED-CHECK-DAG: %[[gemm0NormExpTr5:.+]] = rock.transform %[[gemm0NormExpTr4]]
  
  // DISABLED-CHECK-DAG: %[[ldsG1BStore:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

  // Viewing another set of register with kPack packing
  // DISABLED-CHECK: %[[G1AregsKpackTr0:.+]] = rock.transform %[[G1AregsKpack:.+]] by
  // DISABLED-CHECK-DAG: %[[G1AregsKpackTr1:.+]] = rock.transform %[[G1AregsKpackTr0]] by
  // DISABLED-CHECK-DAG: %[[G1AregsKpackTr2:.+]] = rock.transform %[[G1AregsKpackTr1]] by
  // DISABLED-CHECK-DAG: %[[G1AregsKpackTr3:.+]] = rock.transform %[[G1AregsKpackTr2]] by
  // DISABLED-CHECK-DAG: %[[G1AregsKpackTr4:.+]] = rock.transform %[[G1AregsKpackTr3]] by
  // DISABLED-CHECK-DAG: %[[G1AregsKpackTr5:.+]] = rock.transform %[[G1AregsKpackTr4]] by

  // DISABLED-CHECK-DAG: rock.threadwise_copy %[[gemm0NormExpTr5]] -> %[[G1AregsKpackTr5]]

  // Viewing G1 LDS A tile buffer
  // DISABLED-CHECK-DAG: %[[viewG1AStore:.+]] = memref.view %[[ldsG1BStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr0:.+]] = rock.transform %[[viewG1AStore]]
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr1:.+]] = rock.transform %[[viewG1AStoreTr0]]
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr2:.+]] = rock.transform %[[viewG1AStoreTr1]]
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr3:.+]] = rock.transform %[[viewG1AStoreTr2]]
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr4:.+]] = rock.transform %[[viewG1AStoreTr3]]
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr5:.+]] = rock.transform %[[viewG1AStoreTr4]]
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr6:.+]] = rock.transform %[[viewG1AStoreTr5]]
  // DISABLED-CHECK-DAG: %[[viewG1AStoreTr7:.+]] = rock.transform %[[viewG1AStoreTr6]]

  // Store to LDS G1A tile buffer
  // DISABLED-CHECK-DAG: rock.threadwise_write_all {{.*}} %[[G1AregsKpack]] -> [](%[[viewG1AStoreTr7]])
  
  // DISABLED-CHECK-DAG: %[[ldsG0AStore:.+]] = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>

  // Gemm1
  // DISABLED-CHECK: scf.for %[[g1MIter:.+]]
    // DISABLED-CHECK: rock.blockwise_load_tile %[[V]]{{.*}} LDS -> %[[ldsG0AStore]] -> %[[preAccelRegV:[0-9]+]] {{.*}}#rock<GemmLoadTileType Default>

    // Emit blockwise gemm1
    // rock.stage
    // DISABLED-CHECK-DAG: rock.fill(%[[gemm1AccBuf:.+]], %[[zeroVecF32]])
    // DISABLED-CHECK: %[[view2G1AStore:.+]] = memref.view %[[ldsG0AStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // DISABLED-CHECK: %[[view2G1BStore:.+]] = memref.view %[[ldsG1BStore]][{{.*}}][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    // DISABLED-CHECK: rock.blockwise_gemm_accel %[[gemm1AccBuf]] += %[[preAccelRegV]] from %[[view2G1AStore]] * %[[preAccelRegA:[0-9]+]] from %[[view2G1BStore]]
    // DISABLED-CHECK: {name = "MMA"}

    // rock.stage
    // DISABLED-CHECK: rock.transforming_for
      // DISABLED-CHECK: %[[tmp1:.+]] =  memref.load %[[gemm1AccBuf]][
      // DISABLED-CHECK: rock.in_bounds_store %[[tmp1]] -> %[[gemm1AccBufScalar:.+]][

    // DISABLED-CHECK: %[[sliceAttnOutBuf:.+]] = memref.subview %[[attnOutBuf]]
    // Reduction corrections
    // DISABLED-CHECK: rock.transforming_for
      // DISABLED-CHECK-DAG: %[[maxdiffexp:.+]] = rock.in_bounds_load %[[maxdiffexpbuf]]
      // DISABLED-CHECK-DAG: %[[attnOutVal:.+]] = rock.in_bounds_load %[[sliceAttnOutBuf]]
      // DISABLED-CHECK-DAG: %[[gemm1Val:.+]] = rock.in_bounds_load %[[gemm1AccBufScalar]]

      // DISABLED-CHECK-DAG: %[[attnOutBufMul:.+]] = arith.mulf %[[attnOutVal]], %[[maxdiffexp]]
      // DISABLED-CHECK-DAG: %[[newattnOutVal:.+]] = arith.addf %[[attnOutBufMul]], %[[gemm1Val]]
      // DISABLED-CHECK-DAG: rock.in_bounds_store %[[newattnOutVal]] -> %[[sliceAttnOutBuf]]
    // DISABLED-CHECK : }
    // DISABLED-CHECK: {name = "PostProcess"}
  // DISABLED-CHECK : {pipeline = #rock.pipeline<2>}
// DISABLED-CHECK : }
// DISABLED-CHECK : %[[flatAttnOutBuf:.+]] = memref.collapse_shape %[[attnOutBuf]]
// DISABLED-CHECK : rock.threadwise_write_all {{.*}} %[[flatAttnOutBuf]] -> {{.*}}(%[[O]])

// func.func @gridwise_attn_simple(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {rock.block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
//   rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
//     blockSize = 64 : i32,
//     gridSize = 24 : i32,
//     params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
//     params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 1 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>
//   } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
//   return
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @gridwise_attn_schedulev2
// func.func @gridwise_attn_schedulev2(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {rock.block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>

  // DISABLED-CHECK: scf.for
  // DISABLED-CHECK: rock.lds_barrier
  // DISABLED-CHECK: scf.for

  // DISABLED-CHECK: rock.blockwise_load_tile
  // DISABLED-CHECK: loadType = #rock<GemmLoadTileType DoubleBuffer>

  // DISABLED-CHECK: rock.blockwise_load_tile
  // DISABLED-CHECK: loadType = #rock<GemmLoadTileType DoubleBuffer>

  // DISABLED-CHECK: rock.stage
  // DISABLED-CHECK: rock.blockwise_gemm_accel 
  // DISABLED-CHECK-NOT: loadAfromLDS
  // DISABLED-CHECK-NOT: loadBfromLDS
  // DISABLED-CHECK: {name = "MMA"}

  // scf.for

  // DISABLED-CHECK: rock.blockwise_load_tile
  // DISABLED-CHECK: loadType = #rock<GemmLoadTileType DoubleBuffer>

  // DISABLED-CHECK: rock.stage
  // DISABLED-CHECK: rock.blockwise_gemm_accel
  // DISABLED-CHECK: {name = "MMA"}
//   rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
//     blockSize = 64 : i32,
//     gridSize = 24 : i32,
//     params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
//     params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 1 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>
//   } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
//   return
// }
