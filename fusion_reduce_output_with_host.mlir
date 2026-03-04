// Interleaved transform/fusion output test with reduce and host verification:
//   gemm [1,100,100]
//   T1 = Unmerge{10,10} on n → [1,100,10,10]
//   addf(T1_result, extra1) in 4D space
//   T2 = Merge{10,10} on (n0,n1) → [1,100,100]
//   addf(T2_result, extra2) in 3D space
//   reduce sum axis=1 (M dimension) → [1,1,100]
//   store to flat 100 output
//
// CPU reference semantics:
//   fused[g,m,n] = gemm(A,B)[g,m,n] + extra1[g,m,n] + extra2[g,m,n]
//   C[n] = sum_m( fused[0,m,n] )  for m in 0..99
#map_flat_10000 = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#map_flat_100 = affine_map<(d0, d1, d2) -> (d1 + d2)>
#map_gemm_to_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 10 + d3)>
#map_4d_to_gemm = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 10, d2 mod 10)>
#map_flat_to_4d = affine_map<(d0, d1, d2, d3) -> (d1 * 100 + d2 * 10 + d3)>
#map_mod3 = affine_map<(d0) -> (d0 mod 3)>
#map_gemm_a = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_gemm_b = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map_gemm_c = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#transform_map_a = #rock.transform_map<#map_flat_10000 by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map_b = #rock.transform_map<#map_flat_10000 by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_gemm_to_4d = #rock.transform_map<#map_gemm_to_4d by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Unmerge{10, 10} ["n0", "n1"] at [2, 3] -> ["n"] at [2]>] bounds = [1, 100, 10, 10] -> [1, 100, 100]>
#transform_4d_to_gemm = #rock.transform_map<#map_4d_to_gemm by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Merge{10, 10} ["n"] at [2] -> ["n0", "n1"] at [2, 3]>] bounds = [1, 100, 100] -> [1, 100, 10, 10]>
#transform_flat_to_4d = #rock.transform_map<#map_flat_to_4d by [<Unmerge{100, 10, 10} ["m", "n0", "n1"] at [1, 2, 3] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 10, 10] -> [10000]>
#transform_flat_to_3d = #rock.transform_map<#map_flat_10000 by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_flat_to_reduced = #rock.transform_map<#map_flat_100 by [<Unmerge{1, 100} ["m_red", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1, 100] -> [100]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // ---- GPU kernel ----
  func.func @rock_gemm(%arg_a: tensor<10000xf16>, %arg_b: tensor<10000xf16>, %extra1: tensor<10000xf16>, %extra2: tensor<10000xf16>, %arg_c: tensor<100xf32>) -> tensor<100xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %a = rock.transform %arg_a by #transform_map_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg_b by #transform_map_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    // T1: Unmerge n → (n0, n1) to get 4D
    %gemm_4d = rock.transform %gemm by #transform_gemm_to_4d : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    // First fusion in 4D space
    %ext1 = rock.transform %extra1 by #transform_flat_to_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %fused1 = arith.addf %gemm_4d, %ext1 : tensor<1x100x10x10xf16>
    // T2: Merge (n0, n1) back to n to get 3D
    %fused1_3d = rock.transform %fused1 by #transform_4d_to_gemm : tensor<1x100x10x10xf16> to tensor<1x100x100xf16>
    // Second fusion in 3D space
    %ext2 = rock.transform %extra2 by #transform_flat_to_3d : tensor<10000xf16> to tensor<1x100x100xf16>
    %fused2 = arith.addf %fused1_3d, %ext2 : tensor<1x100x100xf16>
    // Extend to f32 before reduce so atomic accumulation is in f32
    %fused2_f32 = arith.extf %fused2 : tensor<1x100x100xf16> to tensor<1x100x100xf32>
    // Reduce sum along axis=1 (M dimension): [1,100,100] → [1,1,100]
    %reduced = rock.reduce sum %fused2_f32 {axis = 1 : index} : tensor<1x100x100xf32> -> tensor<1x1x100xf32>
    // Store
    %c = rock.transform %arg_c by #transform_flat_to_reduced : tensor<100xf32> to tensor<1x1x100xf32>
    %result = rock.store %reduced to %c by set : tensor<1x1x100xf32> -> tensor<100xf32> to tensor<1x1x100xf32>
    return %result : tensor<100xf32>
  }

  // ---- Main entry point ----
  func.func @main() {
    %alloc_A = memref.alloc() : memref<10000xf16>
    call @_init_buffer_10000(%alloc_A) : (memref<10000xf16>) -> ()
    %alloc_A_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_A, %alloc_A_ref : memref<10000xf16> to memref<10000xf16>

    %alloc_B = memref.alloc() : memref<10000xf16>
    call @_init_buffer_10000(%alloc_B) : (memref<10000xf16>) -> ()
    %alloc_B_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_B, %alloc_B_ref : memref<10000xf16> to memref<10000xf16>

    %alloc_extra1 = memref.alloc() : memref<10000xf16>
    call @_init_buffer_10000(%alloc_extra1) : (memref<10000xf16>) -> ()
    %alloc_extra1_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_extra1, %alloc_extra1_ref : memref<10000xf16> to memref<10000xf16>

    %alloc_extra2 = memref.alloc() : memref<10000xf16>
    call @_init_buffer_10000(%alloc_extra2) : (memref<10000xf16>) -> ()
    %alloc_extra2_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_extra2, %alloc_extra2_ref : memref<10000xf16> to memref<10000xf16>

    // Output buffer must be zero-initialized: rock.reduce is lowered to
    // atomic_add on the store, so the kernel accumulates into this buffer.
    %cst_zero_f32 = arith.constant 0.000000e+00 : f32
    %alloc_C = memref.alloc() : memref<100xf32>
    linalg.fill ins(%cst_zero_f32 : f32) outs(%alloc_C : memref<100xf32>)
    %alloc_C_ref = memref.alloc() : memref<100xf32>
    linalg.fill ins(%cst_zero_f32 : f32) outs(%alloc_C_ref : memref<100xf32>)

    call @rock_gemm_gpu(%alloc_A, %alloc_B, %alloc_extra1, %alloc_extra2, %alloc_C) : (memref<10000xf16>, memref<10000xf16>, memref<10000xf16>, memref<10000xf16>, memref<100xf32>) -> ()
    call @host_naive_fused_gemm_reduce(%alloc_A_ref, %alloc_B_ref, %alloc_extra1_ref, %alloc_extra2_ref, %alloc_C_ref) : (memref<10000xf16>, memref<10000xf16>, memref<10000xf16>, memref<10000xf16>, memref<100xf32>) -> ()
    call @rock_gemm_verify(%alloc_C, %alloc_C_ref) : (memref<100xf32>, memref<100xf32>) -> ()

    memref.dealloc %alloc_A : memref<10000xf16>
    memref.dealloc %alloc_A_ref : memref<10000xf16>
    memref.dealloc %alloc_B : memref<10000xf16>
    memref.dealloc %alloc_B_ref : memref<10000xf16>
    memref.dealloc %alloc_extra1 : memref<10000xf16>
    memref.dealloc %alloc_extra1_ref : memref<10000xf16>
    memref.dealloc %alloc_extra2 : memref<10000xf16>
    memref.dealloc %alloc_extra2_ref : memref<10000xf16>
    memref.dealloc %alloc_C : memref<100xf32>
    memref.dealloc %alloc_C_ref : memref<100xf32>
    return
  }

  // ---- Buffer initialization: cycling pattern [0.5, -1.0, 0.75] ----
  func.func @_init_buffer_10000(%buf: memref<10000xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf16>
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = vector.insert %cst_0, %cst [0] : f16 into vector<3xf16>
    %cst_1 = arith.constant -1.000000e+00 : f16
    %1 = vector.insert %cst_1, %0 [1] : f16 into vector<3xf16>
    %cst_2 = arith.constant 7.500000e-01 : f16
    %2 = vector.insert %cst_2, %1 [2] : f16 into vector<3xf16>
    affine.for %arg0 = 0 to 10000 {
      %3 = affine.apply #map_mod3(%arg0)
      %4 = vector.extract %2[%3] : f16 from vector<3xf16>
      memref.store %4, %buf[%arg0] : memref<10000xf16>
    }
    return
  }

  func.func @_init_buffer_100(%buf: memref<100xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf16>
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = vector.insert %cst_0, %cst [0] : f16 into vector<3xf16>
    %cst_1 = arith.constant -1.000000e+00 : f16
    %1 = vector.insert %cst_1, %0 [1] : f16 into vector<3xf16>
    %cst_2 = arith.constant 7.500000e-01 : f16
    %2 = vector.insert %cst_2, %1 [2] : f16 into vector<3xf16>
    affine.for %arg0 = 0 to 100 {
      %3 = affine.apply #map_mod3(%arg0)
      %4 = vector.extract %2[%3] : f16 from vector<3xf16>
      memref.store %4, %buf[%arg0] : memref<100xf16>
    }
    return
  }

  // ---- CPU reference ----
  // fused[g,m,n] = gemm(A,B)[g,m,n] + extra1[g,m,n] + extra2[g,m,n]
  // C[n] = sum_m( fused[0,m,n] )
  func.func @host_naive_fused_gemm_reduce(%arg0: memref<10000xf16>, %arg1: memref<10000xf16>, %arg2: memref<10000xf16>, %arg3: memref<10000xf16>, %arg4: memref<100xf32>) {
    %A_f32 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg0, %A_f32) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %B_f32 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg1, %B_f32) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %extra1_f32 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg2, %extra1_f32) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %extra2_f32 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg3, %extra2_f32) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %C_f32 = memref.alloc() : memref<10000xf32>

    // GEMM: C[g,m,n] = sum_k( A[g,m,k] * B[g,k,n] )
    %cst_zero = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst_zero : f32) outs(%C_f32 : memref<10000xf32>)
    %expand_A = memref.expand_shape %A_f32 [[0, 1, 2]] output_shape [1, 100, 100] : memref<10000xf32> into memref<1x100x100xf32>
    %expand_B = memref.expand_shape %B_f32 [[0, 1, 2]] output_shape [1, 100, 100] : memref<10000xf32> into memref<1x100x100xf32>
    %expand_C = memref.expand_shape %C_f32 [[0, 1, 2]] output_shape [1, 100, 100] : memref<10000xf32> into memref<1x100x100xf32>
    linalg.generic {indexing_maps = [#map_gemm_a, #map_gemm_b, #map_gemm_c], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expand_A, %expand_B : memref<1x100x100xf32>, memref<1x100x100xf32>) outs(%expand_C : memref<1x100x100xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %mul = arith.mulf %in, %in_1 : f32
      %add = arith.addf %mul, %out : f32
      linalg.yield %add : f32
    }

    // Output fusion: C[g,m,n] = C[g,m,n] + extra1[g,m,n] + extra2[g,m,n]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10000 = arith.constant 10000 : index
    scf.for %i = %c0 to %c10000 step %c1 {
      %c_val = memref.load %C_f32[%i] : memref<10000xf32>
      %e1_val = memref.load %extra1_f32[%i] : memref<10000xf32>
      %e2_val = memref.load %extra2_f32[%i] : memref<10000xf32>
      %tmp = arith.addf %c_val, %e1_val : f32
      %fused = arith.addf %tmp, %e2_val : f32
      memref.store %fused, %C_f32[%i] : memref<10000xf32>
    }

    // Reduce sum along M (axis=1): result[n] = sum_m( C[0, m, n] )
    %result_f32 = memref.alloc() : memref<100xf32>
    linalg.fill ins(%cst_zero : f32) outs(%result_f32 : memref<100xf32>)
    %c100 = arith.constant 100 : index
    scf.for %m = %c0 to %c100 step %c1 {
      scf.for %n = %c0 to %c100 step %c1 {
        %flat_idx = arith.muli %m, %c100 : index
        %flat = arith.addi %flat_idx, %n : index
        %val = memref.load %C_f32[%flat] : memref<10000xf32>
        %cur = memref.load %result_f32[%n] : memref<100xf32>
        %acc = arith.addf %cur, %val : f32
        memref.store %acc, %result_f32[%n] : memref<100xf32>
      }
    }

    memref.copy %result_f32, %arg4 : memref<100xf32> to memref<100xf32>
    memref.dealloc %A_f32 : memref<10000xf32>
    memref.dealloc %B_f32 : memref<10000xf32>
    memref.dealloc %extra1_f32 : memref<10000xf32>
    memref.dealloc %extra2_f32 : memref<10000xf32>
    memref.dealloc %C_f32 : memref<10000xf32>
    memref.dealloc %result_f32 : memref<100xf32>
    return
  }

  func.func @_memcpy_f16_f32_10000(%arg0: memref<10000xf16>, %arg1: memref<10000xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10000 = arith.constant 10000 : index
    scf.for %arg2 = %c0 to %c10000 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<10000xf16>
      %1 = arith.extf %0 : f16 to f32
      memref.store %1, %arg1[%arg2] : memref<10000xf32>
    }
    return
  }

  func.func @_memcpy_f32_f16_100(%arg0: memref<100xf32>, %arg1: memref<100xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    scf.for %arg2 = %c0 to %c100 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<100xf32>
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %arg1[%arg2] : memref<100xf16>
    }
    return
  }

  func.func @rock_gemm_verify(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    %c1_i8 = arith.constant 1 : i8
    %cst = arith.constant 1.000000e-03 : f32
    %cst_2 = arith.constant 1.000000e+02 : f32
    %cst_4 = arith.constant 1.000000e+02 : f32
    %false = arith.constant false
    call @mcpuVerifyFloat(%arg0, %arg1, %cst, %cst_2, %cst_4, %c1_i8, %false) : (memref<100xf32>, memref<100xf32>, f32, f32, f32, i8, i1) -> ()
    return
  }

  func.func @_memcpy_f16_f32_100(%arg0: memref<100xf16>, %arg1: memref<100xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    scf.for %arg2 = %c0 to %c100 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<100xf16>
      %1 = arith.extf %0 : f16 to f32
      memref.store %1, %arg1[%arg2] : memref<100xf32>
    }
    return
  }

  func.func private @mcpuVerifyFloat(memref<100xf32>, memref<100xf32>, f32, f32, f32, i8, i1)

  func.func @rock_gemm_gpu(%arg0: memref<10000xf16>, %arg1: memref<10000xf16>, %arg2: memref<10000xf16>, %arg3: memref<10000xf16>, %arg4: memref<100xf32>) {
    %memref_A = gpu.alloc() : memref<10000xf16>
    gpu.memcpy %memref_A, %arg0 : memref<10000xf16>, memref<10000xf16>
    %memref_B = gpu.alloc() : memref<10000xf16>
    gpu.memcpy %memref_B, %arg1 : memref<10000xf16>, memref<10000xf16>
    %memref_extra1 = gpu.alloc() : memref<10000xf16>
    gpu.memcpy %memref_extra1, %arg2 : memref<10000xf16>, memref<10000xf16>
    %memref_extra2 = gpu.alloc() : memref<10000xf16>
    gpu.memcpy %memref_extra2, %arg3 : memref<10000xf16>, memref<10000xf16>
    %memref_C = gpu.alloc() : memref<100xf32>
    gpu.memcpy %memref_C, %arg4 : memref<100xf32>, memref<100xf32>

    %0 = bufferization.to_tensor %memref_A restrict writable : memref<10000xf16> to tensor<10000xf16>
    %1 = bufferization.to_tensor %memref_B restrict writable : memref<10000xf16> to tensor<10000xf16>
    %2 = bufferization.to_tensor %memref_extra1 restrict writable : memref<10000xf16> to tensor<10000xf16>
    %3 = bufferization.to_tensor %memref_extra2 restrict writable : memref<10000xf16> to tensor<10000xf16>
    %4 = bufferization.to_tensor %memref_C restrict writable : memref<100xf32> to tensor<100xf32>

    %5 = call @rock_gemm(%0, %1, %2, %3, %4) : (tensor<10000xf16>, tensor<10000xf16>, tensor<10000xf16>, tensor<10000xf16>, tensor<100xf32>) -> tensor<100xf32>

    %6 = bufferization.to_buffer %5 : tensor<100xf32> to memref<100xf32>
    memref.copy %6, %memref_C : memref<100xf32> to memref<100xf32>

    gpu.memcpy %arg0, %memref_A : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc %memref_A : memref<10000xf16>
    gpu.memcpy %arg1, %memref_B : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc %memref_B : memref<10000xf16>
    gpu.memcpy %arg2, %memref_extra1 : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc %memref_extra1 : memref<10000xf16>
    gpu.memcpy %arg3, %memref_extra2 : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc %memref_extra2 : memref<10000xf16>
    gpu.memcpy %arg4, %memref_C : memref<100xf32>, memref<100xf32>
    gpu.dealloc %memref_C : memref<100xf32>
    return
  }
}
