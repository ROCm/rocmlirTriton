// Fused GEMM test: C = (A + shared) * B + shared
// Tests using the same tensor for both input and output fusion.
#map = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#map1 = affine_map<(d0) -> (d0 mod 3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %shared: tensor<10000xf16>, %arg2: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<10000xf16> to tensor<1x100x100xf16>
    %s_in = rock.transform %shared by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %s_out = rock.transform %shared by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %a = arith.addf %0, %s_in : tensor<1x100x100xf16>
    %3 = rock.gemm %a * %1 : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %fusion = arith.addf %3, %s_out : tensor<1x100x100xf16>
    %4 = rock.store %fusion to %2 by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %4 : tensor<10000xf16>
  }

  func.func @main() {
    %alloc_A = memref.alloc() : memref<10000xf16>
    call @_init_buffer(%alloc_A) : (memref<10000xf16>) -> ()
    %alloc_A_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_A, %alloc_A_ref : memref<10000xf16> to memref<10000xf16>

    %alloc_B = memref.alloc() : memref<10000xf16>
    call @_init_buffer(%alloc_B) : (memref<10000xf16>) -> ()
    %alloc_B_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_B, %alloc_B_ref : memref<10000xf16> to memref<10000xf16>

    %alloc_shared = memref.alloc() : memref<10000xf16>
    call @_init_buffer(%alloc_shared) : (memref<10000xf16>) -> ()
    %alloc_shared_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_shared, %alloc_shared_ref : memref<10000xf16> to memref<10000xf16>

    %alloc_C = memref.alloc() : memref<10000xf16>
    call @_init_buffer(%alloc_C) : (memref<10000xf16>) -> ()
    %alloc_C_ref = memref.alloc() : memref<10000xf16>
    memref.copy %alloc_C, %alloc_C_ref : memref<10000xf16> to memref<10000xf16>

    call @rock_gemm_gpu(%alloc_A, %alloc_B, %alloc_shared, %alloc_C) : (memref<10000xf16>, memref<10000xf16>, memref<10000xf16>, memref<10000xf16>) -> ()
    call @host_naive_fused_gemm(%alloc_A_ref, %alloc_B_ref, %alloc_shared_ref, %alloc_C_ref) : (memref<10000xf16>, memref<10000xf16>, memref<10000xf16>, memref<10000xf16>) -> ()
    call @rock_gemm_verify(%alloc_C, %alloc_C_ref) : (memref<10000xf16>, memref<10000xf16>) -> ()

    memref.dealloc %alloc_A_ref : memref<10000xf16>
    memref.dealloc %alloc_B_ref : memref<10000xf16>
    memref.dealloc %alloc_shared_ref : memref<10000xf16>
    memref.dealloc %alloc_C_ref : memref<10000xf16>
    memref.dealloc %alloc_A : memref<10000xf16>
    memref.dealloc %alloc_B : memref<10000xf16>
    memref.dealloc %alloc_shared : memref<10000xf16>
    memref.dealloc %alloc_C : memref<10000xf16>
    return
  }

  func.func @_init_buffer(%buf: memref<10000xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf16>
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = vector.insert %cst_0, %cst [0] : f16 into vector<3xf16>
    %cst_1 = arith.constant -1.000000e+00 : f16
    %1 = vector.insert %cst_1, %0 [1] : f16 into vector<3xf16>
    %cst_2 = arith.constant 7.500000e-01 : f16
    %2 = vector.insert %cst_2, %1 [2] : f16 into vector<3xf16>
    affine.for %arg0 = 0 to 10000 {
      %3 = affine.apply #map1(%arg0)
      %4 = vector.extract %2[%3] : f16 from vector<3xf16>
      memref.store %4, %buf[%arg0] : memref<10000xf16>
    }
    return
  }

  // CPU reference: C = (A + shared) * B + shared
  // Input fusion uses shared in (m,k) layout, output uses (m,n) layout.
  // Since both use the same affine formula (d1*100+d2), the flat indexing
  // is the same: shared[i] at position [g=0, d1=i/100, d2=i%100].
  func.func @host_naive_fused_gemm(%arg0: memref<10000xf16>, %arg1: memref<10000xf16>, %arg2: memref<10000xf16>, %arg3: memref<10000xf16>) {
    %A_f32 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg0, %A_f32) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %B_f32 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg1, %B_f32) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %shared_f32 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg2, %shared_f32) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %C_f32 = memref.alloc() : memref<10000xf32>

    // Input fusion: A' = A + shared (elementwise on flat buffers, same m,k layout)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10000 = arith.constant 10000 : index
    scf.for %i = %c0 to %c10000 step %c1 {
      %a_val = memref.load %A_f32[%i] : memref<10000xf32>
      %s_val = memref.load %shared_f32[%i] : memref<10000xf32>
      %a_fused = arith.addf %a_val, %s_val : f32
      memref.store %a_fused, %A_f32[%i] : memref<10000xf32>
    }

    // GEMM: C[g,m,n] = sum_k( A'[g,m,k] * B[g,k,n] )
    %cst_zero = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst_zero : f32) outs(%C_f32 : memref<10000xf32>)
    %expand_A = memref.expand_shape %A_f32 [[0, 1, 2]] output_shape [1, 100, 100] : memref<10000xf32> into memref<1x100x100xf32>
    %expand_B = memref.expand_shape %B_f32 [[0, 1, 2]] output_shape [1, 100, 100] : memref<10000xf32> into memref<1x100x100xf32>
    %expand_C = memref.expand_shape %C_f32 [[0, 1, 2]] output_shape [1, 100, 100] : memref<10000xf32> into memref<1x100x100xf32>
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expand_A, %expand_B : memref<1x100x100xf32>, memref<1x100x100xf32>) outs(%expand_C : memref<1x100x100xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_1 : f32
      %1 = arith.addf %0, %out : f32
      linalg.yield %1 : f32
    }

    // Output fusion: C = C + shared (elementwise on flat buffers, same m,n layout)
    scf.for %i = %c0 to %c10000 step %c1 {
      %c_val = memref.load %C_f32[%i] : memref<10000xf32>
      %s_val = memref.load %shared_f32[%i] : memref<10000xf32>
      %c_fused = arith.addf %c_val, %s_val : f32
      memref.store %c_fused, %C_f32[%i] : memref<10000xf32>
    }

    call @_memcpy_f32_f16_10000(%C_f32, %arg3) : (memref<10000xf32>, memref<10000xf16>) -> ()
    memref.dealloc %A_f32 : memref<10000xf32>
    memref.dealloc %B_f32 : memref<10000xf32>
    memref.dealloc %shared_f32 : memref<10000xf32>
    memref.dealloc %C_f32 : memref<10000xf32>
    return
  }

  func.func @_memcpy_f16_f32_10000(%arg0: memref<10000xf16>, %arg1: memref<10000xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<10000xf16>
    scf.for %arg2 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<10000xf16>
      %1 = arith.extf %0 : f16 to f32
      memref.store %1, %arg1[%arg2] : memref<10000xf32>
    }
    return
  }

  func.func @_memcpy_f32_f16_10000(%arg0: memref<10000xf32>, %arg1: memref<10000xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<10000xf32>
    scf.for %arg2 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<10000xf32>
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %arg1[%arg2] : memref<10000xf16>
    }
    return
  }

  func.func @rock_gemm_verify(%arg0: memref<10000xf16>, %arg1: memref<10000xf16>) {
    %c1_i8 = arith.constant 1 : i8
    %alloc = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg0, %alloc) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %cast = memref.cast %alloc : memref<10000xf32> to memref<?xf32>
    %alloc_0 = memref.alloc() : memref<10000xf32>
    call @_memcpy_f16_f32_10000(%arg1, %alloc_0) : (memref<10000xf16>, memref<10000xf32>) -> ()
    %cast_1 = memref.cast %alloc_0 : memref<10000xf32> to memref<?xf32>
    %cst = arith.constant 1.000000e-03 : f32
    %cst_2 = arith.constant 1.000000e+02 : f32
    %cst_4 = arith.constant 1.000000e+02 : f32
    %false = arith.constant false
    call @mcpuVerifyFloat(%cast, %cast_1, %cst, %cst_2, %cst_4, %c1_i8, %false) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1) -> ()
    memref.dealloc %alloc : memref<10000xf32>
    memref.dealloc %alloc_0 : memref<10000xf32>
    return
  }

  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1)

  func.func @rock_gemm_gpu(%arg0: memref<10000xf16>, %arg1: memref<10000xf16>, %arg2: memref<10000xf16>, %arg3: memref<10000xf16>) {
    %memref_A = gpu.alloc  () : memref<10000xf16>
    gpu.memcpy  %memref_A, %arg0 : memref<10000xf16>, memref<10000xf16>
    %memref_B = gpu.alloc  () : memref<10000xf16>
    gpu.memcpy  %memref_B, %arg1 : memref<10000xf16>, memref<10000xf16>
    %memref_shared = gpu.alloc  () : memref<10000xf16>
    gpu.memcpy  %memref_shared, %arg2 : memref<10000xf16>, memref<10000xf16>
    %memref_C = gpu.alloc  () : memref<10000xf16>
    gpu.memcpy  %memref_C, %arg3 : memref<10000xf16>, memref<10000xf16>

    %0 = bufferization.to_tensor %memref_A restrict writable : memref<10000xf16> to tensor<10000xf16>
    %1 = bufferization.to_tensor %memref_B restrict writable : memref<10000xf16> to tensor<10000xf16>
    %2 = bufferization.to_tensor %memref_shared restrict writable : memref<10000xf16> to tensor<10000xf16>
    %3 = bufferization.to_tensor %memref_C restrict writable : memref<10000xf16> to tensor<10000xf16>

    %4 = call @rock_gemm(%0, %1, %2, %3) : (tensor<10000xf16>, tensor<10000xf16>, tensor<10000xf16>, tensor<10000xf16>) -> tensor<10000xf16>

    %5 = bufferization.to_buffer %4 : tensor<10000xf16> to memref<10000xf16>
    memref.copy %5, %memref_C : memref<10000xf16> to memref<10000xf16>

    gpu.memcpy  %arg0, %memref_A : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc  %memref_A : memref<10000xf16>
    gpu.memcpy  %arg1, %memref_B : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc  %memref_B : memref<10000xf16>
    gpu.memcpy  %arg2, %memref_shared : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc  %memref_shared : memref<10000xf16>
    gpu.memcpy  %arg3, %memref_C : memref<10000xf16>, memref<10000xf16>
    gpu.dealloc  %memref_C : memref<10000xf16>
    return
  }
}
