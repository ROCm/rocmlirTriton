// Fused GEMM test: C_f32 = extf(gemm(truncf(A_f32), truncf(B_f32)))
// Tests type-changing fusions: arith.truncf on inputs (f32 -> f16) and
// arith.extf on output (f16 -> f32).
#map = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#map1 = affine_map<(d0) -> (d0 mod 3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<10000xf32>, %arg1: tensor<10000xf32>, %arg2: tensor<10000xf32>) -> tensor<10000xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<10000xf32> to tensor<1x100x100xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<10000xf32> to tensor<1x100x100xf32>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<10000xf32> to tensor<1x100x100xf32>
    %a = arith.truncf %0 : tensor<1x100x100xf32> to tensor<1x100x100xf16>
    %b = arith.truncf %1 : tensor<1x100x100xf32> to tensor<1x100x100xf16>
    %3 = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_f32 = arith.extf %3 : tensor<1x100x100xf16> to tensor<1x100x100xf32>
    %4 = rock.store %gemm_f32 to %2 by set : tensor<1x100x100xf32> -> tensor<10000xf32> to tensor<1x100x100xf32>
    return %4 : tensor<10000xf32>
  }

  func.func @main() {
    %alloc_A = memref.alloc() : memref<10000xf32>
    call @_init_buffer_f32(%alloc_A) : (memref<10000xf32>) -> ()
    %alloc_A_ref = memref.alloc() : memref<10000xf32>
    memref.copy %alloc_A, %alloc_A_ref : memref<10000xf32> to memref<10000xf32>

    %alloc_B = memref.alloc() : memref<10000xf32>
    call @_init_buffer_f32(%alloc_B) : (memref<10000xf32>) -> ()
    %alloc_B_ref = memref.alloc() : memref<10000xf32>
    memref.copy %alloc_B, %alloc_B_ref : memref<10000xf32> to memref<10000xf32>

    %alloc_C = memref.alloc() : memref<10000xf32>
    call @_init_buffer_f32(%alloc_C) : (memref<10000xf32>) -> ()
    %alloc_C_ref = memref.alloc() : memref<10000xf32>
    memref.copy %alloc_C, %alloc_C_ref : memref<10000xf32> to memref<10000xf32>

    call @rock_gemm_gpu(%alloc_A, %alloc_B, %alloc_C) : (memref<10000xf32>, memref<10000xf32>, memref<10000xf32>) -> ()
    call @host_naive_fused_gemm(%alloc_A_ref, %alloc_B_ref, %alloc_C_ref) : (memref<10000xf32>, memref<10000xf32>, memref<10000xf32>) -> ()
    call @rock_gemm_verify(%alloc_C, %alloc_C_ref) : (memref<10000xf32>, memref<10000xf32>) -> ()

    memref.dealloc %alloc_A_ref : memref<10000xf32>
    memref.dealloc %alloc_B_ref : memref<10000xf32>
    memref.dealloc %alloc_C_ref : memref<10000xf32>
    memref.dealloc %alloc_A : memref<10000xf32>
    memref.dealloc %alloc_B : memref<10000xf32>
    memref.dealloc %alloc_C : memref<10000xf32>
    return
  }

  // Initialize f32 buffer with cycling pattern [0.5, -1.0, 0.75]
  func.func @_init_buffer_f32(%buf: memref<10000xf32>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf32>
    %cst_0 = arith.constant 5.000000e-01 : f32
    %0 = vector.insert %cst_0, %cst [0] : f32 into vector<3xf32>
    %cst_1 = arith.constant -1.000000e+00 : f32
    %1 = vector.insert %cst_1, %0 [1] : f32 into vector<3xf32>
    %cst_2 = arith.constant 7.500000e-01 : f32
    %2 = vector.insert %cst_2, %1 [2] : f32 into vector<3xf32>
    affine.for %arg0 = 0 to 10000 {
      %3 = affine.apply #map1(%arg0)
      %4 = vector.extract %2[%3] : f32 from vector<3xf32>
      memref.store %4, %buf[%arg0] : memref<10000xf32>
    }
    return
  }

  // CPU reference: C_f32 = extf(gemm(truncf(A_f32), truncf(B_f32)))
  // 1. Round A, B to f16 precision
  // 2. GEMM in f32 (matching GPU internal accumulation)
  // 3. Truncate result to f16 (matching GPU f16 GEMM output)
  // 4. Extend back to f32 (matching extf output fusion)
  func.func @host_naive_fused_gemm(%arg0: memref<10000xf32>, %arg1: memref<10000xf32>, %arg2: memref<10000xf32>) {
    %A_f32 = memref.alloc() : memref<10000xf32>
    memref.copy %arg0, %A_f32 : memref<10000xf32> to memref<10000xf32>
    %B_f32 = memref.alloc() : memref<10000xf32>
    memref.copy %arg1, %B_f32 : memref<10000xf32> to memref<10000xf32>
    %C_f32 = memref.alloc() : memref<10000xf32>

    // Input fusion: truncate A, B to f16 precision (round-trip)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10000 = arith.constant 10000 : index
    scf.for %i = %c0 to %c10000 step %c1 {
      %a_val = memref.load %A_f32[%i] : memref<10000xf32>
      %a_f16 = arith.truncf %a_val : f32 to f16
      %a_rounded = arith.extf %a_f16 : f16 to f32
      memref.store %a_rounded, %A_f32[%i] : memref<10000xf32>
      %b_val = memref.load %B_f32[%i] : memref<10000xf32>
      %b_f16 = arith.truncf %b_val : f32 to f16
      %b_rounded = arith.extf %b_f16 : f16 to f32
      memref.store %b_rounded, %B_f32[%i] : memref<10000xf32>
    }

    // GEMM in f32
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

    // Output fusion: truncate GEMM result to f16 (matching GPU), then extend to f32
    scf.for %i = %c0 to %c10000 step %c1 {
      %c_val = memref.load %C_f32[%i] : memref<10000xf32>
      %c_f16 = arith.truncf %c_val : f32 to f16
      %c_rounded = arith.extf %c_f16 : f16 to f32
      memref.store %c_rounded, %C_f32[%i] : memref<10000xf32>
    }

    memref.copy %C_f32, %arg2 : memref<10000xf32> to memref<10000xf32>
    memref.dealloc %A_f32 : memref<10000xf32>
    memref.dealloc %B_f32 : memref<10000xf32>
    memref.dealloc %C_f32 : memref<10000xf32>
    return
  }

  func.func @rock_gemm_verify(%arg0: memref<10000xf32>, %arg1: memref<10000xf32>) {
    %c1_i8 = arith.constant 1 : i8
    %cast = memref.cast %arg0 : memref<10000xf32> to memref<?xf32>
    %cast_1 = memref.cast %arg1 : memref<10000xf32> to memref<?xf32>
    %cst = arith.constant 1.000000e-03 : f32
    %cst_2 = arith.constant 1.000000e+02 : f32
    %cst_4 = arith.constant 1.000000e+02 : f32
    %false = arith.constant false
    call @mcpuVerifyFloat(%cast, %cast_1, %cst, %cst_2, %cst_4, %c1_i8, %false) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1) -> ()
    return
  }

  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1)

  func.func @rock_gemm_gpu(%arg0: memref<10000xf32>, %arg1: memref<10000xf32>, %arg2: memref<10000xf32>) {
    %memref_A = gpu.alloc  () : memref<10000xf32>
    gpu.memcpy  %memref_A, %arg0 : memref<10000xf32>, memref<10000xf32>
    %memref_B = gpu.alloc  () : memref<10000xf32>
    gpu.memcpy  %memref_B, %arg1 : memref<10000xf32>, memref<10000xf32>
    %memref_C = gpu.alloc  () : memref<10000xf32>
    gpu.memcpy  %memref_C, %arg2 : memref<10000xf32>, memref<10000xf32>

    %0 = bufferization.to_tensor %memref_A restrict writable : memref<10000xf32> to tensor<10000xf32>
    %1 = bufferization.to_tensor %memref_B restrict writable : memref<10000xf32> to tensor<10000xf32>
    %2 = bufferization.to_tensor %memref_C restrict writable : memref<10000xf32> to tensor<10000xf32>

    %3 = call @rock_gemm(%0, %1, %2) : (tensor<10000xf32>, tensor<10000xf32>, tensor<10000xf32>) -> tensor<10000xf32>

    %4 = bufferization.to_buffer %3 : tensor<10000xf32> to memref<10000xf32>
    memref.copy %4, %memref_C : memref<10000xf32> to memref<10000xf32>

    gpu.memcpy  %arg0, %memref_A : memref<10000xf32>, memref<10000xf32>
    gpu.dealloc  %memref_A : memref<10000xf32>
    gpu.memcpy  %arg1, %memref_B : memref<10000xf32>, memref<10000xf32>
    gpu.dealloc  %memref_B : memref<10000xf32>
    gpu.memcpy  %arg2, %memref_C : memref<10000xf32>, memref<10000xf32>
    gpu.dealloc  %memref_C : memref<10000xf32>
    return
  }
}
