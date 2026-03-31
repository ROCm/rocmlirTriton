// E2E test for non-contiguous (long stride) output tensors.
// The kernel computes batched GEMM (G=4, M=24, K=16, N=24) followed by
// sigmoid activation, then writes the result into a padded 4x48x24 output
// buffer via rock.expand_strides, where dim1 is expanded from 24 to 48
// (stride=48 on the M dimension instead of contiguous stride=24).
// The arch (gfx1100) and num_cu (96) are placeholders; tests.sh substitutes
// them with detected GPU values via sed.

#map_init = affine_map<(d0) -> (d0 mod 3)>

// B: flat 1536 -> 4x16x24
#map = affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 24 + d2)>
#transform_map = #rock.transform_map<#map by [
  <Unmerge{4, 16, 24} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>
] bounds = [4, 16, 24] -> [1536]>

// A: flat 1536 -> 4x24x16
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 24 + d1) * 16 + d2)>
#transform_map1 = #rock.transform_map<#map1 by [
  <Unmerge{4, 24, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>
] bounds = [4, 24, 16] -> [1536]>

// Merge expanded result: 4x48x24 -> flat 4608
#map2 = affine_map<(d0) -> (d0 floordiv 1152, (d0 mod 1152) floordiv 24, d0 mod 24)>
#transform_map2 = #rock.transform_map<#map2 by [
  <Merge{4, 48, 24} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>
] bounds = [4608] -> [4, 48, 24]>

module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {

  // ---- GPU kernel: batched GEMM + sigmoid with expand_strides on output ----
  func.func @mlir_dot_sigmoid(
      %arg0: tensor<1536xf16>,
      %arg1: tensor<1536xf16>,
      %arg2: tensor<4608xf16>
  ) -> tensor<4608xf16> attributes {
    rock.arch = "amdgcn-amd-amdhsa:gfx1100",
    rock.enable_splitk_for_tuning,
    rock.kernel,
    rock.num_chiplets = 1 : i64,
    rock.num_cu = 96 : i64
  } {
    %cst_one = arith.constant dense<1.000000e+00> : tensor<4x24x24xf16>
    %cst_neg_one = arith.constant dense<-1.000000e+00> : tensor<4x24x24xf16>
    %0 = rock.transform %arg1 by #transform_map : tensor<1536xf16> to tensor<4x16x24xf16>
    %1 = rock.transform %arg0 by #transform_map1 : tensor<1536xf16> to tensor<4x24x16xf16>
    %2 = rock.gemm %1 * %0 : tensor<4x24x16xf16> * tensor<4x16x24xf16> -> tensor<4x24x24xf16>
    %3 = arith.mulf %2, %cst_neg_one : tensor<4x24x24xf16>
    %4 = math.exp %3 : tensor<4x24x24xf16>
    %5 = arith.addf %4, %cst_one : tensor<4x24x24xf16>
    %6 = arith.divf %cst_one, %5 : tensor<4x24x24xf16>
    %7 = rock.expand_strides %6 : tensor<4x24x24xf16> -> tensor<4x48x24xf16>
    %8 = rock.transform %7 by #transform_map2 : tensor<4x48x24xf16> to tensor<4608xf16>
    %9 = rock.store %8 to %arg2 by set : tensor<4608xf16> -> tensor<4608xf16> to tensor<4608xf16>
    return %9 : tensor<4608xf16>
  }

  // ---- Main entry point ----
  func.func @main() {
    %alloc_A = memref.alloc() : memref<1536xf16>
    call @_init_buffer_1536(%alloc_A) : (memref<1536xf16>) -> ()
    %alloc_A_ref = memref.alloc() : memref<1536xf16>
    memref.copy %alloc_A, %alloc_A_ref : memref<1536xf16> to memref<1536xf16>

    %alloc_B = memref.alloc() : memref<1536xf16>
    call @_init_buffer_1536(%alloc_B) : (memref<1536xf16>) -> ()
    %alloc_B_ref = memref.alloc() : memref<1536xf16>
    memref.copy %alloc_B, %alloc_B_ref : memref<1536xf16> to memref<1536xf16>

    %alloc_out = memref.alloc() : memref<4608xf16>
    call @_zero_buffer_4608(%alloc_out) : (memref<4608xf16>) -> ()
    %alloc_out_ref = memref.alloc() : memref<4608xf16>
    call @_zero_buffer_4608(%alloc_out_ref) : (memref<4608xf16>) -> ()

    call @mlir_dot_sigmoid_gpu(%alloc_A, %alloc_B, %alloc_out)
      : (memref<1536xf16>, memref<1536xf16>, memref<4608xf16>) -> ()

    call @host_naive_dot_sigmoid_expand(%alloc_A_ref, %alloc_B_ref, %alloc_out_ref)
      : (memref<1536xf16>, memref<1536xf16>, memref<4608xf16>) -> ()

    call @verify_4608(%alloc_out, %alloc_out_ref)
      : (memref<4608xf16>, memref<4608xf16>) -> ()

    memref.dealloc %alloc_A : memref<1536xf16>
    memref.dealloc %alloc_A_ref : memref<1536xf16>
    memref.dealloc %alloc_B : memref<1536xf16>
    memref.dealloc %alloc_B_ref : memref<1536xf16>
    memref.dealloc %alloc_out : memref<4608xf16>
    memref.dealloc %alloc_out_ref : memref<4608xf16>
    return
  }

  // ---- Buffer initialization: cycling pattern [0.5, -1.0, 0.75] ----
  func.func @_init_buffer_1536(%buf: memref<1536xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf16>
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = vector.insert %cst_0, %cst [0] : f16 into vector<3xf16>
    %cst_1 = arith.constant -1.000000e+00 : f16
    %1 = vector.insert %cst_1, %0 [1] : f16 into vector<3xf16>
    %cst_2 = arith.constant 7.500000e-01 : f16
    %2 = vector.insert %cst_2, %1 [2] : f16 into vector<3xf16>
    affine.for %i = 0 to 1536 {
      %3 = affine.apply #map_init(%i)
      %4 = vector.extract %2[%3] : f16 from vector<3xf16>
      memref.store %4, %buf[%i] : memref<1536xf16>
    }
    return
  }

  func.func @_zero_buffer_4608(%buf: memref<4608xf16>) {
    %zero = arith.constant 0.000000e+00 : f16
    affine.for %i = 0 to 4608 {
      memref.store %zero, %buf[%i] : memref<4608xf16>
    }
    return
  }

  // ---- CPU reference: batched GEMM + sigmoid + strided output ----
  // C[g][m][n] = sigmoid(sum_k(A[g][m][k] * B[g][k][n]))
  // A is indexed as arg0[(g*24+m)*16+k], B is indexed as arg1[(g*16+k)*24+n]
  // Output placed at arg_out[(g*48+m)*24+n] for m < 24
  func.func @host_naive_dot_sigmoid_expand(
      %arg_A: memref<1536xf16>,
      %arg_B: memref<1536xf16>,
      %arg_out: memref<4608xf16>
  ) {
    %A_f32 = memref.alloc() : memref<1536xf32>
    call @_memcpy_f16_f32_1536(%arg_A, %A_f32) : (memref<1536xf16>, memref<1536xf32>) -> ()
    %B_f32 = memref.alloc() : memref<1536xf32>
    call @_memcpy_f16_f32_1536(%arg_B, %B_f32) : (memref<1536xf16>, memref<1536xf32>) -> ()

    %out_f32 = memref.alloc() : memref<4608xf32>
    %cst_zero = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst_zero : f32) outs(%out_f32 : memref<4608xf32>)

    %cst_one = arith.constant 1.000000e+00 : f32
    %cst_acc = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c48 = arith.constant 48 : index

    scf.for %g = %c0 to %c4 step %c1 {
      scf.for %m = %c0 to %c24 step %c1 {
        scf.for %n = %c0 to %c24 step %c1 {
          // GEMM accumulation: sum_k A[g][m][k] * B[g][k][n]
          %gemm_val = scf.for %k = %c0 to %c16 step %c1 iter_args(%sum = %cst_acc) -> (f32) {
            // A[g][m][k] at flat index (g*24+m)*16+k
            %gm = arith.muli %g, %c24 : index
            %gm_m = arith.addi %gm, %m : index
            %gm_m_16 = arith.muli %gm_m, %c16 : index
            %a_idx = arith.addi %gm_m_16, %k : index
            %a_val = memref.load %A_f32[%a_idx] : memref<1536xf32>

            // B[g][k][n] at flat index (g*16+k)*24+n
            %gk = arith.muli %g, %c16 : index
            %gk_k = arith.addi %gk, %k : index
            %gk_k_24 = arith.muli %gk_k, %c24 : index
            %b_idx = arith.addi %gk_k_24, %n : index
            %b_val = memref.load %B_f32[%b_idx] : memref<1536xf32>

            %prod = arith.mulf %a_val, %b_val : f32
            %new_sum = arith.addf %sum, %prod : f32
            scf.yield %new_sum : f32
          }

          // Sigmoid: 1 / (1 + exp(-x))
          %neg = arith.negf %gemm_val : f32
          %exp_val = math.exp %neg : f32
          %one_plus = arith.addf %exp_val, %cst_one : f32
          %sigmoid = arith.divf %cst_one, %one_plus : f32

          // Output at flat index (g*48+m)*24+n
          %g48 = arith.muli %g, %c48 : index
          %g48_m = arith.addi %g48, %m : index
          %g48_m_24 = arith.muli %g48_m, %c24 : index
          %out_idx = arith.addi %g48_m_24, %n : index
          memref.store %sigmoid, %out_f32[%out_idx] : memref<4608xf32>
        }
      }
    }

    call @_memcpy_f32_f16_4608(%out_f32, %arg_out) : (memref<4608xf32>, memref<4608xf16>) -> ()
    memref.dealloc %A_f32 : memref<1536xf32>
    memref.dealloc %B_f32 : memref<1536xf32>
    memref.dealloc %out_f32 : memref<4608xf32>
    return
  }

  // ---- f16/f32 conversion helpers ----
  func.func @_memcpy_f16_f32_1536(%src: memref<1536xf16>, %dst: memref<1536xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %src, %c0 : memref<1536xf16>
    scf.for %i = %c0 to %dim step %c1 {
      %v = memref.load %src[%i] : memref<1536xf16>
      %ext = arith.extf %v : f16 to f32
      memref.store %ext, %dst[%i] : memref<1536xf32>
    }
    return
  }

  func.func @_memcpy_f32_f16_4608(%src: memref<4608xf32>, %dst: memref<4608xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %src, %c0 : memref<4608xf32>
    scf.for %i = %c0 to %dim step %c1 {
      %v = memref.load %src[%i] : memref<4608xf32>
      %trunc = arith.truncf %v : f32 to f16
      memref.store %trunc, %dst[%i] : memref<4608xf16>
    }
    return
  }

  // Extract non-padding elements from 4x48x24 layout into packed 2304xf32.
  // Only positions where m < 24 (within each group's 48-row stride) are
  // valid; the rest are uninitialized padding from rock.expand_strides.
  // src[(g*48+m)*24+n] -> dst[g*576+m*24+n] for g in [0,4), m in [0,24), n in [0,24)
  func.func @_extract_nonpadding(%src: memref<4608xf16>, %dst: memref<2304xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c24 = arith.constant 24 : index
    %c48 = arith.constant 48 : index
    %c576 = arith.constant 576 : index
    scf.for %g = %c0 to %c4 step %c1 {
      scf.for %m = %c0 to %c24 step %c1 {
        scf.for %n = %c0 to %c24 step %c1 {
          %g48 = arith.muli %g, %c48 : index
          %g48_m = arith.addi %g48, %m : index
          %g48_m_24 = arith.muli %g48_m, %c24 : index
          %src_idx = arith.addi %g48_m_24, %n : index
          %val_f16 = memref.load %src[%src_idx] : memref<4608xf16>
          %val_f32 = arith.extf %val_f16 : f16 to f32

          %g576 = arith.muli %g, %c576 : index
          %m24 = arith.muli %m, %c24 : index
          %gm = arith.addi %g576, %m24 : index
          %dst_idx = arith.addi %gm, %n : index
          memref.store %val_f32, %dst[%dst_idx] : memref<2304xf32>
        }
      }
    }
    return
  }

  // ---- Verification (non-padding elements only) ----
  func.func @verify_4608(%gpu: memref<4608xf16>, %ref: memref<4608xf16>) {
    %c1_i8 = arith.constant 1 : i8
    %gpu_packed = memref.alloc() : memref<2304xf32>
    call @_extract_nonpadding(%gpu, %gpu_packed) : (memref<4608xf16>, memref<2304xf32>) -> ()
    %cast_gpu = memref.cast %gpu_packed : memref<2304xf32> to memref<?xf32>
    %ref_packed = memref.alloc() : memref<2304xf32>
    call @_extract_nonpadding(%ref, %ref_packed) : (memref<4608xf16>, memref<2304xf32>) -> ()
    %cast_ref = memref.cast %ref_packed : memref<2304xf32> to memref<?xf32>
    %cst_rtol = arith.constant 1.000000e-03 : f32
    %cst_atol = arith.constant 1.000000e+02 : f32
    %cst_max = arith.constant 1.000000e+02 : f32
    %false = arith.constant false
    call @mcpuVerifyFloat(%cast_gpu, %cast_ref, %cst_rtol, %cst_atol, %cst_max, %c1_i8, %false)
      : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1) -> ()
    memref.dealloc %gpu_packed : memref<2304xf32>
    memref.dealloc %ref_packed : memref<2304xf32>
    return
  }

  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1)

  // ---- GPU wrapper ----
  func.func @mlir_dot_sigmoid_gpu(
      %arg0: memref<1536xf16>,
      %arg1: memref<1536xf16>,
      %arg2: memref<4608xf16>
  ) {
    %dev_A = gpu.alloc () : memref<1536xf16>
    gpu.memcpy %dev_A, %arg0 : memref<1536xf16>, memref<1536xf16>
    %dev_B = gpu.alloc () : memref<1536xf16>
    gpu.memcpy %dev_B, %arg1 : memref<1536xf16>, memref<1536xf16>
    %dev_out = gpu.alloc () : memref<4608xf16>
    gpu.memcpy %dev_out, %arg2 : memref<4608xf16>, memref<4608xf16>

    %t_A = bufferization.to_tensor %dev_A restrict writable : memref<1536xf16> to tensor<1536xf16>
    %t_B = bufferization.to_tensor %dev_B restrict writable : memref<1536xf16> to tensor<1536xf16>
    %t_out = bufferization.to_tensor %dev_out restrict writable : memref<4608xf16> to tensor<4608xf16>

    %result = call @mlir_dot_sigmoid(%t_A, %t_B, %t_out)
      : (tensor<1536xf16>, tensor<1536xf16>, tensor<4608xf16>) -> tensor<4608xf16>

    %buf_result = bufferization.to_buffer %result : tensor<4608xf16> to memref<4608xf16>
    memref.copy %buf_result, %dev_out : memref<4608xf16> to memref<4608xf16>

    gpu.memcpy %arg2, %dev_out : memref<4608xf16>, memref<4608xf16>
    gpu.dealloc %dev_A : memref<1536xf16>
    gpu.dealloc %dev_B : memref<1536xf16>
    gpu.dealloc %dev_out : memref<4608xf16>
    return
  }
}
