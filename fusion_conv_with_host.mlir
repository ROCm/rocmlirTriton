// Fused conv test: output = conv(filter, input + inputfusion) + ofusion
// Conv params: N=2, C=8, K=4, H=W=8, filter 3x3, pad=1, stride=1
// The arch (gfx1100) and num_cu (96) are placeholders; tests.sh substitutes
// them with the detected GPU values via sed before compilation.
#map = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 3 + d2) * 3 + d3) * 8 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 8 + d2) * 8 + d3) * 8 + d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)>
#map3 = affine_map<(d0) -> (d0 mod 3)>
#map4 = affine_map<(d0) -> (0)>
#map5 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 3, 3, 8} ["k", "0", "1", "c"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 3, 3, 8] -> [288]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{2, 8, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 8, 8, 8] -> [1024]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{2, 4, 8, 8} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // ---- GPU kernel with input and output fusions ----
  // Computes: output = conv(filter, input + inputfusion) + ofusion
  func.func @rock_conv(%arg0: tensor<288xf16>, %arg1: tensor<1024xf16>, %inputfusion: tensor<1024xf16>, %arg2: tensor<512xf16>, %arg3: tensor<512xf16>) -> tensor<512xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<288xf16> to tensor<1x4x3x3x8xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<1024xf16> to tensor<2x1x8x8x8xf16>
    %input = rock.transform %inputfusion by #transform_map1 : tensor<1024xf16> to tensor<2x1x8x8x8xf16>
    %a = arith.addf %1, %input : tensor<2x1x8x8x8xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<512xf16> to tensor<2x1x4x8x8xf16>
    %3 = rock.transform %arg3 by #transform_map2 : tensor<512xf16> to tensor<2x1x4x8x8xf16>
    %4 = rock.conv(%0, %a, %3) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : tensor<1x4x3x3x8xf16>, tensor<2x1x8x8x8xf16>, tensor<2x1x4x8x8xf16> -> tensor<2x1x4x8x8xf16>
    %fusion = arith.addf %4, %2 : tensor<2x1x4x8x8xf16>
    %5 = rock.store %fusion to %3 by  set : tensor<2x1x4x8x8xf16> -> tensor<512xf16> to tensor<2x1x4x8x8xf16>
    return %5 : tensor<512xf16>
  }

  // ---- Main entry point ----
  func.func @main() {
    // Allocate and initialize 5 buffers: filter, input, inputfusion, ofusion, output
    %alloc_filter = memref.alloc() : memref<288xf16>
    call @_init_buffer_288(%alloc_filter) : (memref<288xf16>) -> ()
    %alloc_filter_ref = memref.alloc() : memref<288xf16>
    memref.copy %alloc_filter, %alloc_filter_ref : memref<288xf16> to memref<288xf16>

    %alloc_input = memref.alloc() : memref<1024xf16>
    call @_init_buffer_1024(%alloc_input) : (memref<1024xf16>) -> ()
    %alloc_input_ref = memref.alloc() : memref<1024xf16>
    memref.copy %alloc_input, %alloc_input_ref : memref<1024xf16> to memref<1024xf16>

    %alloc_inputfusion = memref.alloc() : memref<1024xf16>
    call @_init_buffer_1024(%alloc_inputfusion) : (memref<1024xf16>) -> ()
    %alloc_inputfusion_ref = memref.alloc() : memref<1024xf16>
    memref.copy %alloc_inputfusion, %alloc_inputfusion_ref : memref<1024xf16> to memref<1024xf16>

    %alloc_ofusion = memref.alloc() : memref<512xf16>
    call @_init_buffer_512(%alloc_ofusion) : (memref<512xf16>) -> ()
    %alloc_ofusion_ref = memref.alloc() : memref<512xf16>
    memref.copy %alloc_ofusion, %alloc_ofusion_ref : memref<512xf16> to memref<512xf16>

    %alloc_output = memref.alloc() : memref<512xf16>
    call @_init_buffer_512(%alloc_output) : (memref<512xf16>) -> ()
    %alloc_output_ref = memref.alloc() : memref<512xf16>
    memref.copy %alloc_output, %alloc_output_ref : memref<512xf16> to memref<512xf16>

    // Run GPU kernel
    call @rock_conv_gpu(%alloc_filter, %alloc_input, %alloc_inputfusion, %alloc_ofusion, %alloc_output) : (memref<288xf16>, memref<1024xf16>, memref<1024xf16>, memref<512xf16>, memref<512xf16>) -> ()

    // Run CPU reference
    call @host_naive_fused_conv(%alloc_filter_ref, %alloc_input_ref, %alloc_inputfusion_ref, %alloc_ofusion_ref, %alloc_output_ref) : (memref<288xf16>, memref<1024xf16>, memref<1024xf16>, memref<512xf16>, memref<512xf16>) -> ()

    // Verify GPU output against CPU reference
    call @rock_conv_verify2(%alloc_output, %alloc_output_ref) : (memref<512xf16>, memref<512xf16>) -> ()

    // Cleanup
    memref.dealloc %alloc_filter_ref : memref<288xf16>
    memref.dealloc %alloc_input_ref : memref<1024xf16>
    memref.dealloc %alloc_inputfusion_ref : memref<1024xf16>
    memref.dealloc %alloc_ofusion_ref : memref<512xf16>
    memref.dealloc %alloc_output_ref : memref<512xf16>
    memref.dealloc %alloc_filter : memref<288xf16>
    memref.dealloc %alloc_input : memref<1024xf16>
    memref.dealloc %alloc_inputfusion : memref<1024xf16>
    memref.dealloc %alloc_ofusion : memref<512xf16>
    memref.dealloc %alloc_output : memref<512xf16>
    return
  }

  // ---- Buffer initialization: cycling pattern [0.5, -1.0, 0.75] ----
  func.func @_init_buffer_288(%buf: memref<288xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf16>
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = vector.insert %cst_0, %cst [0] : f16 into vector<3xf16>
    %cst_1 = arith.constant -1.000000e+00 : f16
    %1 = vector.insert %cst_1, %0 [1] : f16 into vector<3xf16>
    %cst_2 = arith.constant 7.500000e-01 : f16
    %2 = vector.insert %cst_2, %1 [2] : f16 into vector<3xf16>
    affine.for %arg0 = 0 to 288 {
      %3 = affine.apply #map3(%arg0)
      %4 = vector.extract %2[%3] : f16 from vector<3xf16>
      memref.store %4, %buf[%arg0] : memref<288xf16>
    }
    return
  }

  func.func @_init_buffer_1024(%buf: memref<1024xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf16>
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = vector.insert %cst_0, %cst [0] : f16 into vector<3xf16>
    %cst_1 = arith.constant -1.000000e+00 : f16
    %1 = vector.insert %cst_1, %0 [1] : f16 into vector<3xf16>
    %cst_2 = arith.constant 7.500000e-01 : f16
    %2 = vector.insert %cst_2, %1 [2] : f16 into vector<3xf16>
    affine.for %arg0 = 0 to 1024 {
      %3 = affine.apply #map3(%arg0)
      %4 = vector.extract %2[%3] : f16 from vector<3xf16>
      memref.store %4, %buf[%arg0] : memref<1024xf16>
    }
    return
  }

  func.func @_init_buffer_512(%buf: memref<512xf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<3xf16>
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = vector.insert %cst_0, %cst [0] : f16 into vector<3xf16>
    %cst_1 = arith.constant -1.000000e+00 : f16
    %1 = vector.insert %cst_1, %0 [1] : f16 into vector<3xf16>
    %cst_2 = arith.constant 7.500000e-01 : f16
    %2 = vector.insert %cst_2, %1 [2] : f16 into vector<3xf16>
    affine.for %arg0 = 0 to 512 {
      %3 = affine.apply #map3(%arg0)
      %4 = vector.extract %2[%3] : f16 from vector<3xf16>
      memref.store %4, %buf[%arg0] : memref<512xf16>
    }
    return
  }

  // ---- CPU reference: fused conv ----
  // Computes: output = conv(filter, input + inputfusion) + ofusion
  // All arithmetic done in f32 for accuracy.
  func.func @host_naive_fused_conv(%arg0: memref<288xf16>, %arg1: memref<1024xf16>, %arg2: memref<1024xf16>, %arg3: memref<512xf16>, %arg4: memref<512xf16>) {
    // Convert filter, input, inputfusion, ofusion to f32
    %filter_f32 = memref.alloc() : memref<288xf32>
    call @_memcpy_f16_f32_288(%arg0, %filter_f32) : (memref<288xf16>, memref<288xf32>) -> ()
    %input_f32 = memref.alloc() : memref<1024xf32>
    call @_memcpy_f16_f32_1024(%arg1, %input_f32) : (memref<1024xf16>, memref<1024xf32>) -> ()
    %inputfusion_f32 = memref.alloc() : memref<1024xf32>
    call @_memcpy_f16_f32_1024(%arg2, %inputfusion_f32) : (memref<1024xf16>, memref<1024xf32>) -> ()
    %ofusion_f32 = memref.alloc() : memref<512xf32>
    call @_memcpy_f16_f32_512(%arg3, %ofusion_f32) : (memref<512xf16>, memref<512xf32>) -> ()
    %output_f32 = memref.alloc() : memref<512xf32>

    // Input fusion: input' = input + inputfusion (elementwise on flat buffers)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    scf.for %i = %c0 to %c1024 step %c1 {
      %in_val = memref.load %input_f32[%i] : memref<1024xf32>
      %fuse_val = memref.load %inputfusion_f32[%i] : memref<1024xf32>
      %fused = arith.addf %in_val, %fuse_val : f32
      memref.store %fused, %input_f32[%i] : memref<1024xf32>
    }

    // Zero-init output for conv accumulation
    %cst_zero = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst_zero : f32) outs(%output_f32 : memref<512xf32>)

    // Conv: output[n,g,k,oh,ow] += filter[g,k,fh,fw,c] * input'[n,g,c,ih,iw]
    // Layout: filter=gk01c, input=ngc01, output=ngk01
    // padding=[1,1,1,1], strides=[1,1], dilations=[1,1]
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 1 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 8 {
            affine.for %arg9 = 0 to 8 {
              affine.for %arg10 = 0 to 8 {
                affine.for %arg11 = 0 to 3 {
                  affine.for %arg12 = 0 to 3 {
                    %1 = affine.apply #map5(%arg8, %arg11)
                    %2 = affine.apply #map5(%arg9, %arg12)
                    affine.if #set(%1, %2) {
                      %3 = affine.load %filter_f32[(((%arg6 * 4 + %arg7) * 3 + %arg11) * 3 + %arg12) * 8 + %arg10] : memref<288xf32>
                      %4 = affine.load %input_f32[(((%arg5 + %arg6) * 8 + %arg10) * 8 + %1) * 8 + %2] : memref<1024xf32>
                      %5 = affine.load %output_f32[(((%arg5 + %arg6) * 4 + %arg7) * 8 + %arg8) * 8 + %arg9] : memref<512xf32>
                      %6 = arith.mulf %3, %4 : f32
                      %7 = arith.addf %5, %6 : f32
                      affine.store %7, %output_f32[(((%arg5 + %arg6) * 4 + %arg7) * 8 + %arg8) * 8 + %arg9] : memref<512xf32>
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // Output fusion: output = output + ofusion (elementwise on flat buffers)
    scf.for %i = %c0 to %c512 step %c1 {
      %out_val = memref.load %output_f32[%i] : memref<512xf32>
      %o_val = memref.load %ofusion_f32[%i] : memref<512xf32>
      %fused = arith.addf %out_val, %o_val : f32
      memref.store %fused, %output_f32[%i] : memref<512xf32>
    }

    // Convert result back to f16
    call @_memcpy_f32_f16_512(%output_f32, %arg4) : (memref<512xf32>, memref<512xf16>) -> ()

    // Cleanup
    memref.dealloc %filter_f32 : memref<288xf32>
    memref.dealloc %input_f32 : memref<1024xf32>
    memref.dealloc %inputfusion_f32 : memref<1024xf32>
    memref.dealloc %ofusion_f32 : memref<512xf32>
    memref.dealloc %output_f32 : memref<512xf32>
    return
  }

  // ---- Helper: f16 -> f32 copy ----
  func.func @_memcpy_f16_f32_288(%arg0: memref<288xf16>, %arg1: memref<288xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<288xf16>
    scf.for %arg2 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<288xf16>
      %1 = arith.extf %0 : f16 to f32
      memref.store %1, %arg1[%arg2] : memref<288xf32>
    }
    return
  }

  func.func @_memcpy_f16_f32_1024(%arg0: memref<1024xf16>, %arg1: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<1024xf16>
    scf.for %arg2 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<1024xf16>
      %1 = arith.extf %0 : f16 to f32
      memref.store %1, %arg1[%arg2] : memref<1024xf32>
    }
    return
  }

  func.func @_memcpy_f16_f32_512(%arg0: memref<512xf16>, %arg1: memref<512xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<512xf16>
    scf.for %arg2 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<512xf16>
      %1 = arith.extf %0 : f16 to f32
      memref.store %1, %arg1[%arg2] : memref<512xf32>
    }
    return
  }

  // ---- Helper: f32 -> f16 copy ----
  func.func @_memcpy_f32_f16_512(%arg0: memref<512xf32>, %arg1: memref<512xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<512xf32>
    scf.for %arg2 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<512xf32>
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %arg1[%arg2] : memref<512xf16>
    }
    return
  }

  // ---- Verification ----
  func.func @rock_conv_verify2(%arg0: memref<512xf16>, %arg1: memref<512xf16>) {
    %c1_i8 = arith.constant 1 : i8
    %alloc = memref.alloc() : memref<512xf32>
    call @_memcpy_f16_f32_512(%arg0, %alloc) : (memref<512xf16>, memref<512xf32>) -> ()
    %cast = memref.cast %alloc : memref<512xf32> to memref<?xf32>
    %alloc_0 = memref.alloc() : memref<512xf32>
    call @_memcpy_f16_f32_512(%arg1, %alloc_0) : (memref<512xf16>, memref<512xf32>) -> ()
    %cast_1 = memref.cast %alloc_0 : memref<512xf32> to memref<?xf32>
    %cst = arith.constant 1.000000e-03 : f32
    %cst_2 = arith.constant 1.000000e+02 : f32
    %cst_3 = arith.constant 9.99999997E-7 : f32
    %cst_4 = arith.constant 1.000000e+02 : f32
    %false = arith.constant false
    call @mcpuVerifyFloat(%cast, %cast_1, %cst, %cst_2, %cst_4, %c1_i8, %false) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1) -> ()
    memref.dealloc %alloc : memref<512xf32>
    memref.dealloc %alloc_0 : memref<512xf32>
    return
  }

  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1)

  // ---- GPU wrapper ----
  func.func @rock_conv_gpu(%arg0: memref<288xf16>, %arg1: memref<1024xf16>, %arg2: memref<1024xf16>, %arg3: memref<512xf16>, %arg4: memref<512xf16>) {
    // Allocate GPU buffers and copy host -> device
    %memref_filter = gpu.alloc  () : memref<288xf16>
    gpu.memcpy  %memref_filter, %arg0 : memref<288xf16>, memref<288xf16>
    %memref_input = gpu.alloc  () : memref<1024xf16>
    gpu.memcpy  %memref_input, %arg1 : memref<1024xf16>, memref<1024xf16>
    %memref_inputfusion = gpu.alloc  () : memref<1024xf16>
    gpu.memcpy  %memref_inputfusion, %arg2 : memref<1024xf16>, memref<1024xf16>
    %memref_ofusion = gpu.alloc  () : memref<512xf16>
    gpu.memcpy  %memref_ofusion, %arg3 : memref<512xf16>, memref<512xf16>
    %memref_output = gpu.alloc  () : memref<512xf16>
    gpu.memcpy  %memref_output, %arg4 : memref<512xf16>, memref<512xf16>

    // Convert to tensors for the kernel
    %0 = bufferization.to_tensor %memref_filter restrict writable : memref<288xf16> to tensor<288xf16>
    %1 = bufferization.to_tensor %memref_input restrict writable : memref<1024xf16> to tensor<1024xf16>
    %2 = bufferization.to_tensor %memref_inputfusion restrict writable : memref<1024xf16> to tensor<1024xf16>
    %3 = bufferization.to_tensor %memref_ofusion restrict writable : memref<512xf16> to tensor<512xf16>
    %4 = bufferization.to_tensor %memref_output restrict writable : memref<512xf16> to tensor<512xf16>

    // Call the GPU kernel
    %5 = call @rock_conv(%0, %1, %2, %3, %4) : (tensor<288xf16>, tensor<1024xf16>, tensor<1024xf16>, tensor<512xf16>, tensor<512xf16>) -> tensor<512xf16>

    // Copy output back
    %6 = bufferization.to_buffer %5 : tensor<512xf16> to memref<512xf16>
    memref.copy %6, %memref_output : memref<512xf16> to memref<512xf16>

    // Copy device -> host and deallocate
    gpu.memcpy  %arg0, %memref_filter : memref<288xf16>, memref<288xf16>
    gpu.dealloc  %memref_filter : memref<288xf16>
    gpu.memcpy  %arg1, %memref_input : memref<1024xf16>, memref<1024xf16>
    gpu.dealloc  %memref_input : memref<1024xf16>
    gpu.memcpy  %arg2, %memref_inputfusion : memref<1024xf16>, memref<1024xf16>
    gpu.dealloc  %memref_inputfusion : memref<1024xf16>
    gpu.memcpy  %arg3, %memref_ofusion : memref<512xf16>, memref<512xf16>
    gpu.dealloc  %memref_ofusion : memref<512xf16>
    gpu.memcpy  %arg4, %memref_output : memref<512xf16>, memref<512xf16>
    gpu.dealloc  %memref_output : memref<512xf16>
    return
  }
}
