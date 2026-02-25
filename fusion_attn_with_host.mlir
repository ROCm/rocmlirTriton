#map = affine_map<(d0, d1, d2) -> ((d0 * 70 + d1) * 70 + d2)>
#map1 = affine_map<(d0, d1) -> (d0 * 70 + d1)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 70, 70} ["g", "seq_q", "head_qk"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 70, 70] -> [19600]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{2, 70, 70} ["g", "head_qk", "seq_k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 70, 70] -> [9800]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{2, 70, 70} ["g", "seq_k", "head_v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 70, 70] -> [9800]>
#transform_map3 = #rock.transform_map<#map by [<Unmerge{4, 70, 70} ["g", "seq_q", "seq_k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 70, 70] -> [19600]>
#transform_map4 = #rock.transform_map<#map1 by [<Unmerge{4, 70} ["g", "seq_q"] at [0, 1] -> ["raw"] at [0]>] bounds = [4, 70] -> [280]>
#transform_map5 = #rock.transform_map<#map by [<Unmerge{4, 70, 70} ["g", "seq_q", "head_v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 70, 70] -> [19600]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // Kernel: attention with input fusion on Q, preSoftmax bias, and output fusions on result and LSE.
  // Args: Q, fusionInput, K, V, bias, fusion4(lse), fusion5(result), lse_out, result_out
  func.func @rock_attention(%arg0: tensor<19600xf32>, %argFusion: tensor<19600xf32>, %arg1: tensor<9800xf32>, %arg2: tensor<9800xf32>, %arg3: tensor<19600xf32>, %arg4Fusion: tensor<280xf32>, %arg5Fusion: tensor<19600xf32>, %arg4: tensor<280xf32>, %arg5: tensor<19600xf32>) -> (tensor<19600xf32>, tensor<280xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<19600xf32> to tensor<4x70x70xf32>
    %fusionInput = rock.transform %argFusion by #transform_map : tensor<19600xf32> to tensor<4x70x70xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<9800xf32> to tensor<2x70x70xf32>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<9800xf32> to tensor<2x70x70xf32>
    %3 = rock.transform %arg3 by #transform_map3 : tensor<19600xf32> to tensor<4x70x70xf32>
    %fusion4 = rock.transform %arg4Fusion by #transform_map4 : tensor<280xf32> to tensor<4x70xf32>
    %fusion5 = rock.transform %arg5Fusion by #transform_map5 : tensor<19600xf32> to tensor<4x70x70xf32>
    %4 = rock.transform %arg4 by #transform_map4 : tensor<280xf32> to tensor<4x70xf32>
    %5 = rock.transform %arg5 by #transform_map5 : tensor<19600xf32> to tensor<4x70x70xf32>
    %q = arith.addf %0, %fusionInput : tensor<4x70x70xf32>
    %result, %lse = rock.attention{
     qk = %q * %1 : tensor<4x70x70xf32>, tensor<2x70x70xf32>
     qk = elementwise otherIns(%3 : tensor<4x70x70xf32>) {
    ^bb0(%arg6: tensor<4x70x70xf32>, %arg7: tensor<4x70x70xf32>):
      %8 = arith.addf %arg6, %arg7 : tensor<4x70x70xf32>
      rock.yield %8 : tensor<4x70x70xf32>
    }
     softmax(qk) * %2 : tensor<2x70x70xf32>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 2 : i32, numHeadsQ = 4 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<4x70x70xf32>, tensor<4x70xf32>
    %lseFinal = arith.addf %fusion4, %lse : tensor<4x70xf32>
    %resultFinal = arith.addf %fusion5, %result : tensor<4x70x70xf32>
    %6 = rock.store %resultFinal to %5 by  set : tensor<4x70x70xf32> -> tensor<19600xf32> to tensor<4x70x70xf32>
    %7 = rock.store %lseFinal to %4 by  set : tensor<4x70xf32> -> tensor<280xf32> to tensor<4x70xf32>
    return %6, %7 : tensor<19600xf32>, tensor<280xf32>
  }

  func.func @main() {
    %c1_i32 = arith.constant 1 : i32
    call @seedRandomValues(%c1_i32) : (i32) -> ()

    // arg0: Q [19600xf32]
    %alloc_q = memref.alloc() : memref<19600xf32>
    %c-5_i16 = arith.constant -5 : i16
    %c5_i16 = arith.constant 5 : i16
    affine.for %i = 0 to 19600 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_q[%i] : memref<19600xf32>
    }
    %alloc_q_host = memref.alloc() : memref<19600xf32>
    memref.copy %alloc_q, %alloc_q_host : memref<19600xf32> to memref<19600xf32>

    // argFusion: input fusion on Q [19600xf32]
    %alloc_fusionQ = memref.alloc() : memref<19600xf32>
    affine.for %i = 0 to 19600 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_fusionQ[%i] : memref<19600xf32>
    }
    %alloc_fusionQ_host = memref.alloc() : memref<19600xf32>
    memref.copy %alloc_fusionQ, %alloc_fusionQ_host : memref<19600xf32> to memref<19600xf32>

    // arg1: K [9800xf32]
    %alloc_k = memref.alloc() : memref<9800xf32>
    affine.for %i = 0 to 9800 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_k[%i] : memref<9800xf32>
    }
    %alloc_k_host = memref.alloc() : memref<9800xf32>
    memref.copy %alloc_k, %alloc_k_host : memref<9800xf32> to memref<9800xf32>

    // arg2: V [9800xf32]
    %alloc_v = memref.alloc() : memref<9800xf32>
    affine.for %i = 0 to 9800 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_v[%i] : memref<9800xf32>
    }
    %alloc_v_host = memref.alloc() : memref<9800xf32>
    memref.copy %alloc_v, %alloc_v_host : memref<9800xf32> to memref<9800xf32>

    // arg3: bias [19600xf32]
    %alloc_bias = memref.alloc() : memref<19600xf32>
    affine.for %i = 0 to 19600 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_bias[%i] : memref<19600xf32>
    }
    %alloc_bias_host = memref.alloc() : memref<19600xf32>
    memref.copy %alloc_bias, %alloc_bias_host : memref<19600xf32> to memref<19600xf32>

    // arg4Fusion: output fusion on LSE [280xf32]
    %alloc_fusion4 = memref.alloc() : memref<280xf32>
    affine.for %i = 0 to 280 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_fusion4[%i] : memref<280xf32>
    }
    %alloc_fusion4_host = memref.alloc() : memref<280xf32>
    memref.copy %alloc_fusion4, %alloc_fusion4_host : memref<280xf32> to memref<280xf32>

    // arg5Fusion: output fusion on result [19600xf32]
    %alloc_fusion5 = memref.alloc() : memref<19600xf32>
    affine.for %i = 0 to 19600 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_fusion5[%i] : memref<19600xf32>
    }
    %alloc_fusion5_host = memref.alloc() : memref<19600xf32>
    memref.copy %alloc_fusion5, %alloc_fusion5_host : memref<19600xf32> to memref<19600xf32>

    // arg4: LSE output [280xf32]
    %alloc_lse = memref.alloc() : memref<280xf32>
    affine.for %i = 0 to 280 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_lse[%i] : memref<280xf32>
    }
    %alloc_lse_host = memref.alloc() : memref<280xf32>
    memref.copy %alloc_lse, %alloc_lse_host : memref<280xf32> to memref<280xf32>

    // arg5: result output [19600xf32]
    %alloc_out = memref.alloc() : memref<19600xf32>
    affine.for %i = 0 to 19600 {
      %v = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      memref.store %v, %alloc_out[%i] : memref<19600xf32>
    }
    %alloc_out_host = memref.alloc() : memref<19600xf32>
    memref.copy %alloc_out, %alloc_out_host : memref<19600xf32> to memref<19600xf32>

    call @rock_attention_gpu(%alloc_q, %alloc_fusionQ, %alloc_k, %alloc_v, %alloc_bias, %alloc_fusion4, %alloc_fusion5, %alloc_lse, %alloc_out)
      : (memref<19600xf32>, memref<19600xf32>, memref<9800xf32>, memref<9800xf32>, memref<19600xf32>, memref<280xf32>, memref<19600xf32>, memref<280xf32>, memref<19600xf32>) -> ()
    call @host_naive_attention(%alloc_q_host, %alloc_fusionQ_host, %alloc_k_host, %alloc_v_host, %alloc_bias_host, %alloc_fusion4_host, %alloc_fusion5_host, %alloc_lse_host, %alloc_out_host)
      : (memref<19600xf32>, memref<19600xf32>, memref<9800xf32>, memref<9800xf32>, memref<19600xf32>, memref<280xf32>, memref<19600xf32>, memref<280xf32>, memref<19600xf32>) -> ()
    call @verify_f32(%alloc_out, %alloc_out_host) : (memref<19600xf32>, memref<19600xf32>) -> ()

    memref.dealloc %alloc_q : memref<19600xf32>
    memref.dealloc %alloc_q_host : memref<19600xf32>
    memref.dealloc %alloc_fusionQ : memref<19600xf32>
    memref.dealloc %alloc_fusionQ_host : memref<19600xf32>
    memref.dealloc %alloc_k : memref<9800xf32>
    memref.dealloc %alloc_k_host : memref<9800xf32>
    memref.dealloc %alloc_v : memref<9800xf32>
    memref.dealloc %alloc_v_host : memref<9800xf32>
    memref.dealloc %alloc_bias : memref<19600xf32>
    memref.dealloc %alloc_bias_host : memref<19600xf32>
    memref.dealloc %alloc_fusion4 : memref<280xf32>
    memref.dealloc %alloc_fusion4_host : memref<280xf32>
    memref.dealloc %alloc_fusion5 : memref<19600xf32>
    memref.dealloc %alloc_fusion5_host : memref<19600xf32>
    memref.dealloc %alloc_lse : memref<280xf32>
    memref.dealloc %alloc_lse_host : memref<280xf32>
    memref.dealloc %alloc_out : memref<19600xf32>
    memref.dealloc %alloc_out_host : memref<19600xf32>
    return
  }

  func.func private @seedRandomValues(i32)
  func.func private @randomIntegerValue(i16, i16) -> f32

  // CPU reference: attention with all fusions
  // Args: Q, fusionInput, K, V, bias, fusion4(lse), fusion5(result), lse_out, result_out
  func.func @host_naive_attention(%arg0: memref<19600xf32>, %arg1: memref<19600xf32>, %arg2: memref<9800xf32>, %arg3: memref<9800xf32>, %arg4: memref<19600xf32>, %arg5: memref<280xf32>, %arg6: memref<19600xf32>, %arg7: memref<280xf32>, %arg8: memref<19600xf32>) {
    // Load inputs
    %tQ = bufferization.to_tensor %arg0 restrict : memref<19600xf32> to tensor<19600xf32>
    %shape3d = tosa.const_shape {values = dense<[4, 70, 70]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %Q = tosa.reshape %tQ, %shape3d : (tensor<19600xf32>, !tosa.shape<3>) -> tensor<4x70x70xf32>

    %tFusionQ = bufferization.to_tensor %arg1 restrict : memref<19600xf32> to tensor<19600xf32>
    %fusionQ = tosa.reshape %tFusionQ, %shape3d : (tensor<19600xf32>, !tosa.shape<3>) -> tensor<4x70x70xf32>

    // Input fusion: Q = Q + fusionInput
    %Qfused = tosa.add %Q, %fusionQ : (tensor<4x70x70xf32>, tensor<4x70x70xf32>) -> tensor<4x70x70xf32>

    %tK = bufferization.to_tensor %arg2 restrict : memref<9800xf32> to tensor<9800xf32>
    %shape3d_kv = tosa.const_shape {values = dense<[2, 70, 70]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %K = tosa.reshape %tK, %shape3d_kv : (tensor<9800xf32>, !tosa.shape<3>) -> tensor<2x70x70xf32>

    %tV = bufferization.to_tensor %arg3 restrict : memref<9800xf32> to tensor<9800xf32>
    %V = tosa.reshape %tV, %shape3d_kv : (tensor<9800xf32>, !tosa.shape<3>) -> tensor<2x70x70xf32>

    // GQA broadcast K: [2,70,70] -> [2,1,70,70] -> [2,2,70,70] -> [4,70,70]
    %K_exp = tensor.expand_shape %K [[0, 1], [2], [3]] output_shape [2, 1, 70, 70] : tensor<2x70x70xf32> into tensor<2x1x70x70xf32>
    %ones_k = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x70x70xf32>}> : () -> tensor<2x2x70x70xf32>
    %shift0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %K_bc = tosa.mul %K_exp, %ones_k, %shift0 : (tensor<2x1x70x70xf32>, tensor<2x2x70x70xf32>, tensor<1xi8>) -> tensor<2x2x70x70xf32>
    %K4 = tensor.collapse_shape %K_bc [[0, 1], [2], [3]] : tensor<2x2x70x70xf32> into tensor<4x70x70xf32>

    // GQA broadcast V: [2,70,70] -> [4,70,70]
    %V_exp = tensor.expand_shape %V [[0, 1], [2], [3]] output_shape [2, 1, 70, 70] : tensor<2x70x70xf32> into tensor<2x1x70x70xf32>
    %ones_v = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x2x70x70xf32>}> : () -> tensor<2x2x70x70xf32>
    %V_bc = tosa.mul %V_exp, %ones_v, %shift0 : (tensor<2x1x70x70xf32>, tensor<2x2x70x70xf32>, tensor<1xi8>) -> tensor<2x2x70x70xf32>
    %V4 = tensor.collapse_shape %V_bc [[0, 1], [2], [3]] : tensor<2x2x70x70xf32> into tensor<4x70x70xf32>

    // QK = Qfused * K4
    %zero_bias = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %QK = tosa.matmul %Qfused, %K4, %zero_bias, %zero_bias {acc_type = f32} : (tensor<4x70x70xf32>, tensor<4x70x70xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x70x70xf32>

    // PreSoftmax elementwise: QK = QK + bias
    %tBias = bufferization.to_tensor %arg4 restrict : memref<19600xf32> to tensor<19600xf32>
    %bias = tosa.reshape %tBias, %shape3d : (tensor<19600xf32>, !tosa.shape<3>) -> tensor<4x70x70xf32>
    %QK_biased = tosa.add %QK, %bias : (tensor<4x70x70xf32>, tensor<4x70x70xf32>) -> tensor<4x70x70xf32>

    // Softmax (in f32)
    %row_max = tosa.reduce_max %QK_biased {axis = 2 : i32} : (tensor<4x70x70xf32>) -> tensor<4x70x1xf32>
    %sub_max = tosa.sub %QK_biased, %row_max : (tensor<4x70x70xf32>, tensor<4x70x1xf32>) -> tensor<4x70x70xf32>
    %exp_val = tosa.exp %sub_max : (tensor<4x70x70xf32>) -> tensor<4x70x70xf32>
    %row_sum = tosa.reduce_sum %exp_val {axis = 2 : i32} : (tensor<4x70x70xf32>) -> tensor<4x70x1xf32>
    %inv_sum = tosa.reciprocal %row_sum : (tensor<4x70x1xf32>) -> tensor<4x70x1xf32>
    %softmax_out = tosa.mul %exp_val, %inv_sum, %shift0 : (tensor<4x70x70xf32>, tensor<4x70x1xf32>, tensor<1xi8>) -> tensor<4x70x70xf32>

    // result = softmax * V4
    %result = tosa.matmul %softmax_out, %V4, %zero_bias, %zero_bias {acc_type = f32} : (tensor<4x70x70xf32>, tensor<4x70x70xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x70x70xf32>

    // LSE = log(row_sum) + row_max  [4,70,1]
    %log_sum = tosa.log %row_sum : (tensor<4x70x1xf32>) -> tensor<4x70x1xf32>
    %lse_3d = tosa.add %log_sum, %row_max : (tensor<4x70x1xf32>, tensor<4x70x1xf32>) -> tensor<4x70x1xf32>

    // Output fusion on LSE: lseFinal = fusion4 + lse
    %tFusion4 = bufferization.to_tensor %arg5 restrict : memref<280xf32> to tensor<280xf32>
    %shape2d = tosa.const_shape {values = dense<[4, 70]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %fusion4 = tosa.reshape %tFusion4, %shape2d : (tensor<280xf32>, !tosa.shape<2>) -> tensor<4x70xf32>
    %shape2d_1 = tosa.const_shape {values = dense<[4, 70]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %lse_2d = tosa.reshape %lse_3d, %shape2d_1 : (tensor<4x70x1xf32>, !tosa.shape<2>) -> tensor<4x70xf32>
    %lseFinal = tosa.add %fusion4, %lse_2d : (tensor<4x70xf32>, tensor<4x70xf32>) -> tensor<4x70xf32>

    // Output fusion on result: resultFinal = fusion5 + result
    %tFusion5 = bufferization.to_tensor %arg6 restrict : memref<19600xf32> to tensor<19600xf32>
    %fusion5 = tosa.reshape %tFusion5, %shape3d : (tensor<19600xf32>, !tosa.shape<3>) -> tensor<4x70x70xf32>
    %resultFinal = tosa.add %fusion5, %result : (tensor<4x70x70xf32>, tensor<4x70x70xf32>) -> tensor<4x70x70xf32>

    // Store results
    %shape1d_out = tosa.const_shape {values = dense<19600> : tensor<1xindex>} : () -> !tosa.shape<1>
    %result_flat = tosa.reshape %resultFinal, %shape1d_out : (tensor<4x70x70xf32>, !tosa.shape<1>) -> tensor<19600xf32>
    %result_buf = bufferization.to_buffer %result_flat : tensor<19600xf32> to memref<19600xf32>
    memref.copy %result_buf, %arg8 : memref<19600xf32> to memref<19600xf32>

    %shape1d_lse = tosa.const_shape {values = dense<280> : tensor<1xindex>} : () -> !tosa.shape<1>
    %lse_flat = tosa.reshape %lseFinal, %shape1d_lse : (tensor<4x70xf32>, !tosa.shape<1>) -> tensor<280xf32>
    %lse_buf = bufferization.to_buffer %lse_flat : tensor<280xf32> to memref<280xf32>
    memref.copy %lse_buf, %arg7 : memref<280xf32> to memref<280xf32>
    return
  }

  func.func @verify_f32(%arg0: memref<19600xf32>, %arg1: memref<19600xf32>) {
    %c1_i8 = arith.constant 1 : i8
    %cast0 = memref.cast %arg0 : memref<19600xf32> to memref<?xf32>
    %cast1 = memref.cast %arg1 : memref<19600xf32> to memref<?xf32>
    %cst = arith.constant 1.000000e-03 : f32
    %cst_2 = arith.constant 1.000000e+02 : f32
    %cst_4 = arith.constant 1.000000e+02 : f32
    %false = arith.constant false
    call @mcpuVerifyFloat(%cast0, %cast1, %cst, %cst_2, %cst_4, %c1_i8, %false) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1) -> ()
    return
  }

  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1)

  func.func @rock_attention_gpu(%arg0: memref<19600xf32>, %arg1: memref<19600xf32>, %arg2: memref<9800xf32>, %arg3: memref<9800xf32>, %arg4: memref<19600xf32>, %arg5: memref<280xf32>, %arg6: memref<19600xf32>, %arg7: memref<280xf32>, %arg8: memref<19600xf32>) {
    %m0 = gpu.alloc () : memref<19600xf32>
    gpu.memcpy %m0, %arg0 : memref<19600xf32>, memref<19600xf32>
    %m1 = gpu.alloc () : memref<19600xf32>
    gpu.memcpy %m1, %arg1 : memref<19600xf32>, memref<19600xf32>
    %m2 = gpu.alloc () : memref<9800xf32>
    gpu.memcpy %m2, %arg2 : memref<9800xf32>, memref<9800xf32>
    %m3 = gpu.alloc () : memref<9800xf32>
    gpu.memcpy %m3, %arg3 : memref<9800xf32>, memref<9800xf32>
    %m4 = gpu.alloc () : memref<19600xf32>
    gpu.memcpy %m4, %arg4 : memref<19600xf32>, memref<19600xf32>
    %m5 = gpu.alloc () : memref<280xf32>
    gpu.memcpy %m5, %arg5 : memref<280xf32>, memref<280xf32>
    %m6 = gpu.alloc () : memref<19600xf32>
    gpu.memcpy %m6, %arg6 : memref<19600xf32>, memref<19600xf32>
    %m7 = gpu.alloc () : memref<280xf32>
    gpu.memcpy %m7, %arg7 : memref<280xf32>, memref<280xf32>
    %m8 = gpu.alloc () : memref<19600xf32>
    gpu.memcpy %m8, %arg8 : memref<19600xf32>, memref<19600xf32>

    %t0 = bufferization.to_tensor %m0 restrict writable : memref<19600xf32> to tensor<19600xf32>
    %t1 = bufferization.to_tensor %m1 restrict writable : memref<19600xf32> to tensor<19600xf32>
    %t2 = bufferization.to_tensor %m2 restrict writable : memref<9800xf32> to tensor<9800xf32>
    %t3 = bufferization.to_tensor %m3 restrict writable : memref<9800xf32> to tensor<9800xf32>
    %t4 = bufferization.to_tensor %m4 restrict writable : memref<19600xf32> to tensor<19600xf32>
    %t5 = bufferization.to_tensor %m5 restrict writable : memref<280xf32> to tensor<280xf32>
    %t6 = bufferization.to_tensor %m6 restrict writable : memref<19600xf32> to tensor<19600xf32>
    %t7 = bufferization.to_tensor %m7 restrict writable : memref<280xf32> to tensor<280xf32>
    %t8 = bufferization.to_tensor %m8 restrict writable : memref<19600xf32> to tensor<19600xf32>

    %r:2 = call @rock_attention(%t0, %t1, %t2, %t3, %t4, %t5, %t6, %t7, %t8)
      : (tensor<19600xf32>, tensor<19600xf32>, tensor<9800xf32>, tensor<9800xf32>, tensor<19600xf32>, tensor<280xf32>, tensor<19600xf32>, tensor<280xf32>, tensor<19600xf32>) -> (tensor<19600xf32>, tensor<280xf32>)

    %rb0 = bufferization.to_buffer %r#0 : tensor<19600xf32> to memref<19600xf32>
    memref.copy %rb0, %m8 : memref<19600xf32> to memref<19600xf32>
    %rb1 = bufferization.to_buffer %r#1 : tensor<280xf32> to memref<280xf32>
    memref.copy %rb1, %m7 : memref<280xf32> to memref<280xf32>

    gpu.memcpy %arg0, %m0 : memref<19600xf32>, memref<19600xf32>
    gpu.dealloc %m0 : memref<19600xf32>
    gpu.memcpy %arg1, %m1 : memref<19600xf32>, memref<19600xf32>
    gpu.dealloc %m1 : memref<19600xf32>
    gpu.memcpy %arg2, %m2 : memref<9800xf32>, memref<9800xf32>
    gpu.dealloc %m2 : memref<9800xf32>
    gpu.memcpy %arg3, %m3 : memref<9800xf32>, memref<9800xf32>
    gpu.dealloc %m3 : memref<9800xf32>
    gpu.memcpy %arg4, %m4 : memref<19600xf32>, memref<19600xf32>
    gpu.dealloc %m4 : memref<19600xf32>
    gpu.memcpy %arg5, %m5 : memref<280xf32>, memref<280xf32>
    gpu.dealloc %m5 : memref<280xf32>
    gpu.memcpy %arg6, %m6 : memref<19600xf32>, memref<19600xf32>
    gpu.dealloc %m6 : memref<19600xf32>
    gpu.memcpy %arg7, %m7 : memref<280xf32>, memref<280xf32>
    gpu.dealloc %m7 : memref<280xf32>
    gpu.memcpy %arg8, %m8 : memref<19600xf32>, memref<19600xf32>
    gpu.dealloc %m8 : memref<19600xf32>
    return
  }
}
