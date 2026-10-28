// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt --tosa-to-rock %s -o -| FileCheck %s

// f8E4M3FN block scales (NVFP4-style) use a block size of 16 on gfx1250.
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1250"} {
// CHECK-LABEL: @test_matmul_t_block_scaled_e4m3
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}} {quantBlockSize = 16 : i64} : tensor<1x128x256xf4E2M1FN> scaled by tensor<1x128x16xf8E4M3FN> * tensor<1x512x256xf4E2M1FN> scaled by tensor<1x512x16xf8E4M3FN>
func.func @test_matmul_t_block_scaled_e4m3(%a_data: tensor<1x128x256xf4E2M1FN>,
                                           %a_scale: tensor<1x128x16xf8E4M3FN>,
                                           %b_data: tensor<1x512x256xf4E2M1FN>,
                                           %b_scale: tensor<1x512x16xf8E4M3FN>)
                                           -> tensor<1x128x512xf32> attributes {rock.kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_16>}
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x16xf8E4M3FN>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x16xf8E4M3FN>)
      -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>
}
}
