// RUN: rocmlir-opt --tosa-to-rock %s -o -| FileCheck %s

module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
// CHECK-LABEL: @test_matmul_t_block_scaled_basic
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}}
func.func @test_matmul_t_block_scaled_basic(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                             %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                             %b_data: tensor<1x512x256xf4E2M1FN>, 
                                             %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                             -> tensor<1x128x512xf32> attributes {rock.kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>
}

// CHECK-LABEL: @test_matmul_t_block_scaled_batched
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}}
func.func @test_matmul_t_block_scaled_batched(%a_data: tensor<4x128x256xf4E2M1FN>, 
                                               %a_scale: tensor<4x128x8xf8E8M0FNU>,
                                               %b_data: tensor<4x512x256xf4E2M1FN>, 
                                               %b_scale: tensor<4x512x8xf8E8M0FNU>) 
                                               -> tensor<4x128x512xf32> attributes {rock.kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<4x128x256xf4E2M1FN>, tensor<4x128x8xf8E8M0FNU>, tensor<4x512x256xf4E2M1FN>, tensor<4x512x8xf8E8M0FNU>) 
      -> tensor<4x128x512xf32>
  return %result : tensor<4x128x512xf32>
}

// CHECK-LABEL: @test_matmul_t_block_scaled_large_k
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}}
func.func @test_matmul_t_block_scaled_large_k(%a_data: tensor<1x64x512xf4E2M1FN>, 
                                               %a_scale: tensor<1x64x16xf8E8M0FNU>,
                                               %b_data: tensor<1x256x512xf4E2M1FN>, 
                                               %b_scale: tensor<1x256x16xf8E8M0FNU>) 
                                               -> tensor<1x64x256xf32> attributes {rock.kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x64x512xf4E2M1FN>, tensor<1x64x16xf8E8M0FNU>, tensor<1x256x512xf4E2M1FN>, tensor<1x256x16xf8E8M0FNU>) 
      -> tensor<1x64x256xf32>
  return %result : tensor<1x64x256xf32>
}

// Test transpose on A data input
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a
// CHECK: rock.gemm tr %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}}
func.func @test_matmul_t_block_scaled_transpose_a(%a_data: tensor<1x256x256xf4E2M1FN>, 
                                                   %a_scale: tensor<1x256x8xf8E8M0FNU>,
                                                   %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                   %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                   -> tensor<1x256x512xf32> attributes {rock.kernel} {
  %a_tr = "tosa.transpose"(%a_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_tr, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x256x512xf32>
  return %result : tensor<1x256x512xf32>
}

// Test transpose on B data input (toggles B's default transpose)
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}}
func.func @test_matmul_t_block_scaled_transpose_b(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                   %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                   %b_data: tensor<1x256x256xf4E2M1FN>, 
                                                   %b_scale: tensor<1x256x8xf8E8M0FNU>) 
                                                   -> tensor<1x128x256xf32> attributes {rock.kernel} {
  %b_tr = "tosa.transpose"(%b_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_tr, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>) 
      -> tensor<1x128x256xf32>
  return %result : tensor<1x128x256xf32>
}

// Test transpose on A scale only
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a_scale
// CHECK: rock.gemm %{{.*}} scaled by tr %{{.*}} * tr %{{.*}} scaled by %{{.*}}
func.func @test_matmul_t_block_scaled_transpose_a_scale(%a_data: tensor<1x8x256xf4E2M1FN>, 
                                                         %a_scale: tensor<1x8x8xf8E8M0FNU>,
                                                         %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                         %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                         -> tensor<1x8x512xf32> attributes {rock.kernel} {
  %a_scale_tr = "tosa.transpose"(%a_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale_tr, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x8x512xf32>
  return %result : tensor<1x8x512xf32>
}

// Test transpose on B scale only
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b_scale
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}
func.func @test_matmul_t_block_scaled_transpose_b_scale(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                         %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                         %b_data: tensor<1x8x256xf4E2M1FN>, 
                                                         %b_scale: tensor<1x8x8xf8E8M0FNU>) 
                                                         -> tensor<1x128x8xf32> attributes {rock.kernel} {
  %b_scale_tr = "tosa.transpose"(%b_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale_tr {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>) 
      -> tensor<1x128x8xf32>
  return %result : tensor<1x128x8xf32>
}

// Test transpose on both A scale and B scale
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_both_scales
// CHECK: rock.gemm %{{.*}} scaled by tr %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}
func.func @test_matmul_t_block_scaled_transpose_both_scales(%a_data: tensor<1x8x256xf4E2M1FN>, 
                                                             %a_scale: tensor<1x8x8xf8E8M0FNU>,
                                                             %b_data: tensor<1x8x256xf4E2M1FN>, 
                                                             %b_scale: tensor<1x8x8xf8E8M0FNU>) 
                                                             -> tensor<1x8x8xf32> attributes {rock.kernel} {
  %a_scale_tr = "tosa.transpose"(%a_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %b_scale_tr = "tosa.transpose"(%b_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale_tr, %b_data, %b_scale_tr {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>, tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>) 
      -> tensor<1x8x8xf32>
  return %result : tensor<1x8x8xf32>
}

// Test transpose on A data AND A scale together
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a_data_and_scale
// CHECK: rock.gemm tr %{{.*}} scaled by tr %{{.*}} * tr %{{.*}} scaled by %{{.*}}
func.func @test_matmul_t_block_scaled_transpose_a_data_and_scale(%a_data: tensor<1x256x256xf4E2M1FN>, 
                                                                  %a_scale: tensor<1x8x256xf8E8M0FNU>,
                                                                  %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                                  %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                                  -> tensor<1x256x512xf32> attributes {rock.kernel} {
  %a_tr = "tosa.transpose"(%a_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %a_scale_tr = "tosa.transpose"(%a_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x256xf8E8M0FNU>) -> tensor<1x256x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_tr, %a_scale_tr, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x256x512xf32>
  return %result : tensor<1x256x512xf32>
}

// Test transpose on B data AND B scale together
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b_data_and_scale
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by tr %{{.*}}
func.func @test_matmul_t_block_scaled_transpose_b_data_and_scale(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                                  %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                                  %b_data: tensor<1x256x256xf4E2M1FN>, 
                                                                  %b_scale: tensor<1x8x256xf8E8M0FNU>) 
                                                                  -> tensor<1x128x256xf32> attributes {rock.kernel} {
  %b_tr = "tosa.transpose"(%b_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %b_scale_tr = "tosa.transpose"(%b_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x256xf8E8M0FNU>) -> tensor<1x256x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_tr, %b_scale_tr {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>) 
      -> tensor<1x128x256xf32>
  return %result : tensor<1x128x256xf32>
}

// Test output transpose (transpose_c)
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_c
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}} {oTransposed
func.func @test_matmul_t_block_scaled_transpose_c(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                    %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                    %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                    %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                    -> tensor<1x512x128xf32> attributes {rock.kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x128x512xf32>
  %result_tr = "tosa.transpose"(%result) {perms = array<i32: 0, 2, 1>} : (tensor<1x128x512xf32>) -> tensor<1x512x128xf32>
  return %result_tr : tensor<1x512x128xf32>
}

// Test output transpose combined with input transpose on A
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a_and_c
// CHECK: rock.gemm tr %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}} {oTransposed
func.func @test_matmul_t_block_scaled_transpose_a_and_c(%a_data: tensor<1x256x256xf4E2M1FN>, 
                                                          %a_scale: tensor<1x256x8xf8E8M0FNU>,
                                                          %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                          %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                          -> tensor<1x512x256xf32> attributes {rock.kernel} {
  %a_tr = "tosa.transpose"(%a_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_tr, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x256x512xf32>
  %result_tr = "tosa.transpose"(%result) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x512xf32>) -> tensor<1x512x256xf32>
  return %result_tr : tensor<1x512x256xf32>
}

// Test output transpose combined with B data transpose
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b_and_c
// CHECK: rock.gemm %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}} {oTransposed
func.func @test_matmul_t_block_scaled_transpose_b_and_c(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                          %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                          %b_data: tensor<1x256x256xf4E2M1FN>, 
                                                          %b_scale: tensor<1x256x8xf8E8M0FNU>) 
                                                          -> tensor<1x256x128xf32> attributes {rock.kernel} {
  %b_tr = "tosa.transpose"(%b_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_tr, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>) 
      -> tensor<1x128x256xf32>
  %result_tr = "tosa.transpose"(%result) {perms = array<i32: 0, 2, 1>} : (tensor<1x128x256xf32>) -> tensor<1x256x128xf32>
  return %result_tr : tensor<1x256x128xf32>
}
}
