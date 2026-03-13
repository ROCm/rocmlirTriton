// RUN: rocmlir-gen --conv-config "--x2 1 --operation conv_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv_v4r4_wrw_xdlops" -pv --mlir-print-local-scope | FileCheck %s --check-prefix=PV

//PV: [[FIL1:%.*]] = memref.alloc() : memref<2097152xf32>
//PV: [[FIL2:%.*]] = memref.alloc() : memref<2097152xf32>
//PV: call @mlir_gen_igemm_conv_v4r4_wrw_xdlops_0_verify0([[FIL1]], [[FIL2]])
//PV-LABEL: func @conv_bwd_weight_cpu
//PV-SAME: ([[ARG0:%.+]]: tensor<2097152xf32>, [[ARG1:%.+]]: tensor<51380224xf32>, [[ARG2:%.+]]: tensor<33554432xf32>)
//PV: tensor.expand_shape [[ARG0]] {{.*}} output_shape [1, 2048, 1024, 1, 1] : tensor<2097152xf32> into tensor<1x2048x1024x1x1xf32>
//PV-NEXT: %[[EXP0:.*]] = tensor.expand_shape [[ARG1]] {{.*}} output_shape [256, 1, 1024, 14, 14] : tensor<51380224xf32> into tensor<256x1x1024x14x14xf32>
//PV-NEXT: %[[EXP1:.*]] = tensor.expand_shape [[ARG2]] {{.*}} output_shape [256, 1, 2048, 8, 8] : tensor<33554432xf32> into tensor<256x1x2048x8x8xf32>
//PV: %[[f32_0:.*]] = arith.constant 0.000000e+00 : f32
//PV-NEXT: %[[PAD:.*]] = tensor.pad %[[EXP0]] {{.*}} {
//PV-NEXT: ^bb0({{.*}}):
//PV-NEXT: tensor.yield %[[f32_0]] : f32
//PV-NEXT: } : tensor<256x1x1024x14x14xf32> to tensor<256x1x1024x16x16xf32>
//PV: %[[f32_1:.*]] = arith.constant 0.000000e+00 : f32
//PV-NEXT: %[[EMPTY:.*]] = tensor.empty() : tensor<1x2048x1024x1x1xf32>
//PV-NEXT: %[[FILL:.*]] = linalg.fill ins(%[[f32_1]] : f32) outs(%[[EMPTY]] : tensor<1x2048x1024x1x1xf32>) -> tensor<1x2048x1024x1x1xf32>
//PV-NEXT: {{.*}} = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d2, d6 * 2 + d3 * 2, d7 * 2 + d4 * 2)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d1, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[PAD]], %[[EXP1]] : tensor<256x1x1024x16x16xf32>, tensor<256x1x2048x8x8xf32>) outs(%[[FILL]] : tensor<1x2048x1024x1x1xf32>) {
//PV-NEXT: ^bb0(%[[IN:.*]]: f32, %[[IN3:.*]]: f32, %[[OUT:.*]]: f32):
//PV-NEXT: %[[PRD:.*]] = arith.mulf %[[IN]], %[[IN3]] : f32
//PV-NEXT: %[[ACC:.*]] = arith.addf %[[OUT]], %[[PRD]] : f32
//PV-NEXT: linalg.yield %[[ACC]] : f32
//PV-NEXT: } -> tensor<1x2048x1024x1x1xf32>

// RUN:  rocmlir-gen --conv-config "--x2 1 --operation conv_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv_v4r4_wrw_xdlops" -ph -pr | FileCheck %s --check-prefix=PH

//PH: [[FIL:%.*]] = memref.cast %{{.*}} : memref<2097152xf32> to memref<*xf32>
//PH: call @printMemrefF32([[FIL]]) : (memref<*xf32>) -> ()
