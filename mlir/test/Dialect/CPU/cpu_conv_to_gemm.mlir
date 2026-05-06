// Tests for `--cpu-conv-to-gemm`. The pass rewrites a 2-D convolution-shaped
// `linalg.generic` inside a `cpu_verifier` function into a single fused
// `linalg.generic` (5 parallel + 3 reduction iterators, tagged with
// `rock.cpu_fused_conv`) whose indexing maps inline the im2col gather. No
// `tensor.collapse_shape` or 3-D matmul shape is produced at this stage --
// that lowering happens later in `FusedConvToMatmulSchedule`.
//
// Each positive test below checks: (1) the fresh zero-init via `linalg.fill`,
// (2) the 8-D iter-type signature of the fused op, (3) the operand shapes
// going into the contraction (input is the padded/dilated tensor in the
// user's original layout, filter and output keep their user shapes), and
// (4) the mul+add body. Test 9 is a negative case: stride-2 forward conv
// is rejected by the matcher, so no `rock.cpu_fused_conv` op is produced.

// Test 1. 3x3 stride-2 backward-data conv, NHWGC / GKYXC / NHWGK, bf16.
// RUN: rocmlir-gen -batchsize=64 -in_channels=512 -in_h=16 -in_w=16 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t bf16 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-3X3-NHWGC

// BWDDATA-3X3-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-3X3-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<64x16x16x1x512xf32>)
// BWDDATA-3X3-NHWGC:         linalg.generic
// BWDDATA-3X3-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-3X3-NHWGC-SAME:    ins({{.*}} : tensor<64x18x18x1x512xf32>, tensor<1x512x3x3x512xf32>)
// BWDDATA-3X3-NHWGC-SAME:    outs({{.*}} : tensor<64x16x16x1x512xf32>)
// BWDDATA-3X3-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// BWDDATA-3X3-NHWGC:         arith.mulf
// BWDDATA-3X3-NHWGC:         arith.addf

// Test 2. 7x7 stride-2 backward-data conv, NHWGC / GKYXC / NHWGK, f32.
// RUN: rocmlir-gen -batchsize=256 -in_channels=3 -in_h=230 -in_w=230 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=1 --padding_w=1 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-7X7-NHWGC

// BWDDATA-7X7-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-7X7-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<256x230x230x1x3xf32>)
// BWDDATA-7X7-NHWGC:         linalg.generic
// BWDDATA-7X7-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-7X7-NHWGC-SAME:    ins({{.*}} : tensor<256x236x236x1x64xf32>, tensor<1x64x7x7x3xf32>)
// BWDDATA-7X7-NHWGC-SAME:    outs({{.*}} : tensor<256x230x230x1x3xf32>)
// BWDDATA-7X7-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// BWDDATA-7X7-NHWGC:         arith.mulf
// BWDDATA-7X7-NHWGC:         arith.addf

// Test 3. 7x7 stride-2 backward-data conv, NGCHW / GKCYX / NGKHW, f32.
// RUN: rocmlir-gen -batchsize=256 -in_channels=3 -in_h=230 -in_w=230 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=1 --padding_w=1 --operation conv_bwd_data -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-7X7-NGCHW

// BWDDATA-7X7-NGCHW-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-7X7-NGCHW:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<256x1x3x230x230xf32>)
// BWDDATA-7X7-NGCHW:         linalg.generic
// BWDDATA-7X7-NGCHW-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-7X7-NGCHW-SAME:    ins({{.*}} : tensor<256x1x64x236x236xf32>, tensor<1x64x3x7x7xf32>)
// BWDDATA-7X7-NGCHW-SAME:    outs({{.*}} : tensor<256x1x3x230x230xf32>)
// BWDDATA-7X7-NGCHW-SAME:    attrs = {rock.cpu_fused_conv}
// BWDDATA-7X7-NGCHW:         arith.mulf
// BWDDATA-7X7-NGCHW:         arith.addf

// Test 4. Forward 1x1 conv, NHWGC / GKYXC / NHWGK, f32 (ResNet-50 bottleneck shape).
// RUN: rocmlir-gen -batchsize=64 -in_channels=2048 -in_h=7 -in_w=7 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1X1-NHWGC

// FWD-1X1-NHWGC-LABEL: func.func @conv_cpu(
// FWD-1X1-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<64x7x7x1x512xf32>)
// FWD-1X1-NHWGC:         linalg.generic
// FWD-1X1-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1X1-NHWGC-SAME:    ins({{.*}} : tensor<64x7x7x1x2048xf32>, tensor<1x512x1x1x2048xf32>)
// FWD-1X1-NHWGC-SAME:    outs({{.*}} : tensor<64x7x7x1x512xf32>)
// FWD-1X1-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-1X1-NHWGC:         arith.mulf
// FWD-1X1-NHWGC:         arith.addf

// Test 5. Forward 3x3 stride-1 padded conv, NHWGC / GKYXC / NHWGK, f32.
// RUN: rocmlir-gen -batchsize=64 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3X3-NHWGC

// FWD-3X3-NHWGC-LABEL: func.func @conv_cpu(
// FWD-3X3-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<64x28x28x1x128xf32>)
// FWD-3X3-NHWGC:         linalg.generic
// FWD-3X3-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-3X3-NHWGC-SAME:    ins({{.*}} : tensor<64x30x30x1x128xf32>, tensor<1x128x3x3x128xf32>)
// FWD-3X3-NHWGC-SAME:    outs({{.*}} : tensor<64x28x28x1x128xf32>)
// FWD-3X3-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-3X3-NHWGC:         arith.mulf
// FWD-3X3-NHWGC:         arith.addf

// Test 6. Forward 3x3 stride-1 padded conv, NGCHW / GKCYX / NGKHW, f32.
// RUN: rocmlir-gen -batchsize=64 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --operation conv -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3X3-NGCHW

// FWD-3X3-NGCHW-LABEL: func.func @conv_cpu(
// FWD-3X3-NGCHW:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<64x1x128x28x28xf32>)
// FWD-3X3-NGCHW:         linalg.generic
// FWD-3X3-NGCHW-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-3X3-NGCHW-SAME:    ins({{.*}} : tensor<64x1x128x30x30xf32>, tensor<1x128x128x3x3xf32>)
// FWD-3X3-NGCHW-SAME:    outs({{.*}} : tensor<64x1x128x28x28xf32>)
// FWD-3X3-NGCHW-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-3X3-NGCHW:         arith.mulf
// FWD-3X3-NGCHW:         arith.addf

// Test 7. Backward-data 3x3 stride-2 conv, NHWGC / GKYXC / NHWGK, f32.
// RUN: rocmlir-gen -batchsize=64 -in_channels=128 -in_h=58 -in_w=58 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-3X3-S2-NHWGC

// BWDDATA-3X3-S2-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-3X3-S2-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<64x58x58x1x128xf32>)
// BWDDATA-3X3-S2-NHWGC:         linalg.generic
// BWDDATA-3X3-S2-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-3X3-S2-NHWGC-SAME:    ins({{.*}} : tensor<64x60x60x1x128xf32>, tensor<1x128x3x3x128xf32>)
// BWDDATA-3X3-S2-NHWGC-SAME:    outs({{.*}} : tensor<64x58x58x1x128xf32>)
// BWDDATA-3X3-S2-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// BWDDATA-3X3-S2-NHWGC:         arith.mulf
// BWDDATA-3X3-S2-NHWGC:         arith.addf

// Test 8. Forward 1x1 conv, NHWGC / GKYXC / NHWGK, *bf16*. Inputs are extf'd
// to f32 before the fused contraction; result is truncf'd back to bf16.
// RUN: rocmlir-gen -batchsize=64 -in_channels=2048 -in_h=7 -in_w=7 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t bf16 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1X1-NHWGC-BF16

// FWD-1X1-NHWGC-BF16-LABEL: func.func @conv_cpu(
// FWD-1X1-NHWGC-BF16:         arith.extf {{.*}} : bf16 to f32
// FWD-1X1-NHWGC-BF16:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<64x7x7x1x512xf32>)
// FWD-1X1-NHWGC-BF16:         linalg.generic
// FWD-1X1-NHWGC-BF16-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1X1-NHWGC-BF16-SAME:    ins({{.*}} : tensor<64x7x7x1x2048xf32>, tensor<1x512x1x1x2048xf32>)
// FWD-1X1-NHWGC-BF16-SAME:    outs({{.*}} : tensor<64x7x7x1x512xf32>)
// FWD-1X1-NHWGC-BF16-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-1X1-NHWGC-BF16:         arith.mulf
// FWD-1X1-NHWGC-BF16:         arith.addf
// FWD-1X1-NHWGC-BF16:         arith.truncf {{.*}} : f32 to bf16

// Test 9. Forward 1x1 *stride-2* conv, NHWGC / GKYXC / NHWGK, f32. The matcher
// in `--cpu-conv-to-gemm` requires unit stride / dilation, so this op is left
// alone (no `rock.cpu_fused_conv` attribute on any op in the function).
// RUN: rocmlir-gen -batchsize=64 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=2048 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1X1-S2-NEG

// FWD-1X1-S2-NEG-LABEL: func.func @conv_cpu(
// FWD-1X1-S2-NEG:         iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1X1-S2-NEG-SAME:    ins({{.*}} : tensor<64x14x14x1x1024xf32>, tensor<1x2048x1x1x1024xf32>)
// FWD-1X1-S2-NEG-SAME:    outs({{.*}} : tensor<64x7x7x1x2048xf32>)
// FWD-1X1-S2-NEG-NOT:     rock.cpu_fused_conv
