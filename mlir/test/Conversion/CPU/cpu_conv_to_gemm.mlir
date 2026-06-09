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
// (4) the mul+add body.
//
// Negative tests (no `rock.cpu_fused_conv` produced) cover the cases the
// matcher in `--cpu-conv-to-gemm` deliberately leaves alone:
//   * non-unit stride or dilation in the user-facing direction (the matcher
//     in `ConvToGemm.cpp` requires `dims->strides == [1,1]` and
//     `dims->dilations == [1,1]`; see Tests 9, 12, 13);
//   * non-2-D spatial extents -- 3-D convolutions produce a 10-iter
//     `linalg.generic` (6 parallel + 4 reduction) which fails the
//     `outputImage.size() != 2 || filterLoop.size() != 2` check (Test 14).
//
// Note: *group* convolutions (G > 1) are positive cases. `inferConvolutionDims`
// classifies the group axis as a `depth` dim and the matcher only requires
// `dims->depth.size() == 1`, which a single explicit group axis satisfies; the
// per-group GEMM shape falls out of the existing affine maps. Tests 10 and 11
// exercise this for forward and backward-data direction.
//
// 1-D convolutions are positive cases too (Tests 15 and 16 cover forward;
// Test 19 covers backward-data with a non-unit user stride). rocmlir-gen has
// no native 1-D code path (`in_h` is a required argument), so we model 1-D
// as a 2-D conv with `in_h = fil_h = 1`. Structurally that is still a 2-D
// conv-shaped `linalg.generic` with one degenerate spatial axis, and the
// matcher accepts it unchanged.

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
// FWD-1X1-S2-NEG-SAME:    ins({{.*}} : tensor<64x13x13x1x1024xf32>, tensor<1x2048x1x1x1024xf32>)
// FWD-1X1-S2-NEG-SAME:    outs({{.*}} : tensor<64x7x7x1x2048xf32>)
// FWD-1X1-S2-NEG-NOT:     rock.cpu_fused_conv

// Test 10. Forward group conv (G=2), padded, NHWGC / GKYXC /
// NHWGK, f32. Even with multiple groups, `inferConvolutionDims` produces a
// single `depth` dim for the group axis, so this is still an 8-D iter-space
// match for the rewriter.
// RUN: rocmlir-gen -batchsize=4 -groupsize=2 -in_channels=16 -in_h=8 -in_w=8 -out_channels=16 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-GROUP-NHWGC

// FWD-GROUP-NHWGC-LABEL: func.func @conv_cpu(
// FWD-GROUP-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<4x8x8x2x8xf32>)
// FWD-GROUP-NHWGC:         linalg.generic
// FWD-GROUP-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-GROUP-NHWGC-SAME:    ins({{.*}} : tensor<4x10x10x2x8xf32>, tensor<2x8x3x3x8xf32>)
// FWD-GROUP-NHWGC-SAME:    outs({{.*}} : tensor<4x8x8x2x8xf32>)
// FWD-GROUP-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-GROUP-NHWGC:         arith.mulf
// FWD-GROUP-NHWGC:         arith.addf

// Test 11. Backward-data 3x3 stride-2 *group* conv (G=2), NHWGC / GKYXC /
// NHWGK, f32. The user-facing stride-2 backward-data direction maps to a
// stride-1 padded conv in the matcher's canonical direction, so this still
// matches even though the user stride is > 1.
// RUN: rocmlir-gen -batchsize=4 -groupsize=2 -in_channels=16 -in_h=16 -in_w=16 -out_channels=16 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-GROUP-S2-NHWGC

// BWDDATA-GROUP-S2-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-GROUP-S2-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<4x16x16x2x8xf32>)
// BWDDATA-GROUP-S2-NHWGC:         linalg.generic
// BWDDATA-GROUP-S2-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-GROUP-S2-NHWGC-SAME:    ins({{.*}} : tensor<4x18x18x2x8xf32>, tensor<2x8x3x3x8xf32>)
// BWDDATA-GROUP-S2-NHWGC-SAME:    outs({{.*}} : tensor<4x16x16x2x8xf32>)
// BWDDATA-GROUP-S2-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// BWDDATA-GROUP-S2-NHWGC:         arith.mulf
// BWDDATA-GROUP-S2-NHWGC:         arith.addf

// Test 12. Negative case: Forward 3x3 *stride-2* conv. Like Test 9,
// this is rejected by the matcher.
// RUN: rocmlir-gen -batchsize=4 -in_channels=8 -in_h=14 -in_w=14 -out_channels=8 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3X3-S2-NEG

// FWD-3X3-S2-NEG-LABEL: func.func @conv_cpu(
// FWD-3X3-S2-NEG:         iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-3X3-S2-NEG-SAME:    ins({{.*}} : tensor<4x13x13x1x8xf32>, tensor<1x8x3x3x8xf32>)
// FWD-3X3-S2-NEG-SAME:    outs({{.*}} : tensor<4x6x6x1x8xf32>)
// FWD-3X3-S2-NEG-NOT:     rock.cpu_fused_conv

// Test 13. Negative case: Forward 3x3 *dilation-2* conv, rejected by the matcher.
// RUN: rocmlir-gen -batchsize=4 -in_channels=8 -in_h=14 -in_w=14 -out_channels=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3X3-D2-NEG

// FWD-3X3-D2-NEG-LABEL: func.func @conv_cpu(
// FWD-3X3-D2-NEG:         iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-3X3-D2-NEG-SAME:    ins({{.*}} : tensor<4x14x14x1x8xf32>, tensor<1x8x3x3x8xf32>)
// FWD-3X3-D2-NEG-SAME:    outs({{.*}} : tensor<4x10x10x1x8xf32>)
// FWD-3X3-D2-NEG-NOT:     rock.cpu_fused_conv

// Test 14. Negative case: Forward *3-D* 3x3x3 padded conv, NDHWGC / GKZYXC /
// NDHWGK, f32. rocmlir-gen emits a 10-iter `linalg.generic` (6 parallel + 4
// reduction) for 3-D convs, which fails the matcher's
// `outputImage.size() != 2` / `filterLoop.size() != 2` check, so the op is
// left untouched. The matcher walks affine maps rather than tensor positions,
// so this rejection is layout-agnostic -- one 3-D negative is enough.
// RUN: rocmlir-gen -batchsize=2 -in_channels=4 -in_d=8 -in_h=8 -in_w=8 -out_channels=4 -fil_d=3 -fil_h=3 -fil_w=3 --dilation_d=1 --dilation_h=1 --dilation_w=1 --conv_stride_d=1 --conv_stride_h=1 --conv_stride_w=1 --padding_d=1 --padding_h=1 --padding_w=1 --operation conv -fil_layout=gk012c -in_layout=n012gc -out_layout=n012gk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3D-NDHWGC-NEG

// FWD-3D-NDHWGC-NEG-LABEL: func.func @conv_cpu(
// FWD-3D-NDHWGC-NEG:         iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// FWD-3D-NDHWGC-NEG-SAME:    ins({{.*}} : tensor<2x10x10x10x1x4xf32>, tensor<1x4x3x3x3x4xf32>)
// FWD-3D-NDHWGC-NEG-SAME:    outs({{.*}} : tensor<2x8x8x8x1x4xf32>)
// FWD-3D-NDHWGC-NEG-NOT:     rock.cpu_fused_conv

// Test 15. Forward 1-D conv (1x3 filter), padded.
// rocmlir-gen has no native 1-D path (`in_h` is a required argument), so we
// model 1-D as a 2-D conv with `in_h = fil_h = 1`. The matcher still sees a
// 2-D conv-shaped op, so it rewrites it.
// RUN: rocmlir-gen -batchsize=4 -in_channels=8 -in_h=1 -in_w=16 -out_channels=8 -fil_h=1 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=1 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1D-PAD-NHWGC

// FWD-1D-PAD-NHWGC-LABEL: func.func @conv_cpu(
// FWD-1D-PAD-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<4x1x16x1x8xf32>)
// FWD-1D-PAD-NHWGC:         linalg.generic
// FWD-1D-PAD-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1D-PAD-NHWGC-SAME:    ins({{.*}} : tensor<4x1x18x1x8xf32>, tensor<1x8x1x3x8xf32>)
// FWD-1D-PAD-NHWGC-SAME:    outs({{.*}} : tensor<4x1x16x1x8xf32>)
// FWD-1D-PAD-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-1D-PAD-NHWGC:         arith.mulf
// FWD-1D-PAD-NHWGC:         arith.addf

// Test 16. Forward 1-D conv (modeled as `in_h = fil_h = 1` 2-D conv) with
// *no padding* on the W axis, so the output W is `in_w - fil_w + 1 = 14`,
// smaller than the input.
// RUN: rocmlir-gen -batchsize=4 -in_channels=8 -in_h=1 -in_w=16 -out_channels=8 -fil_h=1 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1D-NOPAD-NHWGC

// FWD-1D-NOPAD-NHWGC-LABEL: func.func @conv_cpu(
// FWD-1D-NOPAD-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<4x1x14x1x8xf32>)
// FWD-1D-NOPAD-NHWGC:         linalg.generic
// FWD-1D-NOPAD-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1D-NOPAD-NHWGC-SAME:    ins({{.*}} : tensor<4x1x16x1x8xf32>, tensor<1x8x1x3x8xf32>)
// FWD-1D-NOPAD-NHWGC-SAME:    outs({{.*}} : tensor<4x1x14x1x8xf32>)
// FWD-1D-NOPAD-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-1D-NOPAD-NHWGC:         arith.mulf
// FWD-1D-NOPAD-NHWGC:         arith.addf

// Test 17. Forward 3x3 conv with *no padding*
// RUN: rocmlir-gen -batchsize=4 -in_channels=8 -in_h=14 -in_w=14 -out_channels=8 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3X3-NOPAD-NHWGC

// FWD-3X3-NOPAD-NHWGC-LABEL: func.func @conv_cpu(
// FWD-3X3-NOPAD-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<4x12x12x1x8xf32>)
// FWD-3X3-NOPAD-NHWGC:         linalg.generic
// FWD-3X3-NOPAD-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-3X3-NOPAD-NHWGC-SAME:    ins({{.*}} : tensor<4x14x14x1x8xf32>, tensor<1x8x3x3x8xf32>)
// FWD-3X3-NOPAD-NHWGC-SAME:    outs({{.*}} : tensor<4x12x12x1x8xf32>)
// FWD-3X3-NOPAD-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-3X3-NOPAD-NHWGC:         arith.mulf
// FWD-3X3-NOPAD-NHWGC:         arith.addf

// Test 18. Forward 2-D 5x5 conv with padding=2.
// RUN: rocmlir-gen -batchsize=8 -in_channels=16 -in_h=12 -in_w=12 -out_channels=32 -fil_h=5 -fil_w=5 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=2 --padding_w=2 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-5X5-PAD-NHWGC

// FWD-5X5-PAD-NHWGC-LABEL: func.func @conv_cpu(
// FWD-5X5-PAD-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<8x12x12x1x32xf32>)
// FWD-5X5-PAD-NHWGC:         linalg.generic
// FWD-5X5-PAD-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-5X5-PAD-NHWGC-SAME:    ins({{.*}} : tensor<8x16x16x1x16xf32>, tensor<1x32x5x5x16xf32>)
// FWD-5X5-PAD-NHWGC-SAME:    outs({{.*}} : tensor<8x12x12x1x32xf32>)
// FWD-5X5-PAD-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// FWD-5X5-PAD-NHWGC:         arith.mulf
// FWD-5X5-PAD-NHWGC:         arith.addf

// Test 19. Backward-data 1D conv with user stride 2 on W, modeled as a
// 2D conv with `in_h = fil_h = 1`.
// RUN: rocmlir-gen -batchsize=4 -in_channels=8 -in_h=1 -in_w=16 -out_channels=8 -fil_h=1 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch %arch -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-1D-S2-NHWGC

// BWDDATA-1D-S2-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-1D-S2-NHWGC:         linalg.fill ins({{.*}} : f32) outs({{.*}} : tensor<4x1x16x1x8xf32>)
// BWDDATA-1D-S2-NHWGC:         linalg.generic
// BWDDATA-1D-S2-NHWGC-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-1D-S2-NHWGC-SAME:    ins({{.*}} : tensor<4x1x18x1x8xf32>, tensor<1x8x1x3x8xf32>)
// BWDDATA-1D-S2-NHWGC-SAME:    outs({{.*}} : tensor<4x1x16x1x8xf32>)
// BWDDATA-1D-S2-NHWGC-SAME:    attrs = {rock.cpu_fused_conv}
// BWDDATA-1D-S2-NHWGC:         arith.mulf
// BWDDATA-1D-S2-NHWGC:         arith.addf
