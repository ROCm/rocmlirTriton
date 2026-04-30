// Test 1. 3x3 stride-2 backward-data conv, NHWGC / GKYXC / NHWGK, bf16.
// RUN: rocmlir-gen -batchsize=64 -in_channels=512 -in_h=16 -in_w=16 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t bf16 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-3X3-NHWGC

// BWDDATA-3X3-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-3X3-NHWGC-NOT: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-3X3-NHWGC:      tensor.collapse_shape
// BWDDATA-3X3-NHWGC-SAME: into tensor<1x16384x4608xf32>
// BWDDATA-3X3-NHWGC:      tensor.collapse_shape
// BWDDATA-3X3-NHWGC-SAME: into tensor<1x4608x512xf32>
// BWDDATA-3X3-NHWGC:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// BWDDATA-3X3-NHWGC:      ins({{.*}} : tensor<1x16384x4608xf32>, tensor<1x4608x512xf32>)
// BWDDATA-3X3-NHWGC:      outs({{.*}} : tensor<1x16384x512xf32>)
// BWDDATA-3X3-NHWGC:      arith.mulf
// BWDDATA-3X3-NHWGC:      arith.addf

// Test 2. 7x7 stride-2 backward-data conv, NHWGC / GKYXC / NHWGK, f32.
// RUN: rocmlir-gen -batchsize=256 -in_channels=3 -in_h=230 -in_w=230 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=1 --padding_w=1 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-7X7-NHWGC

// BWDDATA-7X7-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-7X7-NHWGC-NOT:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-7X7-NHWGC:         tensor.collapse_shape
// BWDDATA-7X7-NHWGC:         into tensor<1x13542400x3136xf32>
// BWDDATA-7X7-NHWGC:         tensor.collapse_shape
// BWDDATA-7X7-NHWGC:         into tensor<1x3136x3xf32>
// BWDDATA-7X7-NHWGC:         iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// BWDDATA-7X7-NHWGC:         ins({{.*}} : tensor<1x13542400x3136xf32>, tensor<1x3136x3xf32>)
// BWDDATA-7X7-NHWGC:         outs({{.*}} : tensor<1x13542400x3xf32>)
// BWDDATA-7X7-NHWGC:         arith.mulf
// BWDDATA-7X7-NHWGC:         arith.addf

// Test 3. 7x7 stride-2 backward-data conv, NGCHW / GKCYX / NGCHW, f32.
// RUN: rocmlir-gen -batchsize=256 -in_channels=3 -in_h=230 -in_w=230 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=1 --padding_w=1 --operation conv_bwd_data -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -t f32 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-7X7-NGCHW

// BWDDATA-7X7-NGCHW-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-7X7-NGCHW-NOT: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-7X7-NGCHW:         tensor.collapse_shape
// BWDDATA-7X7-NGCHW:         into tensor<1x13542400x3136xf32>
// BWDDATA-7X7-NGCHW:         tensor.collapse_shape
// BWDDATA-7X7-NGCHW:         into tensor<1x3136x3xf32>
// BWDDATA-7X7-NGCHW:         iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// BWDDATA-7X7-NGCHW:         ins({{.*}} : tensor<1x13542400x3136xf32>, tensor<1x3136x3xf32>)
// BWDDATA-7X7-NGCHW:         outs({{.*}} : tensor<1x13542400x3xf32>)
// BWDDATA-7X7-NGCHW:         arith.mulf
// BWDDATA-7X7-NGCHW:         arith.addf

// Test 4. Forward 1x1 conv, NHWGC / GKYXC / NHWGK, f32.
// Forward 1x1 conv, NHWGC / GKYXC / NHWGK, f32 (ResNet-50 bottleneck shape:
// RUN: rocmlir-gen -batchsize=64 -in_channels=2048 -in_h=7 -in_w=7 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1X1-NHWGC

// FWD-1X1-NHWGC-LABEL: func.func @conv_cpu(
// FWD-1X1-NHWGC-NOT:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1X1-NHWGC:         tensor.collapse_shape
// FWD-1X1-NHWGC-SAME:    into tensor<1x3136x2048xf32>
// FWD-1X1-NHWGC:         tensor.collapse_shape
// FWD-1X1-NHWGC-SAME:    into tensor<1x2048x512xf32>
// FWD-1X1-NHWGC:         iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// FWD-1X1-NHWGC-SAME:    ins({{.*}} : tensor<1x3136x2048xf32>, tensor<1x2048x512xf32>)
// FWD-1X1-NHWGC-SAME:    outs({{.*}} : tensor<1x3136x512xf32>)
// FWD-1X1-NHWGC:         arith.mulf
// FWD-1X1-NHWGC:         arith.addf

// Test 5. Forward 3x3 stride-1 padded conv, NHWGC / GKYXC / NHWGK, f32
// RUN: rocmlir-gen -batchsize=64 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3X3-NHWGC

// FWD-3X3-NHWGC-LABEL: func.func @conv_cpu(
// FWD-3X3-NHWGC-NOT:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-3X3-NHWGC:         tensor.collapse_shape
// FWD-3X3-NHWGC-SAME:    into tensor<1x50176x1152xf32>
// FWD-3X3-NHWGC:         tensor.collapse_shape
// FWD-3X3-NHWGC-SAME:    into tensor<1x1152x128xf32>
// FWD-3X3-NHWGC:         iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// FWD-3X3-NHWGC-SAME:    ins({{.*}} : tensor<1x50176x1152xf32>, tensor<1x1152x128xf32>)
// FWD-3X3-NHWGC-SAME:    outs({{.*}} : tensor<1x50176x128xf32>)
// FWD-3X3-NHWGC:         arith.mulf
// FWD-3X3-NHWGC:         arith.addf

// Test 6. Forward 3x3 stride-1 padded conv, NGCHW / GKCYX / NGKHW, f32. Same
// RUN: rocmlir-gen -batchsize=64 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --operation conv -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -t f32 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-3X3-NGCHW

// FWD-3X3-NGCHW-LABEL: func.func @conv_cpu(
// FWD-3X3-NGCHW-NOT:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-3X3-NGCHW:         tensor.collapse_shape
// FWD-3X3-NGCHW-SAME:    into tensor<1x50176x1152xf32>
// FWD-3X3-NGCHW:         tensor.collapse_shape
// FWD-3X3-NGCHW-SAME:    into tensor<1x1152x128xf32>
// FWD-3X3-NGCHW:         iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// FWD-3X3-NGCHW-SAME:    ins({{.*}} : tensor<1x50176x1152xf32>, tensor<1x1152x128xf32>)
// FWD-3X3-NGCHW-SAME:    outs({{.*}} : tensor<1x50176x128xf32>)
// FWD-3X3-NGCHW:         arith.mulf
// FWD-3X3-NGCHW:         arith.addf

// Test 7. Backward-data 3x3 stride-2 conv, NHWGC / GKYXC / NHWGK, f32
// RUN: rocmlir-gen -batchsize=64 -in_channels=128 -in_h=58 -in_w=58 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv_bwd_data -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_bwd_data_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=BWDDATA-3X3-S2-NHWGC

// BWDDATA-3X3-S2-NHWGC-LABEL: func.func @conv_bwd_data_cpu(
// BWDDATA-3X3-S2-NHWGC-NOT:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// BWDDATA-3X3-S2-NHWGC:         tensor.collapse_shape
// BWDDATA-3X3-S2-NHWGC-SAME:    into tensor<1x215296x1152xf32>
// BWDDATA-3X3-S2-NHWGC:         tensor.collapse_shape
// BWDDATA-3X3-S2-NHWGC-SAME:    into tensor<1x1152x128xf32>
// BWDDATA-3X3-S2-NHWGC:         iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// BWDDATA-3X3-S2-NHWGC-SAME:    ins({{.*}} : tensor<1x215296x1152xf32>, tensor<1x1152x128xf32>)
// BWDDATA-3X3-S2-NHWGC-SAME:    outs({{.*}} : tensor<1x215296x128xf32>)
// BWDDATA-3X3-S2-NHWGC:         arith.mulf
// BWDDATA-3X3-S2-NHWGC:         arith.addf

// Test 8. Forward 1x1 conv, NHWGC / GKYXC / NHWGK, *bf16*
// RUN: rocmlir-gen -batchsize=64 -in_channels=2048 -in_h=7 -in_w=7 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t bf16 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1X1-NHWGC-BF16

// FWD-1X1-NHWGC-BF16-LABEL: func.func @conv_cpu(
// FWD-1X1-NHWGC-BF16-NOT:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1X1-NHWGC-BF16:         arith.extf {{.*}} : bf16 to f32
// FWD-1X1-NHWGC-BF16:         tensor.collapse_shape
// FWD-1X1-NHWGC-BF16-SAME:    into tensor<1x3136x2048xf32>
// FWD-1X1-NHWGC-BF16:         tensor.collapse_shape
// FWD-1X1-NHWGC-BF16-SAME:    into tensor<1x2048x512xf32>
// FWD-1X1-NHWGC-BF16:         iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// FWD-1X1-NHWGC-BF16-SAME:    ins({{.*}} : tensor<1x3136x2048xf32>, tensor<1x2048x512xf32>)
// FWD-1X1-NHWGC-BF16-SAME:    outs({{.*}} : tensor<1x3136x512xf32>)
// FWD-1X1-NHWGC-BF16:         arith.mulf
// FWD-1X1-NHWGC-BF16:         arith.addf
// FWD-1X1-NHWGC-BF16:         arith.truncf {{.*}} : f32 to bf16

// Test 9. Forward 1x1 *stride-2* conv, NHWGC / GKYXC / NHWGK, f32
// RUN: rocmlir-gen -batchsize=64 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=2048 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 --operation conv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -t f32 --arch gfx942 -pv | \
// RUN:   rocmlir-opt --symbol-privatize='exclude=conv_cpu' -symbol-dce | \
// RUN:   rocmlir-opt --cpu-conv-to-gemm --mlir-print-local-scope | \
// RUN:   FileCheck %s --check-prefix=FWD-1X1-S2-NEG

// FWD-1X1-S2-NEG-LABEL: func.func @conv_cpu(
// FWD-1X1-S2-NEG:         iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// FWD-1X1-S2-NEG-SAME:    ins({{.*}} : tensor<64x14x14x1x1024xf32>, tensor<1x2048x1x1x1024xf32>)
// FWD-1X1-S2-NEG-SAME:    outs({{.*}} : tensor<64x7x7x1x2048xf32>)
// FWD-1X1-S2-NEG-NOT:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
