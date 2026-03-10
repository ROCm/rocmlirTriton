// RUN: rocmlir-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1 --operation=conv_bwd_data --arch %arch | rocmlir-driver -c --mlir-print-ir-after=rock-conv-to-gemm 2>&1 | FileCheck %s --check-prefix=STRIDE2

// RUN: rocmlir-gen -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data --arch %arch | rocmlir-driver -c --mlir-print-ir-after=rock-conv-to-gemm 2>&1 | FileCheck %s --check-prefix=STRIDE2_GKYXC

// This config requires a zero initialization for the bwd_data input.
// RUN: rocmlir-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data --arch %arch | FileCheck %s --check-prefix=STRIDE2_1x1_TOP_LEVEL
// Check after -rock-lowering, only gemm with corresponding kernel IDs exists.
// RUN: rocmlir-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data --arch %arch | rocmlir-driver -c --mlir-print-ir-after=rock-conv-to-gemm 2>&1 | FileCheck %s --check-prefix=STRIDE2_1x1_LOWERING

// Multiple gemms are emitted within a single function for stride > 1.
// STRIDE2-COUNT-4: {{rock.gemm.*}}

// STRIDE2_GKYXC: {{rock.gemm.*}}

// STRIDE2_1x1_TOP_LEVEL: %arg2: tensor<{{.*}}xf32> {rock.prefill = 0.000000e+00 : f32}
// STRIDE2_1x1_TOP_LEVEL: [[exp0:%.+]] = rock.transform %arg0 by {{.*}} : tensor<1024xf32> to tensor<1x32x32x1x1xf32>
// STRIDE2_1x1_TOP_LEVEL: [[exp1:%.+]] = rock.transform %arg1 by {{.*}} : tensor<65536xf32> to tensor<32x1x32x8x8xf32>
// STRIDE2_1x1_TOP_LEVEL: {{.*}} = rock.conv_bwd_data([[exp0]], [[exp1]]) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [2 : index, 2 : index]} : tensor<1x32x32x1x1xf32>, tensor<32x1x32x8x8xf32>

// STRIDE2_1x1_LOWERING: {{rock.gemm.*}}
