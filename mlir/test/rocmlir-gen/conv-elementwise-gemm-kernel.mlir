// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation conv_gemm -groupsize=1 -batchsize=2 -in_channels=256 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -gemmO=128 --transC=true --transO=false -fil_layout=gkcyx -in_layout=ngchw -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_conv_gemm
// CHECK-SAME: (%[[filterRaw:.*0]]: tensor<32768xf32>,
// CHECK-SAME: %[[inputRaw:.*1]]: tensor<524288xf32>,
// CHECK-SAME: %[[cRaw:.*2]]: tensor<16384xf32>,
// CHECK-SAME: %[[outputRaw:.*3]]: tensor<262144xf32>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.enable_splitk_for_tuning, rock.kernel}
// CHECK-NEXT: %[[filter:.*]] = rock.transform %[[filterRaw]] {{.*}} : tensor<32768xf32> to tensor<1x128x256x1x1xf32>
// CHECK-NEXT: %[[input:.*]] = rock.transform %[[inputRaw]] {{.*}} : tensor<524288xf32> to tensor<2x1x256x32x32xf32>
// CHECK-NEXT: %[[c:.*]] = rock.transform %[[cRaw]] {{.*}} : tensor<16384xf32> to tensor<1x128x128xf32>
// CHECK-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : tensor<262144xf32> to tensor<1x2048x128xf32>

// CHECK-NEXT: rock.conv_elementwise_gemm
// CHECK-NEXT: ab = conv(%[[filter]], %[[input]])
// CHECK: out = ab * tr %[[c]]
// CHECK: return

// CHECK-LABEL: func.func @host_naive_conv_gemm
// CHECK: %[[convTensor:.*]] = tosa.conv2d %[[inputTensor:.*]], %[[filterTensor:.*]], %{{.*}}, %{{.*}}, %{{.*}} {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x32x32x256xf32>, tensor<128x1x1x256xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x128xf32>
// CHECK-DAG: %[[abTensor:.*]] = tosa.reshape %[[convTensor]], %{{.*}} : (tensor<2x32x32x128xf32>, !tosa.shape<3>) -> tensor<1x2048x128xf32>
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[abTensor]], %[[cTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : (tensor<1x2048x128xf32>, tensor<1x128x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x128xf32>
// CHECK: return

// Verify that stride > 1 with padding that isn't evenly divisible by stride
// gets adjusted for TOSA's exact-divisibility requirement (pad_right trimmed).
// Without the fix, this fails with: 'tosa.conv2d' op expected ... to be wholly
// divisible by stride.
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation conv_gemm -groupsize=1 -batchsize=2 -in_channels=8 -out_channels=8 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -gemmO=4 --transC=false --transO=false -fil_layout=gkyxc -in_layout=nhwgc -t f16 -pv | rocmlir-opt | FileCheck %s --check-prefix=PADTRIM --enable-var-scope

// PADTRIM-LABEL: func.func @host_naive_conv_gemm
// PADTRIM-NOT: tosa.slice
// PADTRIM: tosa.conv2d {{.*}} {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>}
// PADTRIM: tosa.matmul
// PADTRIM: return

// Verify that when pad_right is already 0, the input is sliced instead.
// in_h=8, fil_h=3, stride=2, pad=0: fullExtent = 8-1+0+0-2 = 5, 5%2 = 1,
// pad can't absorb it, so input is sliced from 8 to 7.
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation conv_gemm -groupsize=1 -batchsize=2 -in_channels=8 -out_channels=8 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -gemmO=4 --transC=false --transO=false -fil_layout=gkyxc -in_layout=nhwgc -t f16 -pv | rocmlir-opt | FileCheck %s --check-prefix=SLICE --enable-var-scope

// SLICE-LABEL: func.func @host_naive_conv_gemm
// SLICE: tosa.slice %{{.*}}, %{{.*}}, %{{.*}} : (tensor<2x8x8x8xf16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x7x7x8xf16>
// SLICE: tosa.conv2d {{.*}} {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}
// SLICE: tosa.matmul
// SLICE: return

// A grouped convolution can't be fused with the GEMM that follows it, so
// conv+gemm rejects groupsize > 1. The rejection lives in
// detectMissingArguments(), so it fires whether or not a verifier is
// requested; both RUN lines below must produce the same error.
// RUN: not rocmlir-gen --arch gfx942:sramecc+:xnack- --operation conv_gemm -groupsize=4 -batchsize=2 -in_channels=256 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 -gemmO=128 -t f32 2>&1 | FileCheck %s --check-prefix=GROUP
// RUN: not rocmlir-gen --arch gfx942:sramecc+:xnack- --operation conv_gemm -groupsize=4 -batchsize=2 -in_channels=256 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 -gemmO=128 -t f32 -pv 2>&1 | FileCheck %s --check-prefix=GROUP

// GROUP: Group convolution not supported for conv+gemm
