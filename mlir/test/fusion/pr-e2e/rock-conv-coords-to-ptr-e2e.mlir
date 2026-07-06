// A padded 3x3 conv makes the input tensor's validity mask depend on the GEMM
// K-loop IV, so rock-transforms-invariant-code-motion takes the
// carry path and emits a rock.coords_to_ptr (lowered by
// rock-transforms-to-pointer-arith).

// RUN: rocmlir-gen -pv --operation conv -t i8 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 128 --in_h 28 --in_w 28 --out_channels 128 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   --perf_config="$(rocmlir-gen --operation conv -t i8 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 128 --in_h 28 --in_w 28 --out_channels 128 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --groupsize 1 --emit-tuning-space=quick | sed -n '1p' | sed -e 's/^gemm:v2:/gemm:v3:/' -e 's/$/,1/')" \
// RUN:   | rocmlir-driver -c | rocm-run | FileCheck %s

// CHECK: [1 1 1]
