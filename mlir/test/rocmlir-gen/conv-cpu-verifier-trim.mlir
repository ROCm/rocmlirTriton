// The CPU verifier reference convolution must trim trailing spatial slack off
// the (possibly padded) input so the operand extent matches the linalg.generic
// iteration footprint. Output spatial sizes use floor division, so when the
// input does not tile evenly by the stride there are trailing rows/cols that no
// output window reads. Without the trimming `tensor.extract_slice`, the
// generated `linalg.generic` fails verification (e.g. "inferred input/output
// operand #1 has shape's dimension ...").

// RUN: rocmlir-gen --arch gfx1100 --operation conv -t f32 -fil_layout=gkc01 -in_layout=ngc01 -out_layout=ngk01 -batchsize=1 -groupsize=1 -in_channels=2 -out_channels=2 -in_h=4 -in_w=5 -fil_h=2 -fil_w=2 --conv_stride_h=3 --conv_stride_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 -pv | FileCheck %s --check-prefix=FWD

// FWD-LABEL: func.func @conv_cpu
// FWD: %[[IN:.+]] = tensor.expand_shape %{{.+}} {{.*}} : tensor<40xf32> into tensor<1x1x2x4x5xf32>
// FWD: %[[SLICE:.+]] = tensor.extract_slice %[[IN]][0, 0, 0, 0, 0] [1, 1, 2, 2, 5] [1, 1, 1, 1, 1] : tensor<1x1x2x4x5xf32> to tensor<1x1x2x2x5xf32>
// FWD: linalg.generic {{.*}} ins(%[[SLICE]], %{{.+}} : tensor<1x1x2x2x5xf32>, tensor<1x2x2x2x2xf32>)


// A convolution whose input tiles evenly by the stride needs no trimming: the
// reference must not emit any slice on the input.

// RUN: rocmlir-gen --arch gfx1100 --operation conv -t f32 -fil_layout=gkc01 -in_layout=ngc01 -out_layout=ngk01 -batchsize=1 -groupsize=1 -in_channels=2 -out_channels=2 -in_h=5 -in_w=5 -fil_h=2 -fil_w=2 --conv_stride_h=1 --conv_stride_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 -pv | FileCheck %s --check-prefix=NOTRIM

// NOTRIM-LABEL: func.func @conv_cpu
// NOTRIM-NOT: tensor.extract_slice
