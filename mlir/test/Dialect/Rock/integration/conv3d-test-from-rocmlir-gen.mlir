// COM: 3D forward convolution, gkc012/ngc012/ngk012, f32
// COM: n=1, g=1, c=1, k=1, input=8x8x8, filter=3x3x3, output=6x6x6, stride=1, dilation=1, padding=0
// RUN: rocmlir-gen --arch %arch --operation=conv -t f32 -pv -fil_layout=gkc012 -in_layout=ngc012 -out_layout=ngk012 -batchsize=1 -groupsize=1 -in_channels=1 -out_channels=1 -in_d=8 -in_h=8 -in_w=8 -fil_d=3 -fil_h=3 -fil_w=3 --conv_stride_d=1 --conv_stride_h=1 --conv_stride_w=1 --dilation_d=1 --dilation_h=1 --dilation_w=1 --padding_d=0 --padding_h=0 --padding_w=0 | rocmlir-driver -kernel-pipeline=full | rocm-run | FileCheck %s
// CHECK: [1 1 1]
