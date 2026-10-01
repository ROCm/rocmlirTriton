// RUN: rocmlir-gen --arch gfx1100 --operation conv -t f16 -fil_layout kyxc -in_layout nhwc -out_layout nhwk -batchsize 4 -in_channels 16 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 --padding_h 1 --padding_w 1 --dynamic-dims n | FileCheck %s --check-prefix=N
// RUN: rocmlir-gen --arch gfx1100 --operation conv -t f16 -fil_layout kyxc -in_layout nhwc -out_layout nhwk -batchsize 4 -in_channels 16 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 --padding_h 1 --padding_w 1 --dynamic-dims n,c,k,hi,wi | FileCheck %s --check-prefix=ALL
// RUN: rocmlir-gen --arch gfx1100 --operation conv_bwd_data -t f32 -fil_layout kcyx -in_layout nchw -out_layout nkhw -batchsize 3 -in_channels 16 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 --conv_stride_h 2 --conv_stride_w 2 --dynamic-dims n,c,k | FileCheck %s --check-prefix=BWD
// RUN: not rocmlir-gen --arch gfx1100 --operation conv -t f16 -batchsize 4 -in_channels 16 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 --dynamic-dims n,y 2>&1 | FileCheck %s --check-prefix=BAD-NAME
// RUN: not rocmlir-gen --arch gfx1100 --operation conv_bwd_data -t f32 -batchsize 3 -in_channels 16 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 --dynamic-dims hi 2>&1 | FileCheck %s --check-prefix=BWD-SPATIAL

// The kernel takes logical-rank arguments and stores the conv straight into
// the output; a dynamic hi/wi also makes ho/wo dynamic.
// N: func.func @rock_conv_gk01c_ng01c_ng01k(%{{.*}}: tensor<1x32x3x3x16xf16>, %{{.*}}: tensor<?x1x14x14x16xf16>, %[[OUT:.*]]: tensor<?x1x14x14x32xf16>) -> tensor<?x1x14x14x32xf16>
// N: %[[CONV:.*]] = rock.conv
// N-SAME: -> tensor<?x1x14x14x32xf16>
// N: rock.store %[[CONV]] to %[[OUT]]

// ALL: func.func @rock_conv_gk01c_ng01c_ng01k(%{{.*}}: tensor<1x?x3x3x?xf16>, %{{.*}}: tensor<?x1x?x?x?xf16>, %{{.*}}: tensor<?x1x?x?x?xf16>) -> tensor<?x1x?x?x?xf16>

// BWD: func.func @rock_conv_bwd_data_gkc01_ngc01_ngk01(%{{.*}}: tensor<1x?x?x3x3xf32>, %{{.*}}: tensor<?x1x?x6x6xf32>, %{{.*}}: tensor<?x1x?x14x14xf32>) -> tensor<?x1x?x14x14xf32>
// BWD: rock.conv_bwd_data

// BAD-NAME: invalid --dynamic-dims entry 'y'; expected one of: n,c,k,hi,wi

// BWD-SPATIAL: invalid --dynamic-dims entry 'hi'; expected one of: n,c,k
