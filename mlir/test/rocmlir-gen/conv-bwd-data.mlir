// RUN: rocmlir-gen --arch gfx908:sramecc+:xnack- -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=BWD

// BWD: module attributes {rock.arch = "[[$ARCH:.*]]"}
// BWD: @rock_conv_bwd_data_gk01c_n01gc_n01gk({{.*}} rock.kernel
// BWD: rock.conv_bwd_data(%0, %1) {{.*}}

// Test mixed dtype support: verify that CPU validation function uses correct types
// when fil_dtype, in_dtype, and out_dtype are all different
// RUN: rocmlir-gen --arch gfx942 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=1 -in_channels=32 -out_channels=32 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --operation=conv_bwd_data -fil_dtype f16 -in_dtype f32 -out_dtype f16 -pv | FileCheck %s --check-prefix=MIXED_DTYPE

// MIXED_DTYPE: func.func @rock_conv_bwd_data{{.*}}(%arg0: tensor<{{[0-9]+}}xf16>, %arg1: tensor<{{[0-9]+}}xf16>, %arg2: tensor<{{[0-9]+}}xf32>)
// MIXED_DTYPE: func.func @conv_bwd_data_cpu(%arg0: tensor<{{[0-9]+}}xf16>, %arg1: tensor<{{[0-9]+}}xf16>, %arg2: tensor<{{[0-9]+}}xf32>) -> tensor<{{[0-9]+}}xf32>
