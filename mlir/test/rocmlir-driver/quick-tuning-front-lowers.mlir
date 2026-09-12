// This test exercises the "consumer picks front() of quick tuning space"
// behavior end-to-end. For each scenario, we:
//   1) ask rocmlir-gen to emit the quick tuning space,
//   2) select the first perf-config entry (front()),
//   3) feed it back through rocmlir-gen --perf_config=...,
//   4) ensure rocmlir-driver -c lowers successfully.
//
// This mirrors the MIGraphX skip-benchmarking path expectation that front()
// should be directly usable.
//
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f32 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-GEMM-NT
// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f32 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False --perf_config="$(rocmlir-gen --arch %arch --operation gemm -t f16 -out_datatype f32 -g 1 -m 256 -k 128 -n 256 -transA=False -transB=False --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// RUN: rocmlir-gen --arch %arch --operation gemm -t f32 -g 1 -m 384 -k 192 -n 128 -transA=True -transB=False --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-GEMM-TA
// RUN: rocmlir-gen --arch %arch --operation gemm -t f32 -g 1 -m 384 -k 192 -n 128 -transA=True -transB=False --perf_config="$(rocmlir-gen --arch %arch --operation gemm -t f32 -g 1 -m 384 -k 192 -n 128 -transA=True -transB=False --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -g 1 -head_dim_qk 64 -head_dim_v 64 -num_heads_q 32 -num_heads_kv 32 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-ATTN
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -g 1 -head_dim_qk 64 -head_dim_v 64 -num_heads_q 32 -num_heads_kv 32 -seq_len_q 256 -seq_len_k 256 --perf_config="$(rocmlir-gen --arch %arch --operation attention -t f16 -g 1 -head_dim_qk 64 -head_dim_v 64 -num_heads_q 32 -num_heads_kv 32 -seq_len_q 256 -seq_len_k 256 --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// RUN: rocmlir-gen --arch %arch --operation gemm_gemm -t f16 -g 1 -m 256 -k 128 -n 256 -gemmO 128 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-GEMM-GEMM
// RUN: rocmlir-gen --arch %arch --operation gemm_gemm -t f16 -g 1 -m 256 -k 128 -n 256 -gemmO 128 --perf_config="$(rocmlir-gen --arch %arch --operation gemm_gemm -t f16 -g 1 -m 256 -k 128 -n 256 -gemmO 128 --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// RUN: rocmlir-gen --arch %arch --operation conv_gemm -t f16 -groupsize=1 -batchsize=2 -in_channels=8 -out_channels=8 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -gemmO=4 --transC=false --transO=false -fil_layout=gkyxc -in_layout=nhwgc --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-CONV-GEMM
// RUN: rocmlir-gen --arch %arch --operation conv_gemm -t f16 -groupsize=1 -batchsize=2 -in_channels=8 -out_channels=8 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -gemmO=4 --transC=false --transO=false -fil_layout=gkyxc -in_layout=nhwgc --perf_config="$(rocmlir-gen --arch %arch --operation conv_gemm -t f16 -groupsize=1 -batchsize=2 -in_channels=8 -out_channels=8 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=1 -padding_h_r=1 -padding_w_l=1 -padding_w_r=1 -gemmO=4 --transC=false --transO=false -fil_layout=gkyxc -in_layout=nhwgc --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// RUN: rocmlir-gen --arch %arch --operation conv -t f16 -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw --batchsize=4 --in_channels=16 --out_channels=16 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --groupsize=1 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-CONV
// RUN: rocmlir-gen --arch %arch --operation conv -t f16 -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw --batchsize=4 --in_channels=16 --out_channels=16 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --groupsize=1 --perf_config="$(rocmlir-gen --arch %arch --operation conv -t f16 -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw --batchsize=4 --in_channels=16 --out_channels=16 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 --groupsize=1 --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// FP4's front() can be invalid for a reason no perf-config field states
// outright: a decomposed scaled dot may hand a thread a partial packed-upcast
// group. The arch is pinned because gfx950 is where the scaled matrix
// accelerator exists, and lowering to a pinned target needs no GPU.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f4E2M1FN -out_dtype f32 --scaledGemm -g 1 -m 256 -k 1024 -n 1280 -transA=false -transB=true --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-GEMM-FP4
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f4E2M1FN -out_dtype f32 --scaledGemm -g 1 -m 256 -k 1024 -n 1280 -transA=false -transB=true --perf_config="$(rocmlir-gen --arch gfx950 --operation gemm -t f4E2M1FN -out_dtype f32 --scaledGemm -g 1 -m 256 -k 1024 -n 1280 -transA=false -transB=true --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// The unscaled 4-bit GEMM shares the upcast path, so it gets the same check.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f4E2M1FN -out_datatype f32 -g 1 -m 256 -k 1024 -n 1280 -transA=false -transB=true --perf_config="$(rocmlir-gen --arch gfx950 --operation gemm -t f4E2M1FN -out_datatype f32 -g 1 -m 256 -k 1024 -n 1280 -transA=false -transB=true --emit-tuning-space=quick | sed -n '1p')" | rocmlir-driver -c -o /dev/null
//
// CHECK-GEMM-NT: gemm:
// CHECK-GEMM-TA: gemm:
// CHECK-ATTN: attn:
// CHECK-GEMM-GEMM: attn:
// CHECK-CONV-GEMM: attn:
// CHECK-CONV: gemm:
// CHECK-GEMM-FP4: gemm:
