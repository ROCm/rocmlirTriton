// Compile-only regression test for the SplitKit "Interference" assertion fixed
// by llvm-patches/patch209704.patch: this conv_bwd_data f32 workload on gfx908
// aborts in the greedy register allocator without the fix.

// RUN: rocmlir-gen --operation conv_bwd_data -t f32 \
// RUN:   --arch gfx908:sramecc+:xnack- --num_cu 120 --num_chiplets 1 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 384 --in_h 32 --in_w 32 --out_channels 512 \
// RUN:   --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 \
// RUN:   --groupsize 1 \
// RUN:   --perf_config="gemm:v1:16,64,256,1,1,1,16,3,1,0,0" \
// RUN:   | rocmlir-driver -c | FileCheck %s

// CHECK: triton.hsaco
