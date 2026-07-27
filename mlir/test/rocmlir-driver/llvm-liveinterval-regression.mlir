// Regression for an LLVM AMDGPU backend crash in Greedy Register Allocator
// (LiveIntervalUnion::extract "Inconsistent LiveInterval") seen while tuning
// this convfp16 workload on gfx1200. This is compile-only (no runtime
// execution), so it is stable and fast in CI.
//
// Associated tuning workload:
//   convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 1 -c 128 -H 960 -W 960
//   -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
//
// RUN: rocmlir-gen --operation conv -t f16 --arch gfx1200 --num_cu 32 --num_chiplets 1 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 \
// RUN:   --in_channels 128 --in_h 960 --in_w 960 --out_channels 128 \
// RUN:   --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 \
// RUN:   --groupsize 1 \
// RUN:   --perf_config=gemm:v4:64,16,128,1,1,8,0,1,3,0,0,-1,-1,-1,-1,-1,-1 \
// RUN: | rocmlir-driver -c -arch gfx1200 2>&1 | FileCheck %s
//
// CHECK: triton.hsaco
// CHECK-NOT: Inconsistent LiveInterval
