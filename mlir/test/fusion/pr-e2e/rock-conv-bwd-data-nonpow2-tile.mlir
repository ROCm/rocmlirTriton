// RUN: rocmlir-gen -pv --operation conv_bwd_data -t f32 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 16 --in_h 16 --in_w 16 --out_channels 32 --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --groupsize 1 --perf_config="gemm:v1:80,80,16,1,1,4,16,1,1,1,1" | rocmlir-driver -c | rocm-run | FileCheck %s

// Stride 2 in both spatial dimensions makes bwd_data lower to multiple
// disjoint per-phase stores. The 80x80 perf_config forces
// rock-decompose-nonpow2-tiles to split each blockwise GEMM/store tile, so this
// exercises the interaction between store-result threading and non-pow2 tile
// decomposition end-to-end against the CPU verifier.

// CHECK: [1 1 1]
