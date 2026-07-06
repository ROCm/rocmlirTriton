// End-to-end check that rock-fuse-sibling-loops fuses the per-sub-gemm K-loops
// produced by a backward-data convolution, even when those loops originate from
// *different* GEMMs of the same conv.
//
// A stride-2 conv_bwd_data lowers to four disjoint per-phase GEMMs (one per
// (stride_h, stride_w) phase). The 80x80 perf_config then makes
// rock-decompose-nonpow2-tiles split each of those GEMMs along both M (64 + 16)
// and N (64 + 16) into a 2x2 grid of pow2 sub-gemms, so
// rock-gridwise-gemm-to-blockwise emits 4 phases x 4 = 16 K-loops. Despite the
// differing padding/coordinate math per phase, all 16 loops share the same
// iteration space and are mutually independent, so rock-fuse-sibling-loops
// merges them into a single K-loop carrying all 16 accumulators. We inspect the
// IR right before rock-insert-output-fusion-loads.

// RUN: rocmlir-gen -operation conv_bwd_data -t f16 --arch gfx1100 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 16 --in_h 16 --in_w 16 --out_channels 32 --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --groupsize 1 --perf_config="gemm:v2:80,80,32,1,1,4,0,1,1,0,0,-1,-1,-1,-1,-1,-1" \
// RUN: | rocmlir-driver -c --mlir-print-ir-before=rock-insert-output-fusion-loads -o /dev/null 2>&1 \
// RUN: | FileCheck %s

// A single fused K-loop carries all 16 sub-gemm accumulators (4 stride phases x
// the 2x2 non-pow2 tile decomposition); every per-sub-gemm loop fused into one.
// CHECK: %{{.*}}:16 = scf.for
// CHECK-NOT: scf.for

// All 16 blockwise GEMMs now live inside that single loop.
// CHECK-COUNT-16: rock.blockwise_gemm
// CHECK-NOT: rock.blockwise_gemm
