// Regression tests for strided backward-data LDS overflows on gfx908 and
// gfx942.
//
// With no perf_config, the compiler auto-selects the first conservatively
// applicable quick-tuning entry for the arch/dtype/op. For a strided (>=2)
// backward-data conv with a non-1x1 filter, the Triton lowering stages the
// A (M x K) operand tile expanded in LDS, so the former front entry for
// gfx908 f32 conv (a numStages = 3 config) overflowed shared memory (64 KiB)
// and failed to compile with an LDS-overflow error. The quick-tuning list
// `initParametersF32ConvGfx908` now leads with a fitting numStages = 1 config,
// which orderParams rotates to the front, so selection lands on a config that
// fits LDS.

// RUN: rocmlir-gen -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 \
// RUN:   -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 \
// RUN:   -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 \
// RUN:   -padding_w_l=2 -padding_w_r=0 --operation conv_bwd_data \
// RUN:   -fil_layout=kc012 -in_layout=nc012 -out_layout=nk012 -t f32 \
// RUN:   --arch gfx908:sramecc+:xnack- \
// RUN: | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=AFFIX

// The auto-selected config must fit LDS: numStages = 1 (not the overflowing
// numStages = 3 front entry), with the applicability constraints satisfied
// (kpack = 1, numCTAs = 1, splitKFactor = 1).
// AFFIX: #rock.gemm_params<
// AFFIX-SAME: kpack = 1
// AFFIX-SAME: numCTAs = 1
// AFFIX-SAME: splitKFactor = 1
// AFFIX-SAME: numStages = 1

// The selected config must also actually compile without an LDS overflow.
// RUN: rocmlir-gen -groupsize=1 -batchsize=64 -in_channels=64 -out_channels=64 \
// RUN:   -in_h=4 -in_w=4 -fil_h=2 -fil_w=2 -dilation_h=1 -dilation_w=1 \
// RUN:   -conv_stride_h=2 -conv_stride_w=2 -padding_h_l=2 -padding_h_r=1 \
// RUN:   -padding_w_l=2 -padding_w_r=0 --operation conv_bwd_data \
// RUN:   -fil_layout=kc012 -in_layout=nc012 -out_layout=nk012 -t f32 \
// RUN:   --arch gfx908:sramecc+:xnack- \
// RUN: | rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=NO_OVERFLOW

// NO_OVERFLOW-NOT: exceeds LDS

// rock-fuse-sibling-loops merges the K-loops of kernel IDs that have the same
// trip count, so the merged loop keeps one A tile per kernel ID in LDS. With a
// 4x4 filter and stride 2, all four kernel IDs share one loop and one gradient
// tile, and the first gfx942 f32 conv entry (64x16x64, numStages = 2) needs
// 69632 B. Auto-selection has to skip it. This is the shape of
// fusion/pr-e2e/mixr-bwd-data-conv-dilation1-stride2.mlir, which needs a GPU.
// RUN: rocmlir-gen -groupsize=1 -batchsize=1 -in_channels=384 -out_channels=512 \
// RUN:   -in_h=64 -in_w=64 -fil_h=4 -fil_w=4 -dilation_h=1 -dilation_w=1 \
// RUN:   -conv_stride_h=2 -conv_stride_w=2 -padding_h=1 -padding_w=1 \
// RUN:   --operation conv_bwd_data -fil_layout=gk01c -in_layout=n01gc \
// RUN:   -out_layout=n01gk -t f32 --arch gfx942 \
// RUN: | rocmlir-driver -c 2>&1 | FileCheck %s --check-prefix=GFX942_NO_OVERFLOW

// GFX942_NO_OVERFLOW-NOT: exceeds LDS
