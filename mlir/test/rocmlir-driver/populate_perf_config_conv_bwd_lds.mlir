// Regression test for the strided backward-data LDS applicability bound.
//
// With no perf_config, the compiler auto-selects the first conservatively
// applicable quick-tuning entry. For a strided (>=2) backward-data conv with a
// non-1x1 filter, the Triton lowering stages the A (M x K) operand tile
// expanded in LDS, so a numStages>=2 config can overflow shared memory on
// gfx908 (64 KiB). `isGemmParamsConservativelyApplicable` charges the A-tile an
// `mLdsExpansionFactor = prod_i min(stride_i, filter_i)` (here min(2,2)^2 = 4),
// which rejects the overflowing entries and steers selection to a fitting
// (numStages = 1) config. Before this bound the front entry (numStages = 3)
// was selected and failed to compile with an LDS-overflow error.

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
