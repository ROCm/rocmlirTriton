// Regression test for a Triton bug exposed by small non-power-of-2 kPerBlock values 
// in gfx950: kPerBlock=18 which generates 2 segments, {16,2}. Before the bugfix,
// compilation failed with `builtin.unrealized_conversion_cast`.
// 
// The issue for us was that, for certain small non-power-of-2 kPerBlock values,
// the generated buffer load op was not lowered in LoadStoreOpToLLVM because
// canLoadDirectToLDS returns false and, as a consequence, the op was left unlowered,
// generating illegal IR.
//
// Upstream fix: https://github.com/triton-lang/triton/pull/10928
//
// gfx950 is currently opted out of non-power-of-2 kPerBlock altogether because an
// unfixed LLVM backend bug miscompiles the peeled K loop (see
// rock::supportsNonPow2KPerBlock), so this perf_config is now rejected up front
// instead of reaching codegen. Once that opt-out is lifted, restore the positive
// form of this test: drop the `not`, and check for `triton.hsaco`.

// RUN: rocmlir-gen --operation conv -t f32 \
// RUN:   --arch gfx950:sramecc+:xnack- --num_cu 256 --num_chiplets 8 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 64 --in_channels 32 --in_h 147 --in_w 147 --out_channels 64 \
// RUN:   --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 \
// RUN:   --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   --perf_config="gemm:v4:64,32,18,1,1,1,16,1,2,0,0,-1,-1,-1,-1,-1,-1" \
// RUN:   | not rocmlir-driver -c --arch gfx950:sramecc+:xnack- 2>&1 | FileCheck %s

// CHECK: error: kPerBlock=18 must be a positive power of two
