// Verify that the non-accel (FMA) tuning path proposes non-power-of-two
// kPerBlock candidates.
//
// gfx1201 with f32 takes the FMA (non-accel) path. K=48 has a non-pow2 divisor 24.
// Check that we are actually emitting this value.

// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1201 -g 1 -m 128 -k 48 -n 128 --perf_config= \
// RUN:   | rocmlir-gen --emit-tuning-space=full - \
// RUN:   | FileCheck %s

// CHECK: ,24,
