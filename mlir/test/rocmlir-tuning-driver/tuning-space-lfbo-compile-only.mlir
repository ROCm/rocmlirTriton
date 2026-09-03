// `lfbo` chooses each batch from the previous batch's timings, so it cannot be
// driven by the ahead-of-time compile phase, which never runs a kernel. The
// driver has to say so up front rather than compiling a first batch and leaving
// a truncated artifact bundle behind. Reaching this error also confirms that
// `lfbo` is accepted as a --tuning-space value at all.
//
// Host-only: the rejection happens before any HIP work, so this needs no GPU.

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f32 -g 1 -m 256 -k 256 -n 256 -transA=False -transB=False --perf_config= \
// RUN: | not rocmlir-tuning-driver --tuning-space=lfbo --compile-only=%t.bundle 2>&1 \
// RUN: | FileCheck %s

// CHECK: error: --compile-only cannot be used with the 'lfbo' tuning space
// CHECK-SAME: needs the timings of one batch to decide what to compile next
