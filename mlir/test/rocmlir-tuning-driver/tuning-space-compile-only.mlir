// The adaptive tuning spaces choose each batch from the previous batch's
// timings, so none of them can be driven by the ahead-of-time compile phase,
// which never runs a kernel. The driver has to say so up front rather than
// compiling a first batch and leaving a truncated artifact bundle behind.
//
// Reaching this error also confirms that each of these is accepted as a
// --tuning-space value at all, which for `llm` and `llm-lfbo` is the only
// check that does not need a model.
//
// Host-only: the rejection happens before any HIP work, so this needs no GPU,
// and `llm` gets this far without ever starting its helper.

// RUN: rocmlir-gen --arch %arch -operation gemm -t f16 -out_datatype f32 -g 1 -m 256 -k 256 -n 256 -transA=False -transB=False --perf_config= > %t.mlir

// RUN: not rocmlir-tuning-driver --tuning-space=lfbo --compile-only=%t.bundle %t.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=LFBO
// LFBO: error: --compile-only cannot be used with the 'lfbo' tuning space
// LFBO-SAME: needs the timings of one batch to decide what to compile next

// RUN: not rocmlir-tuning-driver --tuning-space=llm --compile-only=%t.bundle %t.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=LLM
// LLM: error: --compile-only cannot be used with the 'llm' tuning space
// LLM-SAME: needs the timings of one batch to decide what to compile next

// RUN: not rocmlir-tuning-driver --tuning-space=llm-lfbo --compile-only=%t.bundle %t.mlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=HYBRID
// HYBRID: error: --compile-only cannot be used with the 'llm-lfbo' tuning space
// HYBRID-SAME: needs the timings of one batch to decide what to compile next
