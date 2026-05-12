// Tests for the CPU attention verifier. The CPU verifier exists in two flavours:
// -pv: the fast verifier. May not exactly match GPU semantics.
// -pv_strict. Matches GPU semantics.

// =============================================================================
// `-pv_strict` matches GPU semantics
// =============================================================================
//
// (1) No scale/bias, narrow-float type. The GPU's truncf/extf round-trip on
// the first-GEMM output folds away, so the GPU effectively runs the chain at
// f32. The strict CPU verifier promotes the first GEMM output to f32 too:
//   * `tosa.matmul` produces f32, matching the GPU's effective behaviour.
//   * The pre-softmax `tosa.cast` is now f32 -> f32 (no real narrowing).
//   * The output `tosa.cast` casts the f32 second-GEMM result back to the
//     narrow output element type once at the boundary, matching the single
//     bf16 store the GPU performs to the output buffer.

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 -t bf16 -pv_strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-PLAIN

// STRICT-PLAIN-LABEL: func.func @host_naive_attention
// First GEMM: bf16 inputs, f32 output (promoted to match folded GPU semantics).
// STRICT-PLAIN:       tosa.matmul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xf32>
// Pre-softmax cast is now f32 -> f32 (identity, no narrowing).
// STRICT-PLAIN:       tosa.cast {{.*}} : (tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
// Second GEMM: result also f32 (softmax stays at f32 throughout).
// STRICT-PLAIN:       tosa.matmul {{.*}} -> tensor<{{.*}}xf32>
// Output cast: single f32 -> bf16 at the boundary, matching the one bf16
// store the GPU does to the output buffer.
// STRICT-PLAIN:       tosa.cast {{.*}} : (tensor<{{.*}}xf32>) -> tensor<{{.*}}xbf16>
// STRICT-PLAIN:       return

// (2) With scale and/or bias, narrow-float type. The GPU genuinely computes
// scale*QK / qk+bias in the narrow type (the truncf/extf round-trip cannot
// fold because narrow arithmetic sits between). The strict CPU verifier keeps
// the first GEMM output narrow so it rounds between operations the same way:
//   * `tosa.matmul` produces bf16, matching the rounded GPU behaviour.
//   * scale*QK and/or qk+bias run on bf16 tensors.
//   * The pre-softmax `tosa.cast` widens bf16 -> f32 once before softmax.

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 --with-attn-bias -t bf16 -pv_strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-BIAS

// STRICT-BIAS-LABEL: func.func @host_naive_attention
// STRICT-BIAS:       tosa.matmul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// STRICT-BIAS:       tosa.add {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// STRICT-BIAS:       tosa.cast {{.*}} : (tensor<{{.*}}xbf16>) -> tensor<{{.*}}xf32>
// STRICT-BIAS:       return

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 --with-attn-scale -t bf16 -pv_strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-SCALE

// STRICT-SCALE-LABEL: func.func @host_naive_attention
// STRICT-SCALE:       tosa.matmul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// STRICT-SCALE:       tosa.mul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<1xi8>) -> tensor<{{.*}}xbf16>
// STRICT-SCALE:       tosa.cast {{.*}} : (tensor<{{.*}}xbf16>) -> tensor<{{.*}}xf32>
// STRICT-SCALE:       return

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 --with-attn-scale --with-attn-bias -t bf16 -pv_strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-BOTH

// STRICT-BOTH-LABEL: func.func @host_naive_attention
// STRICT-BOTH:       tosa.matmul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// STRICT-BOTH:       tosa.mul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<1xi8>) -> tensor<{{.*}}xbf16>
// STRICT-BOTH:       tosa.add {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// STRICT-BOTH:       tosa.cast {{.*}} : (tensor<{{.*}}xbf16>) -> tensor<{{.*}}xf32>
// STRICT-BOTH:       return

// (3) `f16` narrow-float type behaves the same as bf16: with no scale/bias
// the first-GEMM output is promoted to f32.

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 -t f16 -pv_strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-PLAIN-F16

// STRICT-PLAIN-F16-LABEL: func.func @host_naive_attention
// STRICT-PLAIN-F16:       tosa.matmul {{.*}} : (tensor<{{.*}}xf16>, tensor<{{.*}}xf16>, tensor<{{.*}}xf16>, tensor<{{.*}}xf16>) -> tensor<{{.*}}xf32>
// STRICT-PLAIN-F16:       tosa.cast {{.*}} : (tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
// STRICT-PLAIN-F16:       tosa.matmul {{.*}} -> tensor<{{.*}}xf32>
// STRICT-PLAIN-F16:       tosa.cast {{.*}} : (tensor<{{.*}}xf32>) -> tensor<{{.*}}xf16>
// STRICT-PLAIN-F16:       return

// (4) The `verifier=mlir-strict` alias spelling must produce the same IR as
// `-pv_strict`.
// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 -t bf16 --verifier=mlir-strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-PLAIN

// =============================================================================
// `-pv` does *not* match GPU semantics: the first GEMM output is kept narrow
// even when the GPU effectively runs the chain at f32.
// =============================================================================
//
// (1) No scale/bias, narrow-float type. The GPU effectively runs the first
// GEMM and pre-softmax at f32 (round-trip folds away). The fast `-pv` CPU
// verifier, by design, rounds the first GEMM output to the narrow type:
//   * `tosa.matmul` produces bf16 -- a single rounding the GPU does not do.
//   * The pre-softmax `tosa.cast` widens bf16 -> f32 before softmax, while
//     on the GPU the chain stays at f32 throughout.
// This is the precision divergence `-pv_strict` exists to close. Pinning it
// here documents that `-pv` deliberately trades fidelity for speed.

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 -t bf16 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=NONSTRICT-PLAIN

// NONSTRICT-PLAIN-LABEL: func.func @host_naive_attention
// NONSTRICT-PLAIN:       tosa.matmul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// NONSTRICT-PLAIN:       tosa.cast {{.*}} : (tensor<{{.*}}xbf16>) -> tensor<{{.*}}xf32>
// NONSTRICT-PLAIN:       return

// (2) `-pv` with scale and/or bias keeps the narrow chain too -- same op
// sequence as `-pv_strict` for these cases, because both flavours need to
// match a GPU that genuinely runs in the narrow type. Pinning this confirms
// `-pv` and `-pv_strict` only differ in the no-scale-no-bias case.

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 --with-attn-scale --with-attn-bias -t bf16 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=NONSTRICT-BOTH

// NONSTRICT-BOTH-LABEL: func.func @host_naive_attention
// NONSTRICT-BOTH:       tosa.matmul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// NONSTRICT-BOTH:       tosa.mul {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>, tensor<1xi8>) -> tensor<{{.*}}xbf16>
// NONSTRICT-BOTH:       tosa.add {{.*}} : (tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
// NONSTRICT-BOTH:       tosa.cast {{.*}} : (tensor<{{.*}}xbf16>) -> tensor<{{.*}}xf32>
// NONSTRICT-BOTH:       return

// =============================================================================
// `f32` attention is unaffected by the strict flag: there is no narrow
// rounding to mimic in the first place.
// =============================================================================

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 -t f32 -pv_strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-F32
// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 -t f32 -pv         | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT-F32

// STRICT-F32-LABEL: func.func @host_naive_attention
// STRICT-F32:       tosa.matmul {{.*}} : (tensor<{{.*}}xf32>, tensor<{{.*}}xf32>, tensor<{{.*}}xf32>, tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
// STRICT-F32:       return
