// Positive test for `--verifier=mlir-strict` (a.k.a. `-pv_strict`).
//
// In strict mode `rocmlir-gen` emits a `linalg.generic` pair that bitcasts the
// narrow-float qk tensor through its bit-equivalent integer type. That round
// trip materialises the GPU-faithful narrow-precision rounding 

// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -pv_strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT
// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 --verifier=mlir-strict | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=STRICT
// RUN: rocmlir-gen --arch %arch --operation attention -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias -t bf16 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=NONSTRICT

// Strict mode: the post-`scale*QK` mul and the post-bias add are each followed
// by a bf16 -> i16 -> bf16 bitcast pair lowered as `linalg.generic`s.
//
// STRICT-LABEL: func.func @host_naive_attention
// STRICT:       tosa.mul {{.*}} : ({{.*}}, {{.*}}, tensor<1xi8>) -> [[SHAPE:tensor<.*xbf16>]]
// STRICT:       linalg.generic
// STRICT:         arith.bitcast %{{.*}} : bf16 to i16
// STRICT:       linalg.generic
// STRICT:         arith.bitcast %{{.*}} : i16 to bf16
// STRICT:       tosa.add {{.*}} -> [[SHAPE]]
// STRICT:       linalg.generic
// STRICT:         arith.bitcast %{{.*}} : bf16 to i16
// STRICT:       linalg.generic
// STRICT:         arith.bitcast %{{.*}} : i16 to bf16
// STRICT:       return

// Non-strict (default `-pv`) mode: same op sequence, but no bitcast round trip.
//
// NONSTRICT-LABEL: func.func @host_naive_attention
// NONSTRICT-NOT:   arith.bitcast {{.*}} : bf16 to i16
// NONSTRICT-NOT:   arith.bitcast {{.*}} : i16 to bf16
// NONSTRICT:       return
