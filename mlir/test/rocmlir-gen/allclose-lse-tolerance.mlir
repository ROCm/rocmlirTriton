// Tests that the LSE (log-sum-exp) output of an attention kernel gets relaxed
// tolerances (atol=1.5e-2, rtol=5e-4) while the main attention output (O) keeps
// the normal K-scaled tolerance.
//
// The detection heuristic in rocmlir-gen is:
//   isLSE = hasAttention && !isFirstOutput && isa<Float32Type>(elemType)
//
// This test uses a clone-harness flow with a MIGraphX attention kernel that
// returns two outputs: O (f16) and LSE (f32). After lowering through the
// MIGraphX pipeline, `rock.attention` is present in the module, and the
// verifier emits two `_verify` functions with different tolerances.

// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s \
// RUN:   | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel \
// RUN:   | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention --verifier clone - \
// RUN:   | FileCheck %s --check-prefix=ATTN_LSE --enable-var-scope

// The module must contain a rock.attention op for the heuristic to fire.
// ATTN_LSE: rock.attention

// --- First verify function: attention output O (f16) ---
// Gets normal K-scaled tolerance, NOT the LSE relaxed values.
// ATTN_LSE:      func.func @mlir_attention_verify
// The f16 atol must NOT be the LSE override (1.5e-2 = 0.015).
// ATTN_LSE-NOT:  arith.constant 1.500000e-02 : f32
// f16 K-scaled atol and rtol followed by allclose call.
// ATTN_LSE:      call @mcpuVerifyFloatAllclose

// --- Second verify function: LSE output (f32) ---
// Gets the relaxed LSE tolerance: atol=1.5e-2, rtol=5e-4.
// ATTN_LSE:      func.func @mlir_attention_verify
// ATTN_LSE:      arith.constant 1.500000e-02 : f32
// ATTN_LSE-NEXT: arith.constant 5.000000e-04 : f32
// ATTN_LSE:      call @mcpuVerifyFloatAllclose

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg1: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg2: !migraphx.shaped<12x256x256xsi8, 65536x256x1>, %arg3: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg4: !migraphx.shaped<12x256x256xf16, 65536x256x1>) -> (!migraphx.shaped<12x256x256xf16, 65536x256x1>, !migraphx.shaped<12x256x1xf32, 256x1x1>)  attributes {rock.kernel} {
    %0 = migraphx.literal(dense<1.000000e+01> : tensor<12x256x256xf16>) : <12x256x256xf16, 65536x256x1>
    %1 = migraphx.literal(dense<1.250000e-01> : tensor<12x256x256xf16>) : <12x256x256xf16, 65536x256x1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x1x256>
    %3 = migraphx.transpose %arg3 {permutation = [0, 1, 3, 2]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x1x256>
    %4 = migraphx.reshape %3 {dims = [12, 256, 256]} : <1x12x256x256xf16, 786432x65536x1x256> -> <12x256x256xf16, 65536x1x256>
    %5 = migraphx.dot %arg0, %2 : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x1x256> -> <1x12x256x256xf16, 786432x65536x256x1>
    %6 = migraphx.reshape %5 {dims = [12, 256, 256]} : <1x12x256x256xf16, 786432x65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %7 = migraphx.mul %6, %1 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %8 = migraphx.where %arg2, %7, %0 : <12x256x256xsi8, 65536x256x1>, <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %10 = migraphx.reshape %8 {dims = [12, 256, 256]} : <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %11 = migraphx.reduce_max %10 {axes = [2]} : <12x256x256xf16, 65536x256x1> -> <12x256x1xf16, 256x1x1>
    %12 = migraphx.reshape %11 {dims = [12, 256, 1]} : <12x256x1xf16, 256x1x1> -> <12x256x1xf16, 256x1x1>
    %13 = migraphx.multibroadcast %12 {out_dyn_dims = [], out_lens = [12, 256, 256]} : <12x256x1xf16, 256x1x1> -> <12x256x256xf16, 256x1x0>
    %14 = migraphx.sub %8, %13 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 256x1x0> -> <12x256x256xf16, 65536x256x1>
    %15 = migraphx.exp %14 : <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %16 = migraphx.reshape %15 {dims = [12, 256, 256]} : <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    %17 = migraphx.reduce_sum %16 {axes = [2]} : <12x256x256xf16, 65536x256x1> -> <12x256x1xf16, 256x1x1>
    %18 = migraphx.reshape %17 {dims = [12, 256, 1]} : <12x256x1xf16, 256x1x1> -> <12x256x1xf16, 256x1x1>
    %19 = migraphx.multibroadcast %18 {out_dyn_dims = [], out_lens = [12, 256, 256]} : <12x256x1xf16, 256x1x1> -> <12x256x256xf16, 256x1x0>
    %20 = migraphx.div %15, %19 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 256x1x0> -> <12x256x256xf16, 65536x256x1>
    
    %se = migraphx.convert %17 {target_type = 2 : i64} : <12x256x1xf16, 256x1x1> to <12x256x1xf32, 256x1x1>
    %max = migraphx.convert %11 {target_type = 2 : i64} : <12x256x1xf16, 256x1x1> to <12x256x1xf32, 256x1x1>
    %lse = migraphx.log %se : <12x256x1xf32, 256x1x1> -> <12x256x1xf32, 256x1x1>
    %lse_add = migraphx.add %lse, %max : <12x256x1xf32, 256x1x1>, <12x256x1xf32, 256x1x1> -> <12x256x1xf32, 256x1x1>
    %22 = migraphx.dot %20, %4 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x1x256> -> <12x256x256xf16, 65536x256x1>
    %23 = migraphx.mul %22, %arg4 : <12x256x256xf16, 65536x256x1>, <12x256x256xf16, 65536x256x1> -> <12x256x256xf16, 65536x256x1>
    return %23, %lse_add : !migraphx.shaped<12x256x256xf16, 65536x256x1>, !migraphx.shaped<12x256x1xf32, 256x1x1>
  }
}
