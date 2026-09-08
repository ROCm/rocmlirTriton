// RUN: rocmlir-opt --migraphx-to-tosa --mlir-print-op-generic -verify-diagnostics %s | FileCheck %s --check-prefix=TOSA
// RUN: rocmlir-opt --migraphx-to-tosa="disable-fast-math=true" --mlir-print-op-generic -verify-diagnostics %s | FileCheck %s --check-prefix=IEEE
// RUN: rocmlir-opt -pass-pipeline="builtin.module(func.func(migraphx-to-tosa),func.func(rocmlir-custom-tosa-to-linalg),func.func(tosa-to-linalg-named),func.func(tosa-to-linalg))" -verify-diagnostics %s | FileCheck %s --check-prefixes=LINALG,UNSIGNED
// RUN: rocmlir-opt -pass-pipeline="builtin.module(func.func(migraphx-to-tosa{disable-fast-math=true}),func.func(rocmlir-custom-tosa-to-linalg),func.func(tosa-to-linalg-named),func.func(tosa-to-linalg))" -verify-diagnostics %s | FileCheck %s --check-prefix=LINALG_IEEE

// Every float maximum/minimum/clamp carries a TOSA nan_mode, and which one it
// gets is a whole-pipeline decision rather than a per-op or per-kernel one.
// By default the lowering assumes no NaN occurs anywhere in the kernel's
// dataflow and emits IGNORE; `-disable-fast-math` restores PROPAGATE, i.e.
// exact IEEE semantics.
//
// That assumption is stronger than a statement about the kernel's arguments:
// nothing here proves an operand finite, so an intermediate NaN violates it as
// much as an incoming one. A zero scale making `0 * inf` inside
// `migraphx.quantizelinear`, an accumulation overflowing to `inf + (-inf)`, or
// a division by a zero softmax row-sum all break it. Signed-zero behavior is
// separately relaxed later with `nsz`.
//
// IGNORE is what lets a clamp pair collapse into one `tt.clampf` (a single
// v_med3 on AMD hardware); rock-tosa-to-elementwise.mlir covers that side.
// The LINALG lines below are upstream's reading of IGNORE, not the one the
// kernels get: it keeps the propagating arith.maximumf and undoes the
// propagation with a compare and select per operand. Both are checked because
// the CPU reference still goes through upstream.

// TOSA: sym_name = "max_f32"
// TOSA: "tosa.maximum"
// TOSA-SAME: nan_mode
// TOSA-SAME: IGNORE

// IEEE: sym_name = "max_f32"
// IEEE: "tosa.maximum"
// IEEE-SAME: nan_mode
// IEEE-SAME: PROPAGATE

// LINALG-LABEL: func.func @max_f32
// LINALG: arith.maximumf
// LINALG: arith.cmpf uno
// LINALG: arith.cmpf uno
// LINALG: arith.select
// LINALG: arith.select

// LINALG_IEEE-LABEL: func.func @max_f32
// LINALG_IEEE: arith.maximumf
// LINALG_IEEE-NOT: arith.maxnumf
func.func @max_f32(%arg0: !migraphx.shaped<2x4xf32, 4x1>, %arg1: !migraphx.shaped<2x4xf32, 0x1>) -> !migraphx.shaped<2x4xf32, 4x1> {
  %0 = migraphx.max %arg0, %arg1 : <2x4xf32, 4x1>, <2x4xf32, 0x1> -> <2x4xf32, 4x1>
  return %0 : !migraphx.shaped<2x4xf32, 4x1>
}

// A unit-extent operand with a zero stride still goes through the same path,
// so the mode does not depend on the broadcast shape.

// TOSA: sym_name = "max_unit_zero_stride"
// TOSA: "tosa.maximum"
// TOSA-SAME: nan_mode
// TOSA-SAME: IGNORE

// IEEE: sym_name = "max_unit_zero_stride"
// IEEE: "tosa.maximum"
// IEEE-SAME: nan_mode
// IEEE-SAME: PROPAGATE
func.func @max_unit_zero_stride(%arg0: !migraphx.shaped<1xf32, 0>, %arg1: !migraphx.shaped<1xf32, 0>) -> !migraphx.shaped<1xf32, 0> {
  %0 = migraphx.max %arg0, %arg1 : <1xf32, 0>, <1xf32, 0> -> <1xf32, 0>
  return %0 : !migraphx.shaped<1xf32, 0>
}

// A clip expands to a maximum/minimum pair; both halves take the same mode,
// which is what makes the pair foldable.

// TOSA: sym_name = "clip_f32"
// TOSA: "tosa.maximum"
// TOSA-SAME: IGNORE
// TOSA: "tosa.minimum"
// TOSA-SAME: IGNORE

// IEEE: sym_name = "clip_f32"
// IEEE: "tosa.maximum"
// IEEE-SAME: PROPAGATE
// IEEE: "tosa.minimum"
// IEEE-SAME: PROPAGATE
func.func @clip_f32(%arg0: !migraphx.shaped<2x4xf32, 4x1>, %lo: !migraphx.shaped<2x4xf32, 0x0>, %hi: !migraphx.shaped<2x4xf32, 0x0>) -> !migraphx.shaped<2x4xf32, 4x1> {
  %0 = migraphx.clip %arg0, %lo, %hi : <2x4xf32, 4x1>, <2x4xf32, 0x0>, <2x4xf32, 0x0> -> <2x4xf32, 4x1>
  return %0 : !migraphx.shaped<2x4xf32, 4x1>
}

// A relu is a maximum against zero and follows the same rule, though on its own
// it is a single op rather than a clamp pair, so there is nothing to fold.

// TOSA: sym_name = "relu_f32"
// TOSA: "tosa.maximum"
// TOSA-SAME: nan_mode
// TOSA-SAME: IGNORE

// IEEE: sym_name = "relu_f32"
// IEEE: "tosa.maximum"
// IEEE-SAME: nan_mode
// IEEE-SAME: PROPAGATE
func.func @relu_f32(%arg0: !migraphx.shaped<2x4xf32, 4x1>) -> !migraphx.shaped<2x4xf32, 4x1> {
  %0 = migraphx.relu %arg0 : <2x4xf32, 4x1> -> <2x4xf32, 4x1>
  return %0 : !migraphx.shaped<2x4xf32, 4x1>
}

// Quantizing to a float type saturates through a tosa.clamp rather than the
// integer path's clamped cast, and that clamp takes the mode too. It is a real
// pair once lowered, so this is the second place a v_med3 can come from.

// TOSA: sym_name = "quantize_fp8"
// TOSA: "tosa.clamp"
// TOSA-SAME: nan_mode
// TOSA-SAME: IGNORE

// IEEE: sym_name = "quantize_fp8"
// IEEE: "tosa.clamp"
// IEEE-SAME: nan_mode
// IEEE-SAME: PROPAGATE
func.func @quantize_fp8(%arg0: !migraphx.shaped<2x4xf32, 4x1>, %scale: !migraphx.shaped<2x4xf32, 4x1>, %bias: !migraphx.shaped<2x4xf8E4M3FNUZ, 4x1>) -> !migraphx.shaped<2x4xf8E4M3FNUZ, 4x1> {
  %0 = migraphx.quantizelinear %arg0, %scale, %bias : <2x4xf32, 4x1>, <2x4xf32, 4x1>, !migraphx.shaped<2x4xf8E4M3FNUZ, 4x1> -> <2x4xf8E4M3FNUZ, 4x1>
  return %0 : !migraphx.shaped<2x4xf8E4M3FNUZ, 4x1>
}

// Integer maximum has no NaN to reason about, so it takes neither mode and is
// unaffected by -disable-fast-math. Unsigned needs the custom op because TOSA
// has no unsigned maximum.

// TOSA: sym_name = "max_ui32_broadcast"
// TOSA: "tosa.custom"
// TOSA-SAME: operator_name = "unsigned_max"

// UNSIGNED-LABEL: func.func @max_ui32_broadcast
// UNSIGNED-NOT: arith.maxsi
// UNSIGNED: %[[BROADCAST:.+]] = linalg.generic
// UNSIGNED-SAME: ins(%{{.+}}, %{{.+}} : tensor<2x4xi32>, tensor<1x4xi32>)
// UNSIGNED: linalg.generic
// UNSIGNED-SAME: ins(%{{.+}}, %[[BROADCAST]] : tensor<2x4xi32>, tensor<2x4xi32>)
// UNSIGNED: arith.maxui
// UNSIGNED-NOT: arith.maxsi
func.func @max_ui32_broadcast(%arg0: !migraphx.shaped<2x4xui32, 4x1>, %arg1: !migraphx.shaped<2x4xui32, 0x1>) -> !migraphx.shaped<2x4xui32, 4x1> {
  %0 = migraphx.max %arg0, %arg1 : <2x4xui32, 4x1>, <2x4xui32, 0x1> -> <2x4xui32, 4x1>
  return %0 : !migraphx.shaped<2x4xui32, 4x1>
}
