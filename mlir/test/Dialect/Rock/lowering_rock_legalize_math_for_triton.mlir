// RUN: rocmlir-opt -rock-legalize-math-for-triton -canonicalize -cse -mlir-print-local-scope %s | FileCheck %s
// RUN: rocmlir-opt -rock-legalize-math-for-triton='disable-fast-math=true' -canonicalize -cse -mlir-print-local-scope %s | FileCheck %s --check-prefix=NOFAST

// `disable-fast-math` is the only thing that licenses an approximation; the ops
// go out without fast-math flags because rock-allow-fast-math-flags runs later
// in the pipeline and annotates all of them, including the `arcp` that turns
// the divide below into a v_rcp_f32.
//
// The pass runs ahead of math-extend-to-supported-types, so the narrow types
// below arrive as themselves rather than already promoted to f32. That is what
// lets tanh treat them differently from an f32 the user actually asked for.

// ============================================================
// tanh, f32: left to the math dialect's own expansion, recognisable by the
// compare and the plain math.exp. The approximation is accurate enough only
// where the result is rounded to a narrower type afterwards, and nothing
// rounds an f32.
// ============================================================

// CHECK-LABEL: func.func @tanh_f32
// CHECK: arith.cmpf olt
// CHECK: math.exp %
// CHECK: arith.divf
// CHECK-NOT: math.exp2
// CHECK-NOT: tt.extern_elementwise

// NOFAST-LABEL: func.func @tanh_f32
// NOFAST: tt.extern_elementwise %{{.*}} {libname = "", libpath = "", pure = true, symbol = "__ocml_tanh_f32"} : (tensor<64x64xf32>) -> tensor<64x64xf32>
// NOFAST-NOT: math.exp
func.func @tanh_f32(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// ============================================================
// tanh on a target with v_tanh_f32 (gfx1250): the OCML call lowers to the
// single hardware instruction, so it beats both the expansion and the
// approximation and wins even under fast math.
// ============================================================

// CHECK-LABEL: func.func @tanh_f32_gfx1250
// CHECK: tt.extern_elementwise %{{.*}} {libname = "", libpath = "", pure = true, symbol = "__ocml_tanh_f32"} : (tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: math.exp
func.func @tanh_f32_gfx1250(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.arch = "gfx1250", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// ============================================================
// tanh, f16: the inline exp2 approximation.
// tanh(x) = 2 / (1 + exp2(-2*log2(e)*x)) - 1, five VALU ops.
//
// The identity covers the whole range, so there is no |x| and no sign restore.
// Large negative x overflows the exp2 to +inf and rcp(+inf) is exactly +0,
// leaving the fma on exactly -1. The absence of math.absf/arith.cmpf/
// arith.select below is the point of the test, not an omission.
//
// It is evaluated at f32 and rounded back, not at f16. The f32 scale constant
// (-2.885390e+00, where an f16 one would round to -2.884770e+00) is what pins
// that. Rounding the result to f16 afterwards is what makes an approximation
// admissible here at all: it lands within 1 ULP that way against 2558 ULP
// evaluated at f16. tanh-f16-isa.mlir pins the instruction count.
// ============================================================

// CHECK-LABEL: func.func @tanh_f16
// CHECK-DAG: %[[NEG_ONE:.*]] = arith.constant dense<-1.000000e+00> : tensor<64x64xf32>
// CHECK-DAG: %[[TWO:.*]] = arith.constant dense<2.000000e+00> : tensor<64x64xf32>
// CHECK-DAG: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
// CHECK-DAG: %[[SCALE:.*]] = arith.constant dense<-2.885390e+00> : tensor<64x64xf32>
// CHECK: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64x64xf16> to tensor<64x64xf32>
// CHECK: %[[SCALED:.*]] = arith.mulf %[[EXT]], %[[SCALE]]
// CHECK: %[[EXP:.*]] = math.exp2 %[[SCALED]] : tensor<64x64xf32>
// CHECK: %[[DENOM:.*]] = arith.addf %[[EXP]], %[[ONE]]
// CHECK: %[[RECIP:.*]] = arith.divf %[[ONE]], %[[DENOM]]
// CHECK: %[[FMA:.*]] = math.fma %[[TWO]], %[[RECIP]], %[[NEG_ONE]]
// CHECK: %[[RES:.*]] = arith.truncf %[[FMA]] : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK: return %[[RES]]
// CHECK-NOT: tt.extern_elementwise

// The OCML path widens the same way, libdevice declaring only f32 and f64.
// NOFAST-LABEL: func.func @tanh_f16
// NOFAST: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64x64xf16> to tensor<64x64xf32>
// NOFAST: %[[CALL:.*]] = tt.extern_elementwise %[[EXT]] {{.*}} symbol = "__ocml_tanh_f32"
// NOFAST: arith.truncf %[[CALL]] : tensor<64x64xf32> to tensor<64x64xf16>
func.func @tanh_f16(%arg0: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// ============================================================
// tanh, bf16: the same route as f16. Eight significand bits are fewer than the
// f32 arithmetic is out by, so the rounded result is correctly rounded almost
// everywhere; see the table in emitApproxTanh.
// ============================================================

// CHECK-LABEL: func.func @tanh_bf16
// CHECK: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<64x64xf32>
// CHECK: %[[RES:.*]] = arith.truncf %{{.*}} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK: return %[[RES]]
// CHECK-NOT: tt.extern_elementwise

// NOFAST-LABEL: func.func @tanh_bf16
// NOFAST: tt.extern_elementwise %{{.*}} symbol = "__ocml_tanh_f32"
// NOFAST-NOT: math.exp2
func.func @tanh_bf16(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}

// ============================================================
// Scalars are handled too. The old TOSA-level lowering only rewrote shaped
// types and left scalar tanh to become an unlinkable tanhf libcall.
// ============================================================

// CHECK-LABEL: func.func @tanh_scalar_f16
// CHECK: math.exp2 %{{.*}} : f32
// CHECK: arith.truncf %{{.*}} : f32 to f16
// CHECK-NOT: math.tanh

// NOFAST-LABEL: func.func @tanh_scalar_f16
// NOFAST: tt.extern_elementwise %{{.*}} symbol = "__ocml_tanh_f32"} : (f32) -> f32
// NOFAST-NOT: math.tanh
func.func @tanh_scalar_f16(%arg0: f16) -> f16 attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : f16
  return %0 : f16
}

// ============================================================
// Every float width is handled, not only the ones an approximation suits.
// Nothing may reach rock-to-ttir as a math.tanh, which Triton has no conversion
// for. f64 has its own libdevice entry point and keeps its precision.
// ============================================================

// CHECK-LABEL: func.func @tanh_f64
// CHECK: tt.extern_elementwise %{{.*}} symbol = "__ocml_tanh_f64"} : (tensor<64xf64>) -> tensor<64xf64>
// CHECK-NOT: math.tanh
func.func @tanh_f64(%arg0: tensor<64xf64>) -> tensor<64xf64> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf64>
  return %0 : tensor<64xf64>
}

// Wider than f64 there is no libdevice entry point, so the OCML rewrite
// declines and the generic expansion picks the op up instead. No AMD target has
// f80 arithmetic, so this only matters as a guarantee that an exotic width
// leaves the pass as something Triton can carry rather than as a call to a
// symbol that does not exist. Both fast-math settings take this route: with
// fast math the width is above the approximation's range, without it the OCML
// rewrite runs first and declines on the same check.

// CHECK-LABEL: func.func @tanh_f80
// CHECK: arith.cmpf olt
// CHECK: math.exp %
// CHECK: arith.divf
// CHECK-NOT: tt.extern_elementwise

// NOFAST-LABEL: func.func @tanh_f80
// NOFAST: math.exp %
// NOFAST-NOT: tt.extern_elementwise
func.func @tanh_f80(%arg0: tensor<64xf80>) -> tensor<64xf80> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf80>
  return %0 : tensor<64xf80>
}

// CHECK-LABEL: func.func @powf_f80
// CHECK: math.log
// CHECK: math.exp %
// CHECK-NOT: tt.extern_elementwise
func.func @powf_f80(%arg0: tensor<64xf80>, %arg1: tensor<64xf80>) -> tensor<64xf80> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.powf %arg0, %arg1 : tensor<64xf80>
  return %0 : tensor<64xf80>
}

// The sub-f16 formats all take the same route as f16, whichever exponent and
// mantissa split and whichever NaN convention they use: widen to f32,
// approximate, round back. Their resolution is coarser than f16's, so the f32
// arithmetic has even more room to spare. Under `disable-fast-math` they widen
// around the OCML call instead.

// CHECK-LABEL: func.func @tanh_f8E4M3FN
// CHECK: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64xf8E4M3FN> to tensor<64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64xf32> to tensor<64xf8E4M3FN>
// CHECK-NOT: math.tanh

// NOFAST-LABEL: func.func @tanh_f8E4M3FN
// NOFAST: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64xf8E4M3FN> to tensor<64xf32>
// NOFAST: %[[CALL:.*]] = tt.extern_elementwise %[[EXT]] {{.*}} symbol = "__ocml_tanh_f32"
// NOFAST: arith.truncf %[[CALL]] : tensor<64xf32> to tensor<64xf8E4M3FN>
func.func @tanh_f8E4M3FN(%arg0: tensor<64xf8E4M3FN>) -> tensor<64xf8E4M3FN> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf8E4M3FN>
  return %0 : tensor<64xf8E4M3FN>
}

// CHECK-LABEL: func.func @tanh_f8E5M2
// CHECK: arith.extf %{{.*}} : tensor<64xf8E5M2> to tensor<64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64xf32> to tensor<64xf8E5M2>
// CHECK-NOT: math.tanh
func.func @tanh_f8E5M2(%arg0: tensor<64xf8E5M2>) -> tensor<64xf8E5M2> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf8E5M2>
  return %0 : tensor<64xf8E5M2>
}

// CHECK-LABEL: func.func @tanh_f8E4M3FNUZ
// CHECK: arith.extf %{{.*}} : tensor<64xf8E4M3FNUZ> to tensor<64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64xf32> to tensor<64xf8E4M3FNUZ>
// CHECK-NOT: math.tanh
func.func @tanh_f8E4M3FNUZ(%arg0: tensor<64xf8E4M3FNUZ>) -> tensor<64xf8E4M3FNUZ> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf8E4M3FNUZ>
  return %0 : tensor<64xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @tanh_f8E5M2FNUZ
// CHECK: arith.extf %{{.*}} : tensor<64xf8E5M2FNUZ> to tensor<64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64xf32> to tensor<64xf8E5M2FNUZ>
// CHECK-NOT: math.tanh
func.func @tanh_f8E5M2FNUZ(%arg0: tensor<64xf8E5M2FNUZ>) -> tensor<64xf8E5M2FNUZ> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf8E5M2FNUZ>
  return %0 : tensor<64xf8E5M2FNUZ>
}

// f8E8M0FNU has no mantissa bits at all, so the widen/narrow pair is the only
// thing that can carry a tanh; the arith-expand step earlier in the pipeline
// handles the casts themselves.
// CHECK-LABEL: func.func @tanh_f8E8M0FNU
// CHECK: arith.extf %{{.*}} : tensor<64xf8E8M0FNU> to tensor<64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64xf32> to tensor<64xf8E8M0FNU>
// CHECK-NOT: math.tanh
func.func @tanh_f8E8M0FNU(%arg0: tensor<64xf8E8M0FNU>) -> tensor<64xf8E8M0FNU> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf8E8M0FNU>
  return %0 : tensor<64xf8E8M0FNU>
}

// CHECK-LABEL: func.func @tanh_f4E2M1FN
// CHECK: arith.extf %{{.*}} : tensor<64xf4E2M1FN> to tensor<64xf32>
// CHECK: math.exp2 %{{.*}} : tensor<64xf32>
// CHECK: arith.truncf %{{.*}} : tensor<64xf32> to tensor<64xf4E2M1FN>
// CHECK-NOT: math.tanh
func.func @tanh_f4E2M1FN(%arg0: tensor<64xf4E2M1FN>) -> tensor<64xf4E2M1FN> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf4E2M1FN>
  return %0 : tensor<64xf4E2M1FN>
}

// A native v_tanh_f32 wins for the narrow types too: there is no f8 tanh
// instruction, so the widening has to happen either way and the OCML call the
// backend rewrites into the hardware one costs nothing extra.
// CHECK-LABEL: func.func @tanh_f8_gfx1250
// CHECK: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64xf8E4M3FN> to tensor<64xf32>
// CHECK: %[[CALL:.*]] = tt.extern_elementwise %[[EXT]] {{.*}} symbol = "__ocml_tanh_f32"
// CHECK: arith.truncf %[[CALL]] : tensor<64xf32> to tensor<64xf8E4M3FN>
// CHECK-NOT: math.exp2
func.func @tanh_f8_gfx1250(%arg0: tensor<64xf8E4M3FN>) -> tensor<64xf8E4M3FN> attributes {rock.arch = "gfx1250", rock.kernel} {
  %0 = math.tanh %arg0 : tensor<64xf8E4M3FN>
  return %0 : tensor<64xf8E4M3FN>
}

// ============================================================
// powf is always the library call, fast math or not. Deep learning barely uses
// it, so the cheaper paths this used to take, folding small integral exponents
// into products and handing the rest to the exp/log expansion, were special
// cases earning their complexity on nothing.
// ============================================================

// CHECK-LABEL: func.func @powf_dynamic
// CHECK: tt.extern_elementwise %{{.*}}, %{{.*}} {libname = "", libpath = "", pure = true, symbol = "__ocml_pow_f32"} : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK-NOT: math.log

// NOFAST-LABEL: func.func @powf_dynamic
// NOFAST: tt.extern_elementwise %{{.*}} symbol = "__ocml_pow_f32"
// NOFAST-NOT: math.log
func.func @powf_dynamic(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.powf %arg0, %arg1 : tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// A constant exponent is no longer folded into a multiply. Leaving x^2 as a
// call looks wasteful in isolation, but powf is rare enough that one lowering
// with no cases in it is worth more than the instructions.
// CHECK-LABEL: func.func @powf_const_2
// CHECK: tt.extern_elementwise %{{.*}} symbol = "__ocml_pow_f32"
// CHECK-NOT: arith.mulf
func.func @powf_const_2(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.arch = "gfx942", rock.kernel} {
  %cst = arith.constant dense<2.0> : tensor<64x64xf32>
  %0 = math.powf %arg0, %cst : tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// powf spans the same widths as tanh. f64 keeps its precision.
// CHECK-LABEL: func.func @powf_f64
// CHECK: tt.extern_elementwise %{{.*}} symbol = "__ocml_pow_f64"} : (tensor<64xf64>, tensor<64xf64>) -> tensor<64xf64>
// CHECK-NOT: math.powf
func.func @powf_f64(%arg0: tensor<64xf64>, %arg1: tensor<64xf64>) -> tensor<64xf64> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.powf %arg0, %arg1 : tensor<64xf64>
  return %0 : tensor<64xf64>
}

// Both operands are widened around the call, not just the base.
// CHECK-LABEL: func.func @powf_f16
// CHECK: %[[EXT_X:.*]] = arith.extf %{{.*}} : tensor<64x64xf16> to tensor<64x64xf32>
// CHECK: %[[EXT_Y:.*]] = arith.extf %{{.*}} : tensor<64x64xf16> to tensor<64x64xf32>
// CHECK: %[[CALL:.*]] = tt.extern_elementwise %[[EXT_X]], %[[EXT_Y]] {{.*}} symbol = "__ocml_pow_f32"
// CHECK: %[[RES:.*]] = arith.truncf %[[CALL]] : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK: return %[[RES]]
func.func @powf_f16(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.powf %arg0, %arg1 : tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// CHECK-LABEL: func.func @powf_f8E4M3FN
// CHECK: %[[EXT_X:.*]] = arith.extf %{{.*}} : tensor<64xf8E4M3FN> to tensor<64xf32>
// CHECK: %[[EXT_Y:.*]] = arith.extf %{{.*}} : tensor<64xf8E4M3FN> to tensor<64xf32>
// CHECK: %[[CALL:.*]] = tt.extern_elementwise %[[EXT_X]], %[[EXT_Y]] {{.*}} symbol = "__ocml_pow_f32"
// CHECK: arith.truncf %[[CALL]] : tensor<64xf32> to tensor<64xf8E4M3FN>
// CHECK-NOT: math.log
func.func @powf_f8E4M3FN(%arg0: tensor<64xf8E4M3FN>, %arg1: tensor<64xf8E4M3FN>) -> tensor<64xf8E4M3FN> attributes {rock.arch = "gfx942", rock.kernel} {
  %0 = math.powf %arg0, %arg1 : tensor<64xf8E4M3FN>
  return %0 : tensor<64xf8E4M3FN>
}

// ============================================================
// Not a kernel: left alone. This pass runs before
// rock-serialize-host-funcs, so the host functions still reach it, and they
// are not lowered for the GPU. Skipping on the missing rock.kernel rather
// than on the missing rock.arch is what keeps the arch diagnostic in
// lowering_rock_legalize_math_for_triton_error.mlir meaningful for kernels.
// ============================================================

// CHECK-LABEL: func.func @tanh_host_func
// CHECK: math.tanh
// CHECK-NOT: math.exp2
// CHECK-NOT: tt.extern_elementwise

// NOFAST-LABEL: func.func @tanh_host_func
// NOFAST: math.tanh
// NOFAST-NOT: tt.extern_elementwise
func.func @tanh_host_func(%arg0: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "gfx942"} {
  %0 = math.tanh %arg0 : tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
