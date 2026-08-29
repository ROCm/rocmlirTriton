// Negative tests for rock-tosa-to-elementwise pass.
//
// Mixes plain op-error cases (rejected `tosa.cast` fp->int) and a fatal
// process-aborting case (createClampedFPToInt's source-type guard), so we
// use `not ... 2>&1 | FileCheck` for everything rather than
// -verify-diagnostics: the fatal-error path aborts the process and would
// not be observable through verify-diagnostics. Order matters -- the
// fatal-abort section is placed last so the earlier sections still run
// and emit their CHECK-able errors before the process terminates.

// RUN: not rocmlir-opt --rock-tosa-to-elementwise --split-input-file %s 2>&1 | FileCheck %s

// The MIGraphX pipeline is the only consumer of this pass and it must emit
// `tosa.custom "fp_to_int_cast"` for any float-to-int conversion (so that
// saturating-truncation semantics are preserved). A plain `tosa.cast`
// fp->int is therefore rejected -- it would otherwise force the pass to
// silently choose between TOSA-spec round-to-nearest-even and MIGraphX's
// truncation behaviour.

// CHECK: error: {{.*}}tosa.cast from floating-point to integer is not supported by rock-tosa-to-elementwise
func.func @cast_f32_to_i32_rejected(%arg0: tensor<16xf32>) -> tensor<16xi32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %0 = tosa.cast %arg0 : (tensor<16xf32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}

// -----

// CHECK: error: {{.*}}tosa.cast from floating-point to integer is not supported by rock-tosa-to-elementwise
func.func @cast_f16_to_i8_rejected(%arg0: tensor<16xf16>) -> tensor<16xi8>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %0 = tosa.cast %arg0 : (tensor<16xf16>) -> tensor<16xi8>
  return %0 : tensor<16xi8>
}

// -----

// `fp_to_int_cast` lowers via rock::createClampedFPToInt, which requires
// the source float type to have representable zero, representable
// infinity, AND a signed representation (so it can materialize -inf for
// the case-1 overflow check). Source types that violate any of these are
// rejected via llvm::reportFatalUsageError, which aborts the process.
//
// Today the only LLVM/MLIR float type without a sign bit is F8E8M0FNU,
// which also lacks zero, so this case short-circuits on the first
// sub-condition rather than the new signed-repr one. We still test it
// because the guard is the unit of behaviour; the signed-repr branch is
// added defensively against a future MX/OCP-style unsigned-with-infinity
// format.
// CHECK: rock::createClampedFPToInt: source float type lacks a representable zero, signed representation, or infinity
func.func @fp_to_int_cast_unsigned_float_rejected(%arg0: tensor<16xf8E8M0FNU>) -> tensor<16xi32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "fp_to_int_cast"} : (tensor<16xf8E8M0FNU>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
}
