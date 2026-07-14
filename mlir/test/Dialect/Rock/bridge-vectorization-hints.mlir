// expected-error@-1 {{rock-bridge-vectorization-hints: unknown phase 'bogus' (expected 'stash' or 'apply')}}

// Verifies the two phases of rock-bridge-vectorization-hints, which carry the
// im2col vectorization hint across tritonamdgpu-canonicalize-pointers without a
// Triton submodule patch.

// RUN: rocmlir-opt -rock-bridge-vectorization-hints=phase=stash -split-input-file %s | FileCheck %s --check-prefix=STASH
// RUN: rocmlir-opt -rock-bridge-vectorization-hints=phase=apply -split-input-file %s | FileCheck %s --check-prefix=APPLY

// Negative: any phase other than 'stash'/'apply' must fail the pass. The error
// is emitted on the top-level module (reported at line 0), so the expected-error
// directive is pinned to line 1 above and references it with @-1. This run
// omits -split-input-file so the file is one module and the error fires once.
// RUN: rocmlir-opt -rock-bridge-vectorization-hints=phase=bogus -verify-diagnostics %s

// Stash: the hint on the tt.addptr is copied onto the consuming tt.load under
// Rock-private attr names (which survive canonicalize-pointers), while the
// original attrs on the addptr are left untouched (they get dropped later).
// STASH-LABEL: tt.func @stash_load
//       STASH:   tt.addptr
//  STASH-SAME:     tt.contiguity = dense<[1, 4]>
//       STASH:   tt.load
//  STASH-SAME:     rock.vec_contiguity = dense<[1, 4]> : tensor<2xi32>
//  STASH-SAME:     rock.vec_divisibility = dense<[4, 16]> : tensor<2xi32>
tt.func @stash_load(%arg0: !tt.ptr<f32>) {
  %splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %off = arith.constant dense<0> : tensor<4x8xi32>
  %ptr = tt.addptr %splat, %off {tt.contiguity = dense<[1, 4]> : tensor<2xi32>, tt.divisibility = dense<[4, 16]> : tensor<2xi32>} : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %v = tt.load %ptr : tensor<4x8x!tt.ptr<f32>>
  tt.return
}

// -----

// Apply: the Rock-private marker parked on the tt.load is re-stamped as
// tt.contiguity / tt.divisibility onto the (rebuilt) tt.addptr feeding it, and
// the markers are erased from the load. This mirrors the op that
// convert-buffer-ops resolves via ptr.getDefiningOp<AddPtrOp>().
// APPLY-LABEL: tt.func @apply_load
//       APPLY:   tt.addptr
//  APPLY-SAME:     tt.contiguity = dense<[1, 4]> : tensor<2xi32>
//  APPLY-SAME:     tt.divisibility = dense<[4, 16]> : tensor<2xi32>
//       APPLY:   tt.load
//   APPLY-NOT:     rock.vec_contiguity
//   APPLY-NOT:     rock.vec_divisibility
tt.func @apply_load(%arg0: !tt.ptr<f32>) {
  %splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %off = arith.constant dense<0> : tensor<4x8xi32>
  %ptr = tt.addptr %splat, %off : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %v = tt.load %ptr {rock.vec_contiguity = dense<[1, 4]> : tensor<2xi32>, rock.vec_divisibility = dense<[4, 16]> : tensor<2xi32>} : tensor<4x8x!tt.ptr<f32>>
  tt.return
}

// -----

// Apply cleanup when the pointer is NOT an addptr: if canonicalize-pointers
// folded the pointer arithmetic away (e.g. a zero offset), there is no addptr
// to re-stamp -- but the apply phase must still erase the Rock-private markers
// so they do not leak into the final Triton IR.
// APPLY-LABEL: tt.func @apply_no_addptr
//       APPLY:   tt.load
//   APPLY-NOT:     rock.vec_contiguity
//   APPLY-NOT:     rock.vec_divisibility
tt.func @apply_no_addptr(%arg0: !tt.ptr<f32>) {
  %splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %v = tt.load %splat {rock.vec_contiguity = dense<[1, 4]> : tensor<2xi32>, rock.vec_divisibility = dense<[4, 16]> : tensor<2xi32>} : tensor<4x8x!tt.ptr<f32>>
  tt.return
}
