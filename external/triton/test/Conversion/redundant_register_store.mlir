// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm | FileCheck %s

// The store and atomic conversions skip register indices that a layout maps to
// an element another register in the same thread already holds, which they
// detect via the "register" free-variable mask of getFreeVariableMasks(). A
// layout whose register basis is zero broadcasts that register, so only the
// canonical index is accessed.

// Control: a non-zero register basis addresses two distinct elements, so both
// registers are stored.
#plain = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: @distinct_registers_are_all_stored
  // CHECK-COUNT-2: st.global
  // CHECK-NOT: st.global
  tt.func @distinct_registers_are_all_stored(%ptr: tensor<64x!tt.ptr<f32>, #plain>, %val: tensor<64xf32, #plain>) {
    tt.store %ptr, %val : tensor<64x!tt.ptr<f32>, #plain>
    tt.return
  }
}

// -----

// A zero register basis makes both registers hold the same element, so the
// redundant one is not stored.
#bcast = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: @broadcast_register_store
  // CHECK-COUNT-1: st.global
  // CHECK-NOT: st.global
  tt.func @broadcast_register_store(%ptr: tensor<32x!tt.ptr<f32>, #bcast>, %val: tensor<32xf32, #bcast>) {
    tt.store %ptr, %val : tensor<32x!tt.ptr<f32>, #bcast>
    tt.return
  }
}

// -----

// Same deduplication for the atomic path: the redundant register reuses the
// result of the canonical one instead of issuing a second atomic.
#bcast = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: @broadcast_register_atomic_rmw
  // CHECK-COUNT-1: atom.global
  // CHECK-NOT: atom.global
  tt.func @broadcast_register_atomic_rmw(%ptr: tensor<32x!tt.ptr<f32>, #bcast>, %val: tensor<32xf32, #bcast>) {
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %ptr, %val : (tensor<32x!tt.ptr<f32>, #bcast>, tensor<32xf32, #bcast>) -> tensor<32xf32, #bcast>
    tt.return
  }
}

// -----

// A scalar store has no layout, so getFreeVariableMasks() falls back to
// getAllFreeVarMasks(), which marks every dimension redundant. The store must
// still happen exactly once, under a predicate that elects a single thread.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-LABEL: @scalar_store_is_predicated
  // CHECK: @${{[0-9]+}} st.global
  // CHECK-NOT: st.global
  tt.func @scalar_store_is_predicated(%ptr: !tt.ptr<f32>, %val: f32) {
    tt.store %ptr, %val : !tt.ptr<f32>
    tt.return
  }
}
