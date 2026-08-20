// RUN: rocmlir-opt --rock-fold-oob-buffer-ops --split-input-file %s | FileCheck %s

// Each case pins one of the conditions the fold needs: a statically false
// predicate, an offset the descriptor's numRecords puts out of bounds, a soffset
// that cannot wrap it back in, and a raw descriptor.
//
// The predicates come in two kinds. A magnitude fact, where `urem` by a small
// constant bounds the row below the comparison, and a per-bit fact, where `xor`
// by 2 sets a bit that is otherwise known clear. Only known bits proves the
// second, since as an interval the xor widens back over zero - and that is the
// shape Triton's LinearLayout index math emits for every distributed layout.

// CHECK-LABEL: llvm.func @plain_sentinel_magnitude_fact
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @plain_sentinel_magnitude_fact(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  // %row is in [0, 3], so it is never at or past 16.
  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @plain_sentinel_bit_fact
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @plain_sentinel_bit_fact(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  // Bit 1 of %base is clear, so %base ^ 2 always has it set and is never zero.
  // As an interval this is [0, 19], which contains zero.
  %tid = rocdl.workitem.id.x : i32
  %bit0 = llvm.and %tid, %one : i32
  %high = llvm.and %tid, %sixteen : i32
  %base = llvm.or disjoint %bit0, %high : i32
  %swizzled = llvm.xor %base, %two : i32
  %pred = llvm.icmp "ult" %swizzled, %one : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// The split-soffset shape: the emitter has lifted a uniform part of the offset
// into soffset, so both arms of the out-of-bounds select are out of bounds and
// the two selects share their condition.
// CHECK-LABEL: llvm.func @split_soffset_sentinel
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @split_soffset_sentinel(%ptr: !llvm.ptr<1>, %data: i32, %uniform: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %minusOne = llvm.mlir.constant(-1 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %nonNeg = llvm.icmp "sge" %uniform, %zero : i32
  %soffset = llvm.select %nonNeg, %uniform, %zero : i1, i32
  %negated = llvm.sub %minusOne, %uniform : i32
  %splitOob = llvm.select %nonNeg, %negated, %oob : i1, i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %splitOob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %soffset, %zero : i32
  llvm.return
}

// -----

// A discarded raw load reads as zero, so the load folds instead of being erased.
// CHECK-LABEL: llvm.func @load_folds_to_zero
// CHECK:         %[[ZERO:.*]] = llvm.mlir.zero : vector<4xf32>
// CHECK-NOT:     rocdl.raw.ptr.buffer.load
// CHECK:         llvm.return %[[ZERO]]
llvm.func @load_folds_to_zero(%ptr: !llvm.ptr<1>) -> vector<4xf32> attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  %loaded = rocdl.raw.ptr.buffer.load %rsrc, %voffset, %zero, %zero : vector<4xf32>
  llvm.return %loaded : vector<4xf32>
}

// -----

// The bound is whatever the descriptor was built with, so an offset far below
// the usual sentinel is still discarded by a small buffer.
// CHECK-LABEL: llvm.func @small_num_records_folds
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @small_num_records_folds(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(256 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(4096 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// The same reading the other way: a buffer larger than the 32-bit offset space
// leaves the emitter's own sentinel in bounds, so the fold declines.
// CHECK-LABEL: llvm.func @num_records_beyond_offset_space_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @num_records_beyond_offset_space_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(4294967296 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// A read-modify-write reaches LLVM as an llvm.call_intrinsic, since the ROCDL ops
// for it declare no result. With its returned value unread, a discarded one has
// nothing left to do and is erased like a store.
// CHECK-LABEL: llvm.func @unread_atomic_rmw_erased
// CHECK-NOT:     llvm.call_intrinsic
// CHECK:       llvm.return
llvm.func @unread_atomic_rmw_erased(%ptr: !llvm.ptr<1>, %data: f32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  %old = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"(%data, %rsrc, %voffset, %zero, %zero) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
  llvm.return
}

// -----

// The compare-and-swap is the one atomic spelled as a ROCDL op, and it takes two
// data operands ahead of the descriptor rather than one.
// CHECK-LABEL: llvm.func @unread_atomic_cmpswap_erased
// CHECK-NOT:     rocdl.raw.ptr.buffer.atomic.cmpswap
// CHECK:       llvm.return
llvm.func @unread_atomic_cmpswap_erased(%ptr: !llvm.ptr<1>, %val: i32, %cmp: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  %old = rocdl.raw.ptr.buffer.atomic.cmpswap %val, %cmp, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// Reading the returned value keeps the atomic: unlike a discarded load's zero,
// what a discarded atomic hands back is not a value the pass can fold to.
// CHECK-LABEL: llvm.func @read_atomic_declines
// CHECK:         llvm.call_intrinsic
llvm.func @read_atomic_declines(%ptr: !llvm.ptr<1>, %data: f32) -> f32 attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  %old = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"(%data, %rsrc, %voffset, %zero, %zero) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
  llvm.return %old : f32
}

// -----

// The predicate depends on a runtime value, so nothing is provably discarded.
// CHECK-LABEL: llvm.func @live_predicate_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @live_predicate_declines(%ptr: !llvm.ptr<1>, %data: i32, %bound: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %pred = llvm.icmp "uge" %tid, %bound : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// A non-zero stride is not a descriptor the buffer-op emitter builds, so the
// pass declines rather than assuming the emitter's out-of-bounds encoding.
// CHECK-LABEL: llvm.func @nonzero_stride_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @nonzero_stride_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(16 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// The false arm is in bounds, so the predicate being false says nothing about
// whether the hardware discards the store.
// CHECK-LABEL: llvm.func @wrong_sentinel_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @wrong_sentinel_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %inBounds = llvm.mlir.constant(128 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %inBounds : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// An unconstrained soffset could carry voffset + soffset past 2^32 and wrap it
// back to the start of the buffer. Only a zero soffset or the correlated pair
// rules that out, and this is neither.
// CHECK-LABEL: llvm.func @plain_sentinel_runtime_soffset_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @plain_sentinel_runtime_soffset_declines(%ptr: !llvm.ptr<1>, %data: i32,
                                                  %soffset: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %soffset, %zero : i32
  llvm.return
}

// -----

// The correlated pair without its guard. Nothing bounds %uniform, so the offset
// arm can be any value at all and the sum with soffset can wrap into the buffer.
// CHECK-LABEL: llvm.func @split_soffset_without_sge_guard_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @split_soffset_without_sge_guard_declines(%ptr: !llvm.ptr<1>, %data: i32,
                                                   %uniform: i32, %cond: i1) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %minusOne = llvm.mlir.constant(-1 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %soffset = llvm.select %cond, %uniform, %zero : i1, i32
  %negated = llvm.sub %minusOne, %uniform : i32
  %splitOob = llvm.select %cond, %negated, %oob : i1, i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %splitOob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %soffset, %zero : i32
  llvm.return
}

// -----

// The guard is there but the offset arm negates a different value than the one
// lifted into soffset, so the two no longer sum to all-ones.
// CHECK-LABEL: llvm.func @split_soffset_mismatched_negation_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @split_soffset_mismatched_negation_declines(%ptr: !llvm.ptr<1>, %data: i32,
                                                     %uniform: i32, %other: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %minusOne = llvm.mlir.constant(-1 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %nonNeg = llvm.icmp "sge" %uniform, %zero : i32
  %soffset = llvm.select %nonNeg, %uniform, %zero : i1, i32
  %negated = llvm.sub %minusOne, %other : i32
  %splitOob = llvm.select %nonNeg, %negated, %oob : i1, i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %splitOob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %soffset, %zero : i32
  llvm.return
}

// -----

// OOB_SELECT = 2 turns the bounds check off, so the hardware performs the access
// however far past numRecords the offset is.
// CHECK-LABEL: llvm.func @oob_select_none_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @oob_select_none_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(553807872 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// gfx9 has no OOB_SELECT field, so a zero-stride descriptor always checks the
// offset. These same flags on RDNA would read OOB_SELECT = 0 and decline, so
// the gate is per-target and not per-value.
// CHECK-LABEL: llvm.func @cdna_flags_fold
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @cdna_flags_fold(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx942"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(159744 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// TID_ENABLE makes the bound wave-relative, so numRecords no longer says where
// this lane's access lands.
// CHECK-LABEL: llvm.func @tid_enable_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @tid_enable_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx942"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(8548352 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// A discarded async load to LDS leaves its slot holding whatever was there
// before, which is not the zero a discarded register load reads, so it stays.
// CHECK-LABEL: llvm.func @load_async_lds_survives
// CHECK:         rocdl.raw.ptr.buffer.load.async.lds
llvm.func @load_async_lds_survives(%ptr: !llvm.ptr<1>, %lds: !llvm.ptr<3>) attributes {rock.arch = "gfx942"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(159744 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.load.async.lds %rsrc, %lds, %four, %voffset, %zero, %zero, %zero
  llvm.return
}
