// RUN: rocmlir-opt -rock-fold-oob-buffer-ops -split-input-file %s | FileCheck %s

// The Triton lowering predicates a store by pointing it at INT_MIN, which is
// past the descriptor's NumRecords. With `rock.block_size` bounding
// `rocdl.workitem.id.x`, the range analysis proves the predicate false, so the
// offset is exactly the sentinel and the store can go.

// CHECK-LABEL: llvm.func @erase_predicated_store
// CHECK-NOT: rocdl.raw.ptr.buffer.store
// CHECK: llvm.return
llvm.func @erase_predicated_store(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx1100", rock.block_size = 64 : i32,
                rock.grid_size = 16 : i32} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(805306368 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c128 = llvm.mlir.constant(128 : i32) : i32
  %sentinel = llvm.mlir.constant(-2147483648 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %tid = rocdl.workitem.id.x : i32
  // %tid is in [0, 64], so %tid / 128 is zero and the predicate is false.
  %hi = llvm.udiv %tid, %c128 : i32
  %pred = llvm.icmp "ne" %hi, %zero : i32
  %offset = llvm.select %pred, %tid, %sentinel : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}

// -----

// An offset the analysis proves in bounds must be left alone.

// CHECK-LABEL: llvm.func @keep_inbounds_store
// CHECK: rocdl.raw.ptr.buffer.store
llvm.func @keep_inbounds_store(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx1100", rock.block_size = 64 : i32,
                rock.grid_size = 16 : i32} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(805306368 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c63 = llvm.mlir.constant(63 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %tid = rocdl.workitem.id.x : i32
  %offset = llvm.and %tid, %c63 : i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}

// -----

// Without the launch geometry there is nothing to bound the thread id with, so
// the predicate stays unknown and the store has to stay.

// CHECK-LABEL: llvm.func @keep_store_without_launch_geometry
// CHECK: rocdl.raw.ptr.buffer.store
llvm.func @keep_store_without_launch_geometry(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(805306368 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c128 = llvm.mlir.constant(128 : i32) : i32
  %sentinel = llvm.mlir.constant(-2147483648 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %tid = rocdl.workitem.id.x : i32
  %hi = llvm.udiv %tid, %c128 : i32
  %pred = llvm.icmp "ne" %hi, %zero : i32
  %offset = llvm.select %pred, %tid, %sentinel : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}

// -----

// A non-zero stride makes NumRecords count structured records rather than
// bytes, so the byte offset is not what gets bounds-checked.

// CHECK-LABEL: llvm.func @keep_store_with_structured_descriptor
// CHECK: rocdl.raw.ptr.buffer.store
llvm.func @keep_store_with_structured_descriptor(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx1100", rock.block_size = 64 : i32,
                rock.grid_size = 16 : i32} {
  %stride = llvm.mlir.constant(4 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(805306368 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c128 = llvm.mlir.constant(128 : i32) : i32
  %sentinel = llvm.mlir.constant(-2147483648 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %tid = rocdl.workitem.id.x : i32
  %hi = llvm.udiv %tid, %c128 : i32
  %pred = llvm.icmp "ne" %hi, %zero : i32
  %offset = llvm.select %pred, %tid, %sentinel : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}

// -----

// OOB_SELECT == 0 clamps instead of discarding, so an out-of-range offset still
// writes somewhere and the store must stay.

// CHECK-LABEL: llvm.func @keep_store_with_clamping_descriptor
// CHECK: rocdl.raw.ptr.buffer.store
llvm.func @keep_store_with_clamping_descriptor(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx1100", rock.block_size = 64 : i32,
                rock.grid_size = 16 : i32} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(0 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c128 = llvm.mlir.constant(128 : i32) : i32
  %sentinel = llvm.mlir.constant(-2147483648 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %tid = rocdl.workitem.id.x : i32
  %hi = llvm.udiv %tid, %c128 : i32
  %pred = llvm.icmp "ne" %hi, %zero : i32
  %offset = llvm.select %pred, %tid, %sentinel : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}

// -----

// The workgroup id needs the same treatment: `rock.grid_size` bounds it, and a
// predicate that only follows from that bound must still fold.

// CHECK-LABEL: llvm.func @erase_predicated_store_on_workgroup_id
// CHECK-NOT: rocdl.raw.ptr.buffer.store
// CHECK: llvm.return
llvm.func @erase_predicated_store_on_workgroup_id(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx1100", rock.block_size = 64 : i32,
                rock.grid_size = 16 : i32} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(805306368 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c32 = llvm.mlir.constant(32 : i32) : i32
  %sentinel = llvm.mlir.constant(-2147483648 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %bid = rocdl.workgroup.id.x : i32
  // %bid is in [0, 16], so %bid / 32 is zero and the predicate is false.
  %hi = llvm.udiv %bid, %c32 : i32
  %pred = llvm.icmp "ne" %hi, %zero : i32
  %offset = llvm.select %pred, %bid, %sentinel : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}

// -----

// GFX9 does not have the OOB_SELECT field, so the same flags word means
// something else and the descriptor cannot be decoded.

// CHECK-LABEL: llvm.func @keep_store_on_gfx9
// CHECK: rocdl.raw.ptr.buffer.store
llvm.func @keep_store_on_gfx9(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx90a:sramecc+:xnack-", rock.block_size = 64 : i32,
                rock.grid_size = 16 : i32} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(805306368 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c128 = llvm.mlir.constant(128 : i32) : i32
  %sentinel = llvm.mlir.constant(-2147483648 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %tid = rocdl.workitem.id.x : i32
  %hi = llvm.udiv %tid, %c128 : i32
  %pred = llvm.icmp "ne" %hi, %zero : i32
  %offset = llvm.select %pred, %tid, %sentinel : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}

// -----

// GFX12.5 moved NumRecords out of the word this pass reads.

// CHECK-LABEL: llvm.func @keep_store_on_gfx1250
// CHECK: rocdl.raw.ptr.buffer.store
llvm.func @keep_store_on_gfx1250(%arg0: !llvm.ptr<1>)
    attributes {rock.arch = "gfx1250", rock.block_size = 64 : i32,
                rock.grid_size = 16 : i32} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(1024 : i64) : i64
  %flags = llvm.mlir.constant(805306368 : i32) : i32
  %zero = llvm.mlir.constant(0 : i32) : i32
  %c128 = llvm.mlir.constant(128 : i32) : i32
  %sentinel = llvm.mlir.constant(-2147483648 : i32) : i32
  %data = llvm.mlir.constant(7 : i8) : i8

  %rsrc = rocdl.make.buffer.rsrc %arg0, %stride, %numRecords, %flags : <1> to <8>
  %tid = rocdl.workitem.id.x : i32
  %hi = llvm.udiv %tid, %c128 : i32
  %pred = llvm.icmp "ne" %hi, %zero : i32
  %offset = llvm.select %pred, %tid, %sentinel : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %offset, %zero, %zero : i8
  llvm.return
}
