// RUN: rocmlir-opt --rock-fold-oob-buffer-ops --split-input-file %s | FileCheck %s

// Each case pins one of the conditions the fold needs: a statically false
// predicate, an offset the descriptor's numRecords puts out of bounds, a soffset
// that cannot wrap it back in, and a raw descriptor.
//
// The predicates come in two kinds, one per domain the oracle reads. A
// magnitude fact, where `urem` by ten bounds the row below the comparison, and a
// per-bit fact, where `xor` by 2 sets a bit that is otherwise known clear. Only
// the interval proves the first, since ten is not a power of two and the bits
// allow up to 15; only known bits proves the second, since as an interval the
// xor widens back over zero - and that is the shape Triton's LinearLayout index
// math emits for every distributed layout.

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
  %ten = llvm.mlir.constant(10 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  // %row is in [0, 9], so it is never at or past 10.
  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %ten : i32
  %pred = llvm.icmp "uge" %row, %ten : i32
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

// The two facts above again, each reaching the comparison through a lane read,
// which hands a value to another lane without changing it. One case per domain,
// since a lane read that dropped its operand's fact would decline instead.

// CHECK-LABEL: llvm.func @readfirstlane_keeps_magnitude_fact
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @readfirstlane_keeps_magnitude_fact(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %ten = llvm.mlir.constant(10 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %ten : i32
  %uniform = rocdl.readfirstlane %row : i32
  %pred = llvm.icmp "uge" %uniform, %ten : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @readfirstlane_keeps_bit_fact
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @readfirstlane_keeps_bit_fact(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
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

  %tid = rocdl.workitem.id.x : i32
  %bit0 = llvm.and %tid, %one : i32
  %high = llvm.and %tid, %sixteen : i32
  %base = llvm.or disjoint %bit0, %high : i32
  %swizzled = llvm.xor %base, %two : i32
  %uniform = rocdl.readfirstlane %swizzled : i32
  %pred = llvm.icmp "ult" %uniform, %one : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// The other lane read, carrying the descriptor's bound. Its i32 lane index is
// narrower than the i64 value, so an operand whose width is not the result's has
// to stay allowed on this path.
// CHECK-LABEL: llvm.func @readlane_keeps_the_descriptor_bound
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @readlane_keeps_the_descriptor_bound(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %lane = llvm.mlir.constant(0 : i32) : i32
  %n = llvm.mlir.constant(2147483646 : i64) : i64
  %numRecords = rocdl.readlane %n, %lane : (i64, i32) -> i64
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

// The id registers are bounded by the launch rather than by arithmetic on them,
// and the launch nests five counts: lanes fill a wave, waves fill a workgroup,
// workgroups fill a cluster, and a cluster is what one Triton program is. Each
// of the next cases pins one of them, since a bound short of the launch erases
// work that runs.
//
// A workitem id spans the warps a kernel launches with times the lanes in one.
// CHECK-LABEL: llvm.func @warps_times_lanes_bound_workitem_id
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @warps_times_lanes_bound_workitem_id(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %blockSize = llvm.mlir.constant(128 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    // %tid is in [0, 127], so it is never at or past the block size.
    %tid = rocdl.workitem.id.x : i32
    %pred = llvm.icmp "uge" %tid, %blockSize : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// Warp specialization has widened the block to eight warps and recorded that in
// ttg.total-num-warps, which is what the kernel launches with. Ids through 255
// run and the upper half take the offset arm, so reading ttg.num-warps instead
// would erase this live store.
// CHECK-LABEL: llvm.func @total_num_warps_widens_the_block
// CHECK:         rocdl.raw.ptr.buffer.store
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32,
                   "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @total_num_warps_widens_the_block(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %half = llvm.mlir.constant(128 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %tid = rocdl.workitem.id.x : i32
    %pred = llvm.icmp "uge" %tid, %half : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// A wave id counts waves in the workgroup, so the warp count bounds it on its
// own. This is the register Triton reads on the targets that have one, in place
// of dividing the workitem id by the warp size, so leaving it unbounded would
// lose there what the division gives everywhere else.
// CHECK-LABEL: llvm.func @warps_bound_wave_id
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @warps_bound_wave_id(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250"} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %warps = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    // %wave is in [0, 3], so it is never at or past the warp count.
    %wave = rocdl.wave.id : i32
    %pred = llvm.icmp "uge" %wave, %warps : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// The wave id is the absolute one, so warp specialization widens its range too:
// waves four through seven run and take the offset arm. Bounding it by the base
// ttg.num-warps would erase this live store.
// CHECK-LABEL: llvm.func @total_num_warps_widens_the_waves
// CHECK:         rocdl.raw.ptr.buffer.store
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32,
                   "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @total_num_warps_widens_the_waves(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250"} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %base = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %wave = rocdl.wave.id : i32
    %pred = llvm.icmp "uge" %wave, %base : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// A range attribute on the register, which the lowering writes when it knows a
// narrower bound than the launch does. Eight warps launch, so only the
// attribute puts this comparison out of reach.
// CHECK-LABEL: llvm.func @declared_range_narrows_the_wave_id
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @declared_range_narrows_the_wave_id(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250"} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %two = llvm.mlir.constant(2 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %wave = rocdl.wave.id range <i32, 0, 2> : i32
    %pred = llvm.icmp "uge" %wave, %two : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// The same wave id on a target that has no register for it, which is how every
// target but RDNA4 and gfx1250 gets one: the workitem id divided by the lanes in
// a wave, moved to a scalar register by a lane read. Four warps launch, so the
// division lands in [0, 3], and the lane read has to carry that bound through
// for the comparison to be out of reach. Losing it here would put back the
// predication the register case above folds away.
// CHECK-LABEL: llvm.func @warps_bound_the_divided_wave_id
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @warps_bound_the_divided_wave_id(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %lanes = llvm.mlir.constant(32 : i32) : i32
    %warps = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %tid = rocdl.workitem.id.x : i32
    %wave = llvm.udiv %tid, %lanes : i32
    %uniform = rocdl.readfirstlane %wave : i32
    %pred = llvm.icmp "uge" %uniform, %warps : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// No warp metadata to read. The lanes per warp especially cannot be assumed,
// since a wave64 target would run twice the ids a default of 32 implies.
// CHECK-LABEL: llvm.func @missing_warp_metadata_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @missing_warp_metadata_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %blockSize = llvm.mlir.constant(128 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %pred = llvm.icmp "uge" %tid, %blockSize : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// A workgroup id spans the flat grid, which is every cluster's workgroups: four
// programs of two workgroups each here.
// CHECK-LABEL: llvm.func @programs_times_cluster_bound_workgroup_id
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
module attributes {"ttg.num-ctas" = 2 : i32,
                   rock.grid_size.programs_times_cluster_bound_workgroup_id = 4 : i32} {
  llvm.func @programs_times_cluster_bound_workgroup_id(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250", rock.grid_size = 4 : i32} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(268435456 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %gridSize = llvm.mlir.constant(8 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %wg = rocdl.workgroup.id.x : i32
    %pred = llvm.icmp "uge" %wg, %gridSize : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// The same launch compared against the program count alone. Workgroups 4 to 7
// belong to the later clusters and run, so dropping the cluster factor would
// erase this live store.
// CHECK-LABEL: llvm.func @cluster_widens_the_workgroup_grid
// CHECK:         rocdl.raw.ptr.buffer.store
module attributes {"ttg.num-ctas" = 2 : i32,
                   rock.grid_size.cluster_widens_the_workgroup_grid = 4 : i32} {
  llvm.func @cluster_widens_the_workgroup_grid(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250", rock.grid_size = 4 : i32} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(268435456 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %programs = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %wg = rocdl.workgroup.id.x : i32
    %pred = llvm.icmp "uge" %wg, %programs : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// The cluster id is the other way around: it is the program id once clusters are
// in play, one per program, so the cluster factor does not enter its bound.
// CHECK-LABEL: llvm.func @cluster_id_counts_programs
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
module attributes {"ttg.num-ctas" = 2 : i32,
                   rock.grid_size.cluster_id_counts_programs = 4 : i32} {
  llvm.func @cluster_id_counts_programs(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250", rock.grid_size = 4 : i32} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(268435456 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %programs = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %cluster = rocdl.cluster.id.x : i32
    %pred = llvm.icmp "uge" %cluster, %programs : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// A workgroup's rank inside its own cluster, which the cluster size alone
// bounds. This is the register Triton reads for the CTA id.
// CHECK-LABEL: llvm.func @cluster_workgroup_id_counts_ctas
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
module attributes {"ttg.num-ctas" = 2 : i32} {
  llvm.func @cluster_workgroup_id_counts_ctas(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250", rock.grid_size = 4 : i32} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(268435456 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %ctas = llvm.mlir.constant(2 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %rank = rocdl.cluster.workgroup.id.x : i32
    %pred = llvm.icmp "uge" %rank, %ctas : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// The cluster size is Triton's to state. Without it there is no way to tell how
// many workgroups a program covers.
// CHECK-LABEL: llvm.func @missing_num_ctas_declines
// CHECK:         rocdl.raw.ptr.buffer.store
module attributes {rock.grid_size.missing_num_ctas_declines = 4 : i32} {
  llvm.func @missing_num_ctas_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100", rock.grid_size = 4 : i32} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %programs = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %wg = rocdl.workgroup.id.x : i32
    %pred = llvm.icmp "uge" %wg, %programs : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// The launch reads the module-level per-kernel rock.grid_size, so the stale copy
// the func carries does not get to shrink the bound. Trusting the func's 4 here
// would prove the predicate false and erase a store that workgroups 4 to 7 run.
// CHECK-LABEL: llvm.func @module_grid_size_wins_over_stale_func_copy
// CHECK:         rocdl.raw.ptr.buffer.store
module attributes {"ttg.num-ctas" = 1 : i32,
                   rock.grid_size.module_grid_size_wins_over_stale_func_copy = 8 : i32} {
  llvm.func @module_grid_size_wins_over_stale_func_copy(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100", rock.grid_size = 4 : i32} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %funcGrid = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %wg = rocdl.workgroup.id.x : i32
    %pred = llvm.icmp "uge" %wg, %funcGrid : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
}

// -----

// Without the module attribute there is no program count to read, and the func
// copy is not the launch's, so the bound is unknown and the access stays.
// CHECK-LABEL: llvm.func @missing_module_grid_size_declines
// CHECK:         rocdl.raw.ptr.buffer.store
module attributes {"ttg.num-ctas" = 1 : i32} {
  llvm.func @missing_module_grid_size_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100", rock.grid_size = 4 : i32} {
    %stride = llvm.mlir.constant(0 : i16) : i16
    %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
    %flags = llvm.mlir.constant(822243328 : i32) : i32
    %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

    %zero = llvm.mlir.constant(0 : i32) : i32
    %programs = llvm.mlir.constant(4 : i32) : i32
    %oob = llvm.mlir.constant(-2147483648 : i32) : i32
    %offset = llvm.mlir.constant(64 : i32) : i32

    %wg = rocdl.workgroup.id.x : i32
    %pred = llvm.icmp "uge" %wg, %programs : i32
    %voffset = llvm.select %pred, %offset, %oob : i1, i32
    rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
    llvm.return
  }
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
// CHECK-NOT:     rocdl.raw.ptr.buffer.load
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

// The ROCDL ops that declare no result, whose returned value is therefore
// always unread. An atomic-max output write is one of these once nothing needs
// the old value back.
// CHECK-LABEL: llvm.func @unread_atomic_fmax_erased
// CHECK-NOT:     rocdl.raw.ptr.buffer.atomic.fmax
// CHECK:       llvm.return
llvm.func @unread_atomic_fmax_erased(%ptr: !llvm.ptr<1>, %data: f32) attributes {rock.arch = "gfx1100"} {
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
  rocdl.raw.ptr.buffer.atomic.fmax %data, %rsrc, %voffset, %zero, %zero : f32
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

// The same for the ROCDL spelling, the only one of those that has a result to
// read at all.
// CHECK-LABEL: llvm.func @read_atomic_cmpswap_declines
// CHECK:         rocdl.raw.ptr.buffer.atomic.cmpswap
llvm.func @read_atomic_cmpswap_declines(%ptr: !llvm.ptr<1>, %val: i32, %cmp: i32) -> i32 attributes {rock.arch = "gfx1100"} {
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
  llvm.return %old : i32
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

// Both selects are well formed but pick their arms on different conditions, so
// the pair is no longer correlated. With %cond true and %uniform negative,
// soffset takes a value near 2^32 while the offset takes the plain sentinel,
// and that sum wraps back into the buffer.
// CHECK-LABEL: llvm.func @split_soffset_unshared_condition_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @split_soffset_unshared_condition_declines(%ptr: !llvm.ptr<1>, %data: i32,
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

  %nonNeg = llvm.icmp "sge" %uniform, %zero : i32
  %soffset = llvm.select %cond, %uniform, %zero : i1, i32
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

// The correlated pair is well formed, but the arm it falls back to when the
// lifted value is negative is an ordinary in-bounds offset rather than a
// sentinel, so that half of the access still reaches memory.
// CHECK-LABEL: llvm.func @split_soffset_in_bounds_fallback_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @split_soffset_in_bounds_fallback_declines(%ptr: !llvm.ptr<1>, %data: i32,
                                                    %uniform: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %minusOne = llvm.mlir.constant(-1 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %nonNeg = llvm.icmp "sge" %uniform, %zero : i32
  %soffset = llvm.select %nonNeg, %uniform, %zero : i1, i32
  %negated = llvm.sub %minusOne, %uniform : i32
  %splitOob = llvm.select %nonNeg, %negated, %offset : i1, i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %splitOob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %soffset, %zero : i32
  llvm.return
}

// -----

// The split shape proves nothing about a buffer larger than the offset arm's own
// lower bound of 2^31, which is all the shared condition gives. Both fallback
// arms are past this larger bound, so that is the only reason to decline.
// CHECK-LABEL: llvm.func @split_soffset_num_records_past_split_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @split_soffset_num_records_past_split_declines(%ptr: !llvm.ptr<1>, %data: i32,
                                                        %uniform: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483649 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %minusOne = llvm.mlir.constant(-1 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  // 4000000000 unsigned, past the numRecords above.
  %farOob = llvm.mlir.constant(-294967296 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %nonNeg = llvm.icmp "sge" %uniform, %zero : i32
  %soffset = llvm.select %nonNeg, %uniform, %zero : i1, i32
  %negated = llvm.sub %minusOne, %uniform : i32
  %splitOob = llvm.select %nonNeg, %negated, %farOob : i1, i32

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

// On gfx1250 only flags bits [3:0] reach the descriptor. These are the flags
// Triton builds there, whose low four bits are all zero.
// CHECK-LABEL: llvm.func @gfx1250_flags_fold
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @gfx1250_flags_fold(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250"} {
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
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// The flags above plus bit 0, swizzle_enable. A swizzled buffer addresses
// elements by a swizzle pattern rather than a flat offset, so numRecords no
// longer says which offsets fall outside.
// CHECK-LABEL: llvm.func @gfx1250_swizzle_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @gfx1250_swizzle_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243329 : i32) : i32
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

// A numRecords nothing constrains. The bound an access has to clear is read as
// an upper bound, so an unconstrained one is the whole of i64 and no offset
// clears it.
// CHECK-LABEL: llvm.func @dynamic_bound_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @dynamic_bound_declines(%ptr: !llvm.ptr<1>, %data: i32, %n: i64) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %n, %flags : <1> to <8>

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

// A numRecords that is dynamic but bounded: widened from an i16 and increased
// by one, so it lands in [1, 65536], which the sentinel offset clears whatever
// the runtime value is. The bound is a fact about the descriptor rather than a
// constant to match.
// CHECK-LABEL: llvm.func @dynamic_bounded_numrecords_folds
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @dynamic_bounded_numrecords_folds(%ptr: !llvm.ptr<1>, %data: i32, %n16: i16) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %one = llvm.mlir.constant(1 : i64) : i64
  %wide = llvm.zext %n16 : i16 to i64
  %n = llvm.add %wide, %one : i64
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %n, %flags : <1> to <8>

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

// A numRecords loaded from memory. Neither domain models memory, so the load
// gets its type's whole range, which no offset clears.
// CHECK-LABEL: llvm.func @loaded_bound_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @loaded_bound_declines(%ptr: !llvm.ptr<1>, %data: i32, %nptr: !llvm.ptr) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %n = llvm.load %nptr : !llvm.ptr -> i64
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %n, %flags : <1> to <8>

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

// The same load, narrowed to i16 and widened back, then moved off zero, so it
// provably lands in [1, 65536]. That the fold fires here is what shows the load
// above yields a real full-range fact that later ops refine, rather than no
// fact at all.
// CHECK-LABEL: llvm.func @loaded_then_narrowed_folds
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @loaded_then_narrowed_folds(%ptr: !llvm.ptr<1>, %data: i32, %nptr: !llvm.ptr) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %one = llvm.mlir.constant(1 : i64) : i64
  %n64 = llvm.load %nptr : !llvm.ptr -> i64
  %n16 = llvm.trunc %n64 : i64 to i16
  %wide = llvm.zext %n16 : i16 to i64
  %n = llvm.add %wide, %one : i64
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %n, %flags : <1> to <8>

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

// A zero numRecords marks the resource unbound rather than empty, so every
// offset clearing zero says nothing about what the hardware does.
// CHECK-LABEL: llvm.func @zero_num_records_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @zero_num_records_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(0 : i64) : i64
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

// An all-ones numRecords marks a resource whose size is not tracked. The
// sentinel here is -1 so that the offset does reach the bound: without the
// check on the field's extremes this would fold.
// CHECK-LABEL: llvm.func @all_ones_num_records_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @all_ones_num_records_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(4294967295 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-1 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// The three cases below write a constant whose attribute is narrower than the
// value's own type, which llvm.mlir.constant allows. Translation to LLVM IR
// sign-extends such an attribute unless it is unsigned or i1, so reading one as
// written is reading a different number than the access gets.

// Sign extension makes this numRecords the same all-ones sentinel as -1 : i64,
// which leaves the offset in bounds. Zero-extended it would be four billion,
// an ordinary extent that the offset does reach.
// CHECK-LABEL: llvm.func @narrow_num_records_sign_extends
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @narrow_num_records_sign_extends(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1250"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(-1 : i32) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-1 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// The same mismatch on the offset rather than the bound, so that the rule has to
// hold in the folding direction too: sign-extended, -1 : i16 is the all-ones
// offset that puts a masked lane out of bounds, while 65535 would sit inside
// this buffer.
// CHECK-LABEL: llvm.func @narrow_oob_offset_sign_extends
// CHECK-NOT:     rocdl.raw.ptr.buffer.store
// CHECK:       llvm.return
llvm.func @narrow_oob_offset_sign_extends(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-1 : i16) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// A boolean attribute, the case translation zero-extends instead. `true` is
// offset one, which this two-record buffer holds; sign-extended it would be all
// ones and the store would fold away.
// CHECK-LABEL: llvm.func @boolean_offset_zero_extends
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @boolean_offset_zero_extends(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(true) : i32
  %offset = llvm.mlir.constant(0 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %zero : i32
  llvm.return
}

// -----

// A dynamic numRecords whose range reaches zero. Its largest value is one the
// sentinel clears, but it could be the unbound encoding at runtime, so the
// range has to avoid zero and not merely be small.
// CHECK-LABEL: llvm.func @dynamic_range_including_zero_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @dynamic_range_including_zero_declines(%ptr: !llvm.ptr<1>, %data: i32, %n16: i16) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %n = llvm.zext %n16 : i16 to i64
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %n, %flags : <1> to <8>

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

// CPol::VOLATILE in the cache policy. The bit is not a hardware one, so the
// offset still puts the access out of bounds, but a volatile access has to
// reach the hardware as written.
// CHECK-LABEL: llvm.func @volatile_store_survives
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @volatile_store_survives(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %volatile = llvm.mlir.constant(-2147483648 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  rocdl.raw.ptr.buffer.store %data, %rsrc, %voffset, %zero, %volatile : i32
  llvm.return
}

// -----

// The same for a load, which would otherwise be replaced by zero.
// CHECK-LABEL: llvm.func @volatile_load_survives
// CHECK:         rocdl.raw.ptr.buffer.load
llvm.func @volatile_load_survives(%ptr: !llvm.ptr<1>) -> i32 attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>

  %zero = llvm.mlir.constant(0 : i32) : i32
  %volatile = llvm.mlir.constant(-2147483648 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %oob = llvm.mlir.constant(-2147483648 : i32) : i32
  %offset = llvm.mlir.constant(64 : i32) : i32

  %tid = rocdl.workitem.id.x : i32
  %row = llvm.urem %tid, %four : i32
  %pred = llvm.icmp "uge" %row, %sixteen : i32
  %voffset = llvm.select %pred, %offset, %oob : i1, i32
  %data = rocdl.raw.ptr.buffer.load %rsrc, %voffset, %zero, %volatile : i32
  llvm.return %data : i32
}

// -----

// A store in a block with no predecessors. The solvers never visit it, so its
// operands carry no facts and the fold declines. The block survives only
// because this pass erases accesses rather than code: the canonicalizer that
// follows it in the pipeline is what removes unreachable blocks.
// CHECK-LABEL: llvm.func @unreachable_block_declines
// CHECK:         rocdl.raw.ptr.buffer.store
llvm.func @unreachable_block_declines(%ptr: !llvm.ptr<1>, %data: i32) attributes {rock.arch = "gfx1100"} {
  %stride = llvm.mlir.constant(0 : i16) : i16
  %numRecords = llvm.mlir.constant(2147483646 : i64) : i64
  %flags = llvm.mlir.constant(822243328 : i32) : i32
  %rsrc = rocdl.make.buffer.rsrc %ptr, %stride, %numRecords, %flags : <1> to <8>
  llvm.br ^live

^dead:
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
  llvm.br ^live

^live:
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
