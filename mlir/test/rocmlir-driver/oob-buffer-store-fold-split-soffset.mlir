// The companion to oob-buffer-store-fold.mlir, for the shape that test does not
// reach. When tritonamdgpu-annotate-buffer-op-split-safety lets the buffer-op
// emitter lift a uniform part of the offset into an soffset SGPR, the emitter
// stops using a bare sentinel and emits a correlated pair instead:
//
//   %c    = llvm.icmp "sge" %uniform, 0
//   soff  = llvm.select %c, %uniform, 0
//   %oob  = llvm.select %c, llvm.sub(-1, %uniform), -2147483648
//
// Both arms are still discarded, but seeing that takes the two selects together:
// on their own the offset is unbounded below and the soffset unbounded above.
//
// A 2x5 by 5x5 dot is enough, since M and N are both padded up to the tile size.
// gfx1100 and gfx1200 take the split path and erase 31 of 32 stores; the other
// targets keep a zero soffset here and go through the plain sentinel.
// perf_config is pinned so the counts do not move with the tuning heuristics.

// RUN: sed s/##TOKEN_ARCH##/gfx1100/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1100 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx1100 | FileCheck %s --check-prefix=SPLIT
// RUN: sed s/##TOKEN_ARCH##/gfx1200/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1200 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx1200 | FileCheck %s --check-prefix=SPLIT
// RUN: sed s/##TOKEN_ARCH##/gfx942/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx942 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx942 | FileCheck %s --check-prefix=GFX942
// RUN: sed s/##TOKEN_ARCH##/gfx950/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx950 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx950 | FileCheck %s --check-prefix=GFX950
// RUN: sed s/##TOKEN_ARCH##/gfx1250/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1250 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx1250 | FileCheck %s --check-prefix=GFX1250

// The statistics are the only place the load side of the fold is pinned, since a
// load that stops folding leaves no trace in the store checks above. They go to
// stderr, hence the separate runs.
// RUN: sed s/##TOKEN_ARCH##/gfx1100/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1100 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx1100 -o /dev/null -mlir-pass-statistics 2>&1 | FileCheck %s --check-prefix=STATS-SPLIT
// RUN: sed s/##TOKEN_ARCH##/gfx1200/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1200 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx1200 -o /dev/null -mlir-pass-statistics 2>&1 | FileCheck %s --check-prefix=STATS-SPLIT
// RUN: sed s/##TOKEN_ARCH##/gfx942/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx942 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx942 -o /dev/null -mlir-pass-statistics 2>&1 | FileCheck %s --check-prefix=STATS-CDNA
// RUN: sed s/##TOKEN_ARCH##/gfx950/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx950 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx950 -o /dev/null -mlir-pass-statistics 2>&1 | FileCheck %s --check-prefix=STATS-CDNA
// RUN: sed s/##TOKEN_ARCH##/gfx1250/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx1250 | rocmlir-driver -kernel-pipeline=gpu,triton -arch gfx1250 -o /dev/null -mlir-pass-statistics 2>&1 | FileCheck %s --check-prefix=STATS-GFX1250

// The correlated select pair. The lone `icmp "sge"` in the kernel is the
// split-safety guard, so it anchors the shape.
// SPLIT: %[[NONNEG:.*]] = llvm.icmp "sge" %[[UNIFORM:.*]], %{{.*}} : i32
// SPLIT: llvm.select %[[NONNEG]], %[[UNIFORM]], %{{.*}} : i1, i32
// SPLIT: %[[NEGATED:.*]] = llvm.sub %{{.*}}, %[[UNIFORM]] : i32
// SPLIT: llvm.select %[[NONNEG]], %[[NEGATED]], %{{.*}} : i1, i32

// SPLIT-COUNT-1: rocdl.raw.ptr.buffer.store
// SPLIT-NOT: rocdl.raw.ptr.buffer.store

// Pinning the plain-sentinel targets keeps a change in the split-safety analysis
// visible as a failure rather than as silently narrowed coverage.
// GFX942-NOT: llvm.icmp "sge"
// GFX942-COUNT-4: rocdl.raw.ptr.buffer.store
// GFX942-NOT: rocdl.raw.ptr.buffer.store

// GFX950-NOT: llvm.icmp "sge"
// GFX950-COUNT-4: rocdl.raw.ptr.buffer.store
// GFX950-NOT: rocdl.raw.ptr.buffer.store

// GFX1250-NOT: llvm.icmp "sge"
// GFX1250-COUNT-5: rocdl.raw.ptr.buffer.store
// GFX1250-NOT: rocdl.raw.ptr.buffer.store

// STATS-SPLIT:      RockFoldOobBufferOpsPass
// STATS-SPLIT-NEXT:   (S) 0 num-erased-atomics
// STATS-SPLIT-NEXT:   (S) 31 num-erased-stores
// STATS-SPLIT-NEXT:   (S) 91 num-folded-loads

// STATS-CDNA:      RockFoldOobBufferOpsPass
// STATS-CDNA-NEXT:   (S) 0 num-erased-atomics
// STATS-CDNA-NEXT:   (S) 12 num-erased-stores
// STATS-CDNA-NEXT:   (S) 44 num-folded-loads

// STATS-GFX1250:      RockFoldOobBufferOpsPass
// STATS-GFX1250-NEXT:   (S) 0 num-erased-atomics
// STATS-GFX1250-NEXT:   (S) 27 num-erased-stores
// STATS-GFX1250-NEXT:   (S) 91 num-folded-loads

module {
  func.func @mlir_dot_add_sigmoid(%arg0: !migraphx.shaped<2x5xf32, 5x1>, %arg1: !migraphx.shaped<5x5xf32, 5x1>, %arg2: !migraphx.shaped<2x5xf32, 5x1>) -> !migraphx.shaped<2x5xf32, 5x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr", rock.num_cu = 256 : i64} {
    %0 = migraphx.dot %arg0, %arg1 {perf_config = "gemm:v5:64,64,64,1,1,4,16,1,2,0,0,-1,-1,-1,-1,-1,-1,-1"} : <2x5xf32, 5x1>, <5x5xf32, 5x1> -> <2x5xf32, 5x1>
    %1 = migraphx.add %0, %arg2 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %2 = migraphx.sigmoid %1 : <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    return %2 : !migraphx.shaped<2x5xf32, 5x1>
  }
}
