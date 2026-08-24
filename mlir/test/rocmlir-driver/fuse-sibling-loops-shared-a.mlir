// End-to-end check that rock-fuse-sibling-loops (+ CSE) deduplicates the shared
// A tile load when a non-power-of-two N tile is decomposed into pow2 sub-gemms.
//
// The perf_config requests nPerBlock = 80 (mPerBlock stays pow2), so
// rock-decompose-nonpow2-tiles splits the GEMM along N into 64 + 16 segments,
// and rock-gridwise-gemm-to-blockwise emits one K-loop per segment. Both loops
// load the same A tile (tensor<128x32xf16>); the B tile differs per N-segment.
// rock-fuse-sibling-loops merges the loops and the following CSE collapses the
// now-co-located duplicate A load. We inspect the IR right before
// rock-insert-output-fusion-loads.
//
// Without fusion+CSE there would be two K-loops and four operand loads (2 A +
// 2 B). After fusion there is a single loop with three loads: one shared A plus
// the two B N-segments.

// RUN: rocmlir-gen -operation gemm -t f16 -out_datatype f16 --arch gfx1100 -g 1 -m 77 -k 768 -n 36480 -transA=False -transB=False -transO=False --perf_config=gemm:v2:128,80,32,1,1,4,0,1,1,0,0,-1,-1,-1,-1,-1,-1 \
// RUN: | rocmlir-driver -c --mlir-print-ir-before=rock-insert-output-fusion-loads -o /dev/null 2>&1 \
// RUN: | FileCheck %s

// A single fused K-loop carrying both N-segment accumulators.
// CHECK: %{{.*}}:2 = scf.for
// CHECK-NOT: scf.for

// Three operand loads survive inside it: the shared A tile (deduplicated) and
// the two decomposed B N-segments.
// CHECK-COUNT-3: rock.load_marker
// CHECK-NOT: rock.load_marker
