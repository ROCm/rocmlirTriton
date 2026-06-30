// End-to-end check that rock-fuse-sibling-loops (+ CSE) deduplicates shared
// operand loads when BOTH the M and N block tiles are non-power-of-two.
//
// The perf_config requests mPerBlock = nPerBlock = 80, so
// rock-decompose-nonpow2-tiles splits the GEMM along both M (64 + 16) and N
// (64 + 16), producing four pow2 sub-gemms, and rock-gridwise-gemm-to-blockwise
// emits one K-loop each. Across the four loops the A tiles depend only on the
// M-segment (two distinct: M=64, M=16) and the B tiles only on the N-segment
// (two distinct: N=64, N=16), so the same A/B tiles are reloaded by multiple
// loops. rock-fuse-sibling-loops merges all four loops into one and the
// following CSE collapses every duplicated load. We inspect the IR right before
// rock-insert-output-fusion-loads.
//
// Without fusion+CSE there would be four K-loops and eight operand loads. After
// fusion there is a single loop holding the four sub-gemms and only four loads:
// the two M-segment A tiles and the two N-segment B tiles.

// RUN: rocmlir-gen -operation gemm -t f16 -out_datatype f16 --arch gfx1100 -g 1 -m 77 -k 768 -n 36480 -transA=False -transB=False -transO=False --perf_config=gemm:v2:80,80,32,1,1,4,0,1,1,0,0,-1,-1,-1,-1,-1,-1 \
// RUN: | rocmlir-driver -c --mlir-print-ir-before=rock-insert-output-fusion-loads -o /dev/null 2>&1 \
// RUN: | FileCheck %s

// A single fused K-loop carrying all four sub-gemm accumulators.
// CHECK: %{{.*}}:4 = scf.for
// CHECK-NOT: scf.for

// Four operand loads survive: two M-segment A tiles and two N-segment B tiles,
// each shared across the sub-gemms that use it.
// CHECK-COUNT-4: rock.load_marker
// CHECK-NOT: rock.load_marker
