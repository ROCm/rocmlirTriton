// The quick-tuning problem hash, and the narrowing it drives, end to end.
//
// The unit tests (unittests/Dialect/Rock/QuickTuning*Tests.cpp) pin the hash
// function and the shard semantics against literal keys. What only a real
// operation can show is that the key rocmlir-gen serializes *is* that literal:
// the hashes checked below are the hashes of the keys quoted in
// kGoldenHashes there, so these RUN lines are what ties the shipped shard data
// to the operations it was measured on.

//===----------------------------------------------------------------------===//
// Hash stability
//===----------------------------------------------------------------------===//

// A shard is written by one build and read by the next, so a problem's hash may
// not move. These three are the problems the gfx908_gemm_i8 shard records.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -transA -m 4096 -n 4096 -k 4096 --emit-quick-tuning-hash | FileCheck %s --check-prefix=CHECK-HASH-4096
// CHECK-HASH-4096: 0x1ef54fffbb33963d

// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -transB -m 2048 -n 2048 -k 2048 --emit-quick-tuning-hash | FileCheck %s --check-prefix=CHECK-HASH-2048
// CHECK-HASH-2048: 0x69d6730b3d8f5b2e

// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --emit-quick-tuning-hash | FileCheck %s --check-prefix=CHECK-HASH-1024
// CHECK-HASH-1024: 0xdf076fce32a0c348

// The key names neither the architecture nor the data type, because the shard it
// is probed in already selects on both. Below, the 1024 problem is generated
// for four other targets and has to keep the hash it had above.
// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --emit-quick-tuning-hash > %t.hashes
// RUN: rocmlir-gen --arch gfx1100 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --emit-quick-tuning-hash >> %t.hashes
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t f16 -g 1 -m 1024 -n 1024 -k 1024 --emit-quick-tuning-hash >> %t.hashes
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t f32 -g 1 -m 1024 -n 1024 -k 1024 --emit-quick-tuning-hash >> %t.hashes
// RUN: sort -u %t.hashes | FileCheck %s --check-prefix=CHECK-HASH-INVARIANT
// CHECK-HASH-INVARIANT-COUNT-1: 0xdf076fce32a0c348
// CHECK-HASH-INVARIANT-NOT: 0x

// Shape and layout are in the key, so problems that differ in either do not
// collide -- in particular the 1024 problem above is not what a neighbouring
// shape hashes to.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1023 -n 1024 -k 1024 --emit-quick-tuning-hash | FileCheck %s --check-prefix=CHECK-HASH-NEIGHBOUR
// CHECK-HASH-NEIGHBOUR-NOT: 0xdf076fce32a0c348
// CHECK-HASH-NEIGHBOUR: 0x{{[0-9a-f]{16}$}}

//===----------------------------------------------------------------------===//
// Per-problem narrowing of the quick-tuning space
//===----------------------------------------------------------------------===//

// A problem the database has measurements for sweeps exactly its recorded
// top-N. The gfx908
// gemm i8 1024 problem records a 64x64x128 non-split-K best and a 16x64x128
// splitKFactor=4 one, in that order: the head of the list is what a
// skip-benchmarking consumer runs, so it has to be the config that is legal in
// every fusion context.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-KNOWN
// CHECK-KNOWN:      gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,{{.*}}splitKFactor=1,
// CHECK-KNOWN-NEXT: gemm:mPerBlock=16,nPerBlock=64,kPerBlock=128,{{.*}}splitKFactor=4,

// The same shape one row shorter has no measurements, so it sweeps the set cover
// unchanged: the 64x64x128 config is still there, five entries in rather than
// first, and the split-K one is absent from the cover entirely.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1023 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-UNKNOWN --implicit-check-not='splitKFactor=4,'
// CHECK-UNKNOWN:      gemm:mPerBlock=16,nPerBlock=64,kPerBlock=128,
// CHECK-UNKNOWN-NEXT: gemm:mPerBlock=32,nPerBlock=64,kPerBlock=128,
// CHECK-UNKNOWN-NEXT: gemm:mPerBlock=16,nPerBlock=16,kPerBlock=128,
// CHECK-UNKNOWN-NEXT: gemm:mPerBlock=128,nPerBlock=64,kPerBlock=64,
// CHECK-UNKNOWN-NEXT: gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,

// A hit returns exactly the shard's generated top-N; a miss returns the
// monolith's full set cover.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-KNOWN-SIZE
// CHECK-KNOWN-SIZE: {{^ *5$}}
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1023 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-UNKNOWN-SIZE
// CHECK-UNKNOWN-SIZE: {{^ *8$}}

// A fallback key uses the borrowed key's set cover but does not probe its
// problem map.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t f4E2M1FN -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-FALLBACK
// CHECK-FALLBACK: gemm:mPerBlock=16,nPerBlock=64,kPerBlock=128,
