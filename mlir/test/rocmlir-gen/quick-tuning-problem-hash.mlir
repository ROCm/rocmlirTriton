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
// is probed in already selects on both. Dropping them is what lets a lookup that
// reached its key by substitution -- f4 borrowing i8's list, gfx906 borrowing
// gfx908's -- still find its problem there. Below, the 1024 problem is generated
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

// A problem the database has measurements for sweeps them first. The gfx908
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

// The recorded bests are prepended to the cover rather than replacing it, and
// the 64x64x128 config the two have in common is swept once, so the known
// problem's list is exactly one entry longer than the unknown one's.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-KNOWN-SIZE
// CHECK-KNOWN-SIZE: {{^ *9$}}
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1023 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-UNKNOWN-SIZE
// CHECK-UNKNOWN-SIZE: {{^ *8$}}

// A lookup that only reached gfx908_gemm_i8 by substitution finds the problem
// there all the same, because key resolution is deliberately problem-agnostic.
// f4 has no lists of its own anywhere, so it borrows i8's.
// RUN: rocmlir-gen --arch gfx908 --operation=gemm -t f4E2M1FN -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-FALLBACK
// CHECK-FALLBACK: gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,{{.*}}splitKFactor=1,

//===----------------------------------------------------------------------===//
// ROCMLIR_QUICK_TUNING_LIST_MAX
//===----------------------------------------------------------------------===//

// The cap is what makes the narrowing shorten a sweep rather than just reorder
// it, and it bounds the whole list, the recorded bests included.
// RUN: ROCMLIR_QUICK_TUNING_LIST_MAX=3 rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-CAP-3
// CHECK-CAP-3: {{^ *3$}}

// RUN: ROCMLIR_QUICK_TUNING_LIST_MAX=1 rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | FileCheck %s --check-prefix=CHECK-CAP-1
// CHECK-CAP-1:      gemm:mPerBlock=64,nPerBlock=64,kPerBlock=128,{{.*}}splitKFactor=1,
// CHECK-CAP-1-NOT:  gemm:

// An unknown problem is not capped: without measurements to lead with there is
// nothing to shorten, and truncating the set cover would just lose coverage.
// RUN: ROCMLIR_QUICK_TUNING_LIST_MAX=1 rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1023 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-CAP-UNKNOWN
// CHECK-CAP-UNKNOWN: {{^ *8$}}

// An unparseable or non-positive value falls back to the default of 30, which is
// above the whole list here, so the known problem keeps all 9 entries.
// RUN: ROCMLIR_QUICK_TUNING_LIST_MAX=nonsense rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-CAP-DEFAULT
// RUN: ROCMLIR_QUICK_TUNING_LIST_MAX=0 rocmlir-gen --arch gfx908 --operation=gemm -t i8 -g 1 -m 1024 -n 1024 -k 1024 --num_cu=120 --emit-tuning-space=quick | wc -l | FileCheck %s --check-prefix=CHECK-CAP-DEFAULT
// CHECK-CAP-DEFAULT: {{^ *9$}}
