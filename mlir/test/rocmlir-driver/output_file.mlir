// Verify rocmlir-driver writes its result to the file given by -o instead of
// stdout.

// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx1030 -n 128 -k 8 -m 256 \
// RUN: | rocmlir-driver -o %t
// RUN: test -f %t
// RUN: FileCheck %s < %t

// CHECK: rock.gemm
