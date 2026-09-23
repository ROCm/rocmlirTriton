// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Negative tests for rock-emulate-narrow-types.
// The upstream gather_to_lds cases target AMDGPU rewrite patterns that this
// pass does not run; cover a memref op the fork actually refuses to legalize.

// RUN: not rocmlir-opt -rock-emulate-narrow-types -split-input-file %s 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'memref.copy'
func.func @copy_distinct_layouts(%src: memref<32xi4>, %dst: memref<32xi4, strided<[2]>>) {
  memref.copy %src, %dst : memref<32xi4> to memref<32xi4, strided<[2]>>
  func.return
}
