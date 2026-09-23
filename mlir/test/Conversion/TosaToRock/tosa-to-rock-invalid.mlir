// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt --tosa-to-rock -verify-diagnostics --split-input-file %s

// expected-error @+1 {{func op does not have the kernel attribute}}
func.func @no_kernel_attribute_test() {
  func.return
}
