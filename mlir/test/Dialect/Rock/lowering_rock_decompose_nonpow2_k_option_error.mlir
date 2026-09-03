// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt -rock-decompose-nonpow2-k="dot-k=3" -verify-diagnostics %s

// expected-error @+1 {{dot-k must be zero or a positive power of two}}
func.func @invalid_dot_k() attributes {rock.kernel} {
  return
}
