// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// rock.fp16_accumulation must survive rock-affix-params attribute validation
// and reach rock-gridwise-gemm-to-blockwise, which picks a 16-bit accumulator.

// RUN: rocmlir-opt -rock-affix-params -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise %s | FileCheck %s

// CHECK-LABEL: func.func @gfx1250_f16_fp16_accumulation
// CHECK-SAME: rock.fp16_accumulation
// CHECK: rock.blockwise_gemm
// CHECK-SAME: -> tensor<{{[0-9]+x[0-9]+}}xf16>
func.func @gfx1250_f16_fp16_accumulation(%a : tensor<1x256x256xf16>, %b : tensor<1x256x256xf16>, %c : tensor<1x256x256xf16>) -> tensor<1x256x256xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1250", rock.num_cu = 256 : i64, rock.fp16_accumulation} {
  %result = rock.gemm %a * %b : tensor<1x256x256xf16> * tensor<1x256x256xf16> -> tensor<1x256x256xf16>
  %out = rock.store %result to %c by set : tensor<1x256x256xf16> -> tensor<1x256x256xf16> to tensor<1x256x256xf16>
  return %out : tensor<1x256x256xf16>
}

// CHECK-LABEL: func.func @gfx1250_f16_default_accumulation
// CHECK: rock.blockwise_gemm
// CHECK-SAME: -> tensor<{{[0-9]+x[0-9]+}}xf32>
func.func @gfx1250_f16_default_accumulation(%a : tensor<1x256x256xf16>, %b : tensor<1x256x256xf16>, %c : tensor<1x256x256xf16>) -> tensor<1x256x256xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1250", rock.num_cu = 256 : i64} {
  %result = rock.gemm %a * %b : tensor<1x256x256xf16> * tensor<1x256x256xf16> -> tensor<1x256x256xf16>
  %out = rock.store %result to %c by set : tensor<1x256x256xf16> -> tensor<1x256x256xf16> to tensor<1x256x256xf16>
  return %out : tensor<1x256x256xf16>
}
