//===- mixr_stream_k.cpp - C/MLIR API test of fusability with Stream-K ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a lit test (not a separate binary): it reuses the
// `mlir-mixr-split-k-test` driver with its `-stream-k` flag, which emits a v5
// stream-K perf_config (streamKMultiple >= 1, splitKFactor pinned to 1) instead
// of the split-K one. Stream-K reuses split-K's fusion-legality and
// output-prefill machinery, so this exercises the same CAPI paths
// (mlirIsModuleFusible, mlirGetPrefillArgsInfo) for the stream-K perf_config.
//
//===----------------------------------------------------------------------===//

// clang-format off

// Stream-K is fusible with a plain elementwise-free GEMM (same as split-K).
// RUN: mlir-mixr-split-k-test -t=DataType::F32 -stream-k 1 -use-ew-op=false 2>&1 | FileCheck %s --check-prefix=STREAMK_WITHOUT_EW

// A relu/max output fusion cannot be reconstructed from atomic-added partial
// sums, so stream-K (like split-K) reports the fused GEMM as not fusible.
// RUN: mlir-mixr-split-k-test -t=DataType::F32 -stream-k 1 -use-ew-op=true  2>&1 | FileCheck %s --check-prefix=STREAMK_WITH_EW

// A GEMM whose tile grid leaves a ragged tail for gfx942's num_cu actually
// hybrid-decomposes: the split-K remainder atomic_adds into a zero-prefilled
// output, so the kernel result argument (index 2) carries a prefill = 0 value.
// RUN: mlir-mixr-split-k-test -t=DataType::F32 -target-arch="gfx942:sramecc+:xnack-" -m 640 -n 448 -k 256 -stream-k 1 -use-ew-op=false 2>&1 | FileCheck %s --check-prefix=STREAMK_DECOMPOSE

// clang-format on

// STREAMK_WITHOUT_EW: splitk selection likelihood: never
// STREAMK_WITHOUT_EW: is fusible: true
// STREAMK_WITHOUT_EW: num prefill args: 0

// STREAMK_WITH_EW: is fusible: false

// STREAMK_DECOMPOSE: is fusible: true
// STREAMK_DECOMPOSE: num prefill args: 1
// STREAMK_DECOMPOSE: prefill arg indices: 2
// STREAMK_DECOMPOSE: prefill arg init values: 0.00
