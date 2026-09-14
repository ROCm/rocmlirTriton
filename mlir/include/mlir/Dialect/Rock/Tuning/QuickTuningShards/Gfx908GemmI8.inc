//===- Gfx908GemmI8.inc - quick-tuning shard for gfx908_gemm_i8 -----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hand-written reference for the generated shard format. The three problem
// hashes are pinned by QuickTuningProblemKeyTests.cpp.
//
//===----------------------------------------------------------------------===//

// clang-format off

#ifdef QUICK_TUNING_DB_ARRAYS

static const char *const kCfgGfx908GemmI8[] = {
    "gemm:v1:16,64,128,1,1,4,16,1,2,0,0",
    "gemm:v1:32,64,128,1,1,4,16,1,2,0,0",
    "gemm:v1:16,16,128,1,1,1,16,1,2,0,0",
    "gemm:v1:128,64,64,1,1,4,16,1,2,0,0",
    "gemm:v1:64,64,128,1,1,4,32,1,2,0,0",
    "gemm:v1:64,128,64,1,1,8,32,1,2,0,0",
    "gemm:v1:32,128,128,1,1,4,32,1,1,0,0",
    "gemm:v1:256,256,64,1,1,4,32,1,2,0,0",
    "gemm:v1:16,64,128,1,1,4,16,4,2,0,0"
};

static const uint64_t kProbGfx908GemmI8[] = {
    0x1ef54fffbb33963dULL,
    0x69d6730b3d8f5b2eULL,
    0xdf076fce32a0c348ULL
};

static const uint16_t kProbCfgGfx908GemmI8[] = {
    7, 0, 1, 2, 3,
    5, 0, 1, 2, 3,
    8, 4, 0, 1, 2
};

#endif // QUICK_TUNING_DB_ARRAYS

#ifdef QUICK_TUNING_DB_ENTRIES
{"gfx908_gemm_i8",
 kCfgGfx908GemmI8, /*numConfigs=*/9,
 kProbGfx908GemmI8, kProbCfgGfx908GemmI8,
 /*numProblems=*/3, /*numTopN=*/5},
#endif // QUICK_TUNING_DB_ENTRIES
