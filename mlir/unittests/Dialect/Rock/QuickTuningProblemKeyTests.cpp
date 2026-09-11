//===- QuickTuningProblemKeyTests.cpp - Tests for the problem hash --------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningProblemKey.h"
#include "llvm/ADT/Twine.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

// Quick-tuning problem keys paired with the hashes the shards record them
// under. The hash is a measured problem's identity, so a shard written by one
// build has to stay readable by the next: changing either the key spelling or
// the hash function retires every recorded best without saying so. These are
// the keys quoted in the gfx908_gemm_i8 shard; rocmlir-gen prints these same
// hashes for
//   --operation=gemm -g 1 [-transA|-transB] -m M -n N -k K
// at every architecture and every data type (see
// test/rocmlir-gen/quick-tuning-problem-hash.mlir).
static const std::pair<StringRef, uint64_t> kGoldenHashes[] = {
    {"Gemm -transA true -transB false -transO false -g 1 -m 4096 -n 4096 -k "
     "4096",
     0x1ef54fffbb33963dULL},
    {"Gemm -transA false -transB true -transO false -g 1 -m 2048 -n 2048 -k "
     "2048",
     0x69d6730b3d8f5b2eULL},
    {"Gemm -transA false -transB false -transO false -g 1 -m 1024 -n 1024 -k "
     "1024",
     0xdf076fce32a0c348ULL},
};

TEST(QuickTuningProblemKeyTest, GoldenHashesAreStable) {
  for (auto &[key, hash] : kGoldenHashes)
    EXPECT_EQ(hash, hashQuickTuningProblemKey(key)) << "for key " << key;
}

TEST(QuickTuningProblemKeyTest, KernelTypeIsPartOfTheHashedKey) {
  // Key resolution is problem-agnostic, so a lookup may probe a shard belonging
  // to another operation -- a gemm+gemm one borrows attention's list, for
  // instance. Naming the kernel type in the hashed key is what makes that safe
  // by construction, rather than by hoping two operations' shape fields never
  // spell the same string.
  for (auto &[key, hash] : kGoldenHashes) {
    StringRef fields = key.drop_front(StringRef("Gemm").size());
    for (StringRef kernelType :
         {"Conv", "ConvBwdData", "Attention", "GemmElementwiseGemm",
          "ConvElementwiseGemm"}) {
      std::string other = (Twine(kernelType) + fields).str();
      EXPECT_NE(hash, hashQuickTuningProblemKey(other)) << "for key " << other;
    }
  }
}
