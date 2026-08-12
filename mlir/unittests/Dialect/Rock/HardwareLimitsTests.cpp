//===- HardwareLimitsTests.cpp - Tests for AMD hardware limits ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"

#include "hip/hip_runtime_api.h"
#include "gtest/gtest.h"

#include <cstdint>

using namespace mlir::rock;

TEST(HardwareLimitsTest, MaxGridSizeMatchesHip) {
  int deviceCount = 0;
  hipError_t status = hipGetDeviceCount(&deviceCount);
  if (status == hipErrorNoDevice || deviceCount == 0)
    GTEST_SKIP() << "no HIP devices available";
  ASSERT_EQ(status, hipSuccess) << hipGetErrorString(status);

  for (int device = 0; device < deviceCount; ++device) {
    int maxGridDimX = 0;
    status = hipDeviceGetAttribute(&maxGridDimX, hipDeviceAttributeMaxGridDimX,
                                   device);
    ASSERT_EQ(status, hipSuccess) << hipGetErrorString(status);
    EXPECT_EQ(maxHardwareGridSize, static_cast<uint32_t>(maxGridDimX));
  }
}
