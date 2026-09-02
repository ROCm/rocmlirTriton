//===- HardwareLimitsTests.cpp - Tests for AMD hardware limits ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/DeviceInfo.h"

#include "hip/hip_runtime_api.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>

using namespace mlir::rock;

TEST(HardwareLimitsTest, LimitsMatchHip) {
  int deviceCount = 0;
  hipError_t status = hipGetDeviceCount(&deviceCount);
  if (status == hipErrorNoDevice)
    GTEST_SKIP() << "no HIP devices available";
  ASSERT_EQ(status, hipSuccess) << hipGetErrorString(status);
  if (deviceCount == 0)
    GTEST_SKIP() << "no HIP devices available";

  for (int device = 0; device < deviceCount; ++device) {
    int maxGridDimX = 0;
    status = hipDeviceGetAttribute(&maxGridDimX, hipDeviceAttributeMaxGridDimX,
                                   device);
    ASSERT_EQ(status, hipSuccess) << hipGetErrorString(status);
    EXPECT_LE(static_cast<uint32_t>(maxGridDimX), maxHardwareGridSize);

    int maxThreadsPerBlock = 0;
    status = hipDeviceGetAttribute(
        &maxThreadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, device);
    ASSERT_EQ(status, hipSuccess) << hipGetErrorString(status);
    EXPECT_EQ(maxHardwareWorkgroupSize, maxThreadsPerBlock);
  }
}

// `getMinNumCU` is a floor on what a real device can report, and nothing
// enforces it while lowering: a caller running on a compute-partitioned device
// legitimately passes fewer CUs than the family's flagship, so rejecting that
// would fail compilation on valid input. Assert the bound here instead, where
// a part or partition mode that drops below the table shows up as a test
// failure on the machine that has it rather than as silently wrong tuning.
TEST(HardwareLimitsTest, NativeNumCUIsAtLeastTheArchFloor) {
  int deviceCount = 0;
  hipError_t status = hipGetDeviceCount(&deviceCount);
  if (status == hipErrorNoDevice)
    GTEST_SKIP() << "no HIP devices available";
  ASSERT_EQ(status, hipSuccess) << hipGetErrorString(status);
  if (deviceCount == 0)
    GTEST_SKIP() << "no HIP devices available";

  // getNativeNumCU always reports device 0, so this checks that one device.
  hipDeviceProp_t prop;
  status = hipGetDeviceProperties(&prop, 0);
  ASSERT_EQ(status, hipSuccess) << hipGetErrorString(status);
  llvm::StringRef arch(prop.gcnArchName);

  std::optional<int64_t> nativeNumCU = mlir::rock::getNativeNumCU(arch);
  ASSERT_TRUE(nativeNumCU.has_value()) << arch.str();
  EXPECT_GE(*nativeNumCU, mlir::rock::getMinNumCU(arch)) << arch.str();
}
