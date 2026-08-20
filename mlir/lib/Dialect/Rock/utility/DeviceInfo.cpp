//===- DeviceInfo.cpp - Native AMD device information ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/DeviceInfo.h"

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#ifndef __HIP_DEVICE_COMPILE__
#define __HIP_DEVICE_COMPILE__ 0
#endif
#ifndef USE_PROF_API
#define USE_PROF_API 0
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++98-compat-extra-semi"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wundef"
#include <hip/hip_runtime_api.h>
#pragma clang diagnostic pop

#include <cstdlib>
#include <tuple>

namespace mlir::rock {

std::optional<int64_t> getNativeNumCU(StringRef targetArch) {
#if defined(_WIN32)
  if (_putenv_s("GPU_ENABLE_WGP_MODE", "1") != 0)
    return std::nullopt;
#else
  if (setenv("GPU_ENABLE_WGP_MODE", "1", 1) != 0)
    return std::nullopt;
#endif

  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    return std::nullopt;

  StringRef targetChip = std::get<0>(parseArchString(targetArch));
  // Assume all visible devices have the same architecture and use device 0.
  hipDeviceProp_t properties;
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess)
    return std::nullopt;

  StringRef deviceChip =
      std::get<0>(parseArchString(StringRef(properties.gcnArchName)));
  if ((targetChip.empty() || deviceChip == targetChip) &&
      properties.multiProcessorCount > 0)
    return properties.multiProcessorCount;

  return std::nullopt;
}

} // namespace mlir::rock
