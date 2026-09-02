//===- DeviceInfo.cpp - Native AMD device information ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/DeviceInfo.h"

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "llvm/Support/raw_ostream.h"

#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <tuple>

namespace mlir::rock {

static constexpr const char *kWGPModeEnvVar = "GPU_ENABLE_WGP_MODE";

void requestWGPScheduling() {
  static bool warned = false;
  const char *current = std::getenv(kWGPModeEnvVar);
  if (current && StringRef(current) == "0" && !warned) {
    warned = true;
    llvm::errs() << "warning: " << kWGPModeEnvVar
                 << "=0 requests CU-mode scheduling, but grid sizes are "
                    "computed per workgroup processor; overriding to 1\n";
  }
#if defined(_WIN32)
  (void)_putenv_s(kWGPModeEnvVar, "1");
#else
  (void)setenv(kWGPModeEnvVar, "1", /*overwrite=*/1);
#endif
}

std::optional<NativeDeviceInfo> getNativeDeviceInfo(unsigned deviceId) {
  requestWGPScheduling();

  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount <= 0 ||
      deviceId >= static_cast<unsigned>(deviceCount))
    return std::nullopt;

  hipDeviceProp_t properties;
  if (hipGetDeviceProperties(&properties, static_cast<int>(deviceId)) !=
      hipSuccess)
    return std::nullopt;
  if (properties.multiProcessorCount <= 0)
    return std::nullopt;

  return NativeDeviceInfo{std::string(properties.gcnArchName),
                          properties.multiProcessorCount};
}

std::optional<int64_t> getNativeNumCU(StringRef targetArch) {
  // Assume all visible devices share an architecture and consult device 0.
  std::optional<NativeDeviceInfo> info = getNativeDeviceInfo();
  if (!info)
    return std::nullopt;

  StringRef targetChip = std::get<0>(parseArchString(targetArch));
  StringRef deviceChip = std::get<0>(parseArchString(info->arch));
  if (!targetChip.empty() && deviceChip != targetChip)
    return std::nullopt;

  return info->numCU;
}

} // namespace mlir::rock
