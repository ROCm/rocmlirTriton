//===- DeviceInfo.h - Native AMD device information -----------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_DEVICEINFO_H
#define MLIR_DIALECT_ROCK_UTILITY_DEVICEINFO_H

#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::rock {

/// Ask HIP to schedule per workgroup processor, so that a device report counts
/// WGPs rather than CUs on the architectures that distinguish the two. Rock
/// sizes grids in whatever unit runs one workgroup, which is the WGP where one
/// exists, and `hipDeviceProp_t::multiProcessorCount` follows the mode the
/// process runs in.
///
/// HIP reads the setting while it initializes, so this has to run before the
/// process makes any HIP call; tools should call it at the top of `main`. It is
/// idempotent, and it warns once when it overrides an explicit request for CU
/// mode instead of silently disagreeing with the caller.
void requestWGPScheduling();

/// What a visible HIP device reports about itself.
struct NativeDeviceInfo {
  /// The device's own architecture string, e.g. "gfx942:sramecc+:xnack-".
  std::string arch;
  /// Scheduling units, normalized to WGPs by `requestWGPScheduling`.
  int64_t numCU;
};

/// How many HIP devices are visible, or 0 when HIP is unavailable at runtime.
int64_t getNativeDeviceCount();

/// Query `deviceId`, or std::nullopt when HIP is unavailable at runtime, no
/// device is visible, or the device reports a nonsensical count.
std::optional<NativeDeviceInfo> getNativeDeviceInfo(unsigned deviceId = 0);

/// The scheduling-unit count of a visible device whose architecture matches
/// `targetArch`, or std::nullopt when there is none. An empty `targetArch`
/// matches any device. Comparison is on the bare gfx token, so target features
/// such as `:xnack-` do not have to agree.
///
/// Callers that compile for an architecture other than the one in front of
/// them must not use a foreign device's count, hence the match.
std::optional<int64_t> getNativeNumCU(StringRef targetArch);

} // namespace mlir::rock

#endif // MLIR_DIALECT_ROCK_UTILITY_DEVICEINFO_H
