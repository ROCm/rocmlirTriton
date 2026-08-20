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

namespace mlir::rock {

/// Query the native CU count for a HIP device matching `targetArch`.
///
/// WGP mode is enabled before HIP initialization. Returns std::nullopt when
/// HIP is unavailable, no device is present, or no visible device matches the
/// requested architecture.
std::optional<int64_t> getNativeNumCU(StringRef targetArch);

} // namespace mlir::rock

#endif // MLIR_DIALECT_ROCK_UTILITY_DEVICEINFO_H
