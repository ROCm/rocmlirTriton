//===- CpuTileLUT.cpp - Per-CPU matmul tile-size lookup -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of `lookupHostCpuTileSizes`. The host is identified via
// LLVM's `sys::getHostCPUName()`, which already maps CPUID family/model
// fingerprints to a canonical microarchitecture name (`sapphirerapids`,
// `znver4`, `icelake-server`, ...). That string is the LUT key.
//
// The LUT is intentionally small and explicit. Each per-CPU function takes
// the matmul shape `(M, N, K)` and returns the tile triple the author
// measured to be best on that microarchitecture. Adding a new CPU is a
// localized change: write one function, add one row to `kCpuTable`.
//
//===----------------------------------------------------------------------===//

#include "CpuTileLUT.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#define DEBUG_TYPE "cpu-tile-lut"

using namespace mlir;

namespace {

/// Signature of a per-CPU tile-size picker. Each implementation may inspect
/// `(M, N, K)` and return the tile triple it considers best on its target
/// microarchitecture. `ShapedType::kDynamic` may appear for dims that
/// weren't statically known; entries should treat those as "unknown" and
/// pick a conservative default.
using CpuPicker = cpu::CpuTileTriple (*)(int64_t M, int64_t N, int64_t K);

//===----------------------------------------------------------------------===//
// Per-CPU pickers.
//===----------------------------------------------------------------------===//

/// Intel Sapphire Rapids (Xeon Platinum 84xx series). AVX-512 + AMX, 2 MiB
/// L2 per core, 56-105 MiB L3 per socket. These initial values are
/// placeholders intended to be replaced by measurement -- the point of the
/// LUT is to make that measurement workflow possible. The shape parameters
/// are ignored for now; once tuning data is in, splits can be added.
static cpu::CpuTileTriple pickSapphireRapids(int64_t /*M*/, int64_t /*N*/,
                                             int64_t /*K*/) {
  // TODO(AIROCMLIR-812): Replace with measured values per shape bucket.
  return {/*mFuse=*/256, /*nFuse=*/64, /*kTile=*/128};
}

//===----------------------------------------------------------------------===//
// Dispatch table.
//===----------------------------------------------------------------------===//

struct CpuEntry {
  llvm::StringRef cpuName;
  CpuPicker picker;
};

// Keep this sorted by `cpuName` for readability. Lookup is linear because
// the table is small; if it grows past a handful of entries, switch to
// `llvm::StringSwitch`.
static constexpr CpuEntry kCpuTable[] = {
    {"sapphirerapids", &pickSapphireRapids},
};

} // namespace

namespace mlir {
namespace cpu {

std::optional<CpuTileTriple> lookupHostCpuTileSizes(int64_t M, int64_t N,
                                                    int64_t K) {
  llvm::StringRef hostCpu = llvm::sys::getHostCPUName();
  LLVM_DEBUG(llvm::dbgs() << "cpu-tile-lut: host CPU = '" << hostCpu << "'\n");

  for (const CpuEntry &entry : kCpuTable) {
    if (entry.cpuName == hostCpu) {
      CpuTileTriple t = entry.picker(M, N, K);
      LLVM_DEBUG(llvm::dbgs()
                 << "cpu-tile-lut: hit for '" << hostCpu << "' -> mFuse="
                 << t.mFuse << " nFuse=" << t.nFuse << " kTile=" << t.kTile
                 << " (M=" << M << " N=" << N << " K=" << K << ")\n");
      return t;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "cpu-tile-lut: miss for '" << hostCpu
                          << "', caller should fall back\n");
  return std::nullopt;
}

} // namespace cpu
} // namespace mlir
