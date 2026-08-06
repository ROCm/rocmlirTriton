//===- ArtifactIO.h - Kernel-bundle types, format, and IO -----------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The cross-compilation kernel-bundle format: the in-memory representation of a
// compiled perf-config space, its on-disk (de)serialization, and the generic
// (de)compression / file-IO primitives those routines build on. All of this is
// independent of the tuning loop and HIP benchmarking themselves.
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_TUNING_DRIVER_ARTIFACT_IO_H
#define ROCMLIR_TUNING_DRIVER_ARTIFACT_IO_H

#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "../utils/performance/common/benchmarkUtils.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rocmlir::tuningdriver {

enum class CompilationStatus {
  NotApplicable,     // Config not applicable for this kernel
  CompilationFailed, // Config applicable but compilation failed
  TimedOut,          // Compilation exceeded the per-config timeout
  Success            // Successfully compiled
};

struct CompilationResult {
  llvm::SmallString<64> perfConfig;
  CompilationStatus status = CompilationStatus::NotApplicable;
  std::string hsacoBinary; // Single HSACO binary containing all kernels
  llvm::SmallVector<mlir::rock::KernelInfo>
      kernels; // Info for each kernel (name, block/grid sizes)
  std::string blobPath;
};

struct BufferLayout {
  llvm::SmallVector<size_t> byteLengths;
  llvm::SmallVector<benchmark::DataType> dataTypes;
};

// Decoded kernel-bundle manifest metadata (everything except the per-config
// list, which is materialized into CompilationResult[]).
struct ManifestInfo {
  uint64_t hipVersion = 0;
  std::string arch;
  int64_t numCUs = 0;
  int64_t numChiplets = 0;
  int64_t numConfigs = 0;
};

//===----------------------------------------------------------------------===//
// Generic (de)compression framing and file IO primitives.
//===----------------------------------------------------------------------===//

/// Self-describing compressed container: an 8-byte little-endian uncompressed
/// size followed by a zstd stream. Used for both HSACO blobs and the manifest
/// itself (the manifest has no outer place to record its own size). Callers are
/// responsible for checking llvm::compression::zstd::isAvailable() first.
/// The frame carries no format tag, so bundles written by a build using a
/// different compressor are not readable here (they fail to inflate).
std::string compressFramed(llvm::StringRef raw);

/// Inverse of compressFramed: validate the 8-byte size header (including the
/// implementation's uncompressed-size limit), then inflate the trailing zstd
/// stream. Returns failure() (after logging) on a truncated or corrupt blob.
mlir::FailureOr<std::string> decompressFramed(llvm::StringRef framed);

/// Write `contents` verbatim to `path` as raw bytes (no text translation).
/// Returns failure() after logging on any open/write error.
mlir::LogicalResult writeFileContents(llvm::StringRef path,
                                      llvm::StringRef contents);

/// Read the entire file at `path` into a string as raw bytes. Returns
/// failure() after logging if the file cannot be opened.
mlir::FailureOr<std::string> readFileContents(llvm::StringRef path);

//===----------------------------------------------------------------------===//
// Kernel-bundle serialization (streaming).
//===----------------------------------------------------------------------===//
//
// A bundle is written incrementally so the driver never has to hold every
// compiled HSACO in memory at once (the whole exhaustive space can be many
// GiB):
//
//   1. beginArtifactBundle(dir)          -- once, before compilation.
//   2. writeArtifactBlob(dir, i, hsaco)  -- per Success config, from any
//   thread;
//                                           the caller frees its in-memory
//                                           HSACO immediately after this
//                                           returns.
//   3. finalizeArtifactBundle(dir, ...)  -- once, after compilation.
//
// The bundle is assembled in a sibling `<dir>.tmp` and only renamed onto `dir`
// by finalizeArtifactBundle, so an interrupted compile never leaves a
// half-written dir that looks complete.

/// Prepare (or reset) the staging area for a bundle: verify zstd is available,
/// remove any stale `<dir>.tmp`, and create `<dir>.tmp/configs`. Must be called
/// before any writeArtifactBlob. Checking zstd here fails the run fast, before
/// spending compile time, if the build lacks compression support.
mlir::LogicalResult beginArtifactBundle(llvm::StringRef dir);

/// Compress `hsacoBinary` and write it into the staging area as a single blob,
/// returning its path relative to the final bundle dir (recorded in the
/// manifest, consumed by loadArtifacts). The blob is named from `configIndex`
/// so concurrent compile threads never collide without a shared counter.
/// Thread-safe: distinct `configIndex` values write distinct files.
mlir::FailureOr<std::string> writeArtifactBlob(llvm::StringRef dir,
                                               size_t configIndex,
                                               llvm::StringRef hsacoBinary);

/// Write the (compressed) manifest describing the whole perf-config space, then
/// atomically rename `<dir>.tmp` onto `dir`. For each Success config, the
/// manifest records `result.blobPath` (the relative path returned by
/// writeArtifactBlob), so blobs must already be on disk. Non-Success configs
/// need no blob.
mlir::LogicalResult
finalizeArtifactBundle(llvm::StringRef dir,
                       llvm::ArrayRef<CompilationResult> results,
                       const BufferLayout &layout, llvm::StringRef arch,
                       int64_t numCUs, int64_t numChiplets);

/// Inverse of the bundle-writing routines above, but LAZY: parse the manifest
/// into per-config metadata + blob *paths* only. HSACO blobs are decompressed
/// just-in-time by the caller, so peak memory stays below the in-process path
/// for large spaces.
mlir::LogicalResult loadArtifacts(llvm::StringRef dir,
                                  std::vector<CompilationResult> &outResults,
                                  BufferLayout &outLayout,
                                  ManifestInfo &outInfo);

} // namespace rocmlir::tuningdriver

#endif // ROCMLIR_TUNING_DRIVER_ARTIFACT_IO_H
