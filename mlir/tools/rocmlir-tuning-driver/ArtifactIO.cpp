//===- ArtifactIO.cpp - Kernel-bundle types, format, and IO ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArtifactIO.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>
#include <vector>

#include <hip/hip_runtime.h>

using namespace mlir;
using namespace rocmlir::tuningdriver;
using llvm::ArrayRef;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::StringRef;

namespace {

// A bundle contains one manifest or one config's HSACO per compressed file.
// Both are normally at most a few MiB; leave ample headroom while preventing a
// corrupt size header from requesting an effectively unbounded allocation.
constexpr uint64_t kMaxUncompressedArtifactSize = 256ULL * 1024 * 1024;

void appendU64LE(std::string &s, uint64_t v) {
  for (int i = 0; i < 8; ++i)
    s.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

uint64_t readU64LE(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

// The staging directory a bundle is assembled in before being atomically
// renamed onto its final `dir`. All bundle-writing routines derive it the same
// way so they agree on where blobs and the manifest live.
SmallString<256> stagingDirFor(StringRef dir) {
  SmallString<256> tmp(dir);
  tmp += ".tmp";
  return tmp;
}

// Manifest spelling of a CompilationStatus. The switch is exhaustive (no
// `default:`) so adding an enumerator is a compile-time error here rather than
// a config silently recorded under the wrong status.
StringRef statusToString(CompilationStatus status) {
  switch (status) {
  case CompilationStatus::NotApplicable:
    return "not_applicable";
  case CompilationStatus::CompilationFailed:
    return "compilation_failed";
  case CompilationStatus::TimedOut:
    return "timed_out";
  case CompilationStatus::Success:
    return "success";
  }
  llvm_unreachable("unhandled CompilationStatus");
}

// Inverse of statusToString. std::nullopt means the manifest carried a status
// this build does not know, which callers must reject rather than guess at.
std::optional<CompilationStatus> statusFromString(StringRef status) {
  return llvm::StringSwitch<std::optional<CompilationStatus>>(status)
      .Case("not_applicable", CompilationStatus::NotApplicable)
      .Case("compilation_failed", CompilationStatus::CompilationFailed)
      .Case("timed_out", CompilationStatus::TimedOut)
      .Case("success", CompilationStatus::Success)
      .Default(std::nullopt);
}

} // namespace

std::string rocmlir::tuningdriver::compressFramed(StringRef raw) {
  SmallVector<uint8_t> compressed;
  llvm::compression::zstd::compress(
      ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(raw.data()),
                        raw.size()),
      compressed);
  std::string out;
  // 8 bytes for the uint64_t little-endian size header (see appendU64LE),
  // followed by the zstd payload.
  out.reserve(8 + compressed.size());
  appendU64LE(out, raw.size());
  out.append(reinterpret_cast<const char *>(compressed.data()),
             compressed.size());
  return out;
}

FailureOr<std::string>
rocmlir::tuningdriver::decompressFramed(StringRef framed) {
  // `framed` is raw binary, and StringRef::size() is a plain byte count (no
  // Unicode decoding), so this checks there are at least 8 bytes for the
  // uint64_t little-endian size header before reading it.
  if (framed.size() < 8) {
    llvm::errs() << "compressed blob too small to contain a size header\n";
    return failure();
  }
  uint64_t uncompressedSize =
      readU64LE(reinterpret_cast<const uint8_t *>(framed.data()));
  if (uncompressedSize > kMaxUncompressedArtifactSize) {
    llvm::errs() << "compressed blob declares an uncompressed size of "
                 << uncompressedSize << " bytes, exceeding the "
                 << kMaxUncompressedArtifactSize << "-byte limit\n";
    return failure();
  }
  StringRef payload = framed.drop_front(8);
  SmallVector<uint8_t> out;
  if (llvm::Error e = llvm::compression::zstd::decompress(
          ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(payload.data()),
                            payload.size()),
          out, uncompressedSize)) {
    llvm::errs() << "zstd decompression failed: "
                 << llvm::toString(std::move(e)) << "\n";
    return failure();
  }
  return std::string(reinterpret_cast<const char *>(out.data()), out.size());
}

LogicalResult rocmlir::tuningdriver::writeFileContents(StringRef path,
                                                       StringRef contents) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "failed to open " << path
                 << " for writing: " << ec.message() << "\n";
    return failure();
  }
  os << contents;
  os.flush();
  if (os.has_error()) {
    llvm::errs() << "failed to write " << path << ": " << os.error().message()
                 << "\n";
    return failure();
  }
  return success();
}

FailureOr<std::string> rocmlir::tuningdriver::readFileContents(StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buf =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buf) {
    llvm::errs() << "failed to read " << path << ": "
                 << buf.getError().message() << "\n";
    return failure();
  }
  return (*buf)->getBuffer().str();
}

LogicalResult rocmlir::tuningdriver::beginArtifactBundle(StringRef dir) {
  if (!llvm::compression::zstd::isAvailable()) {
    llvm::errs() << "zstd is required for --compile-only but is not available "
                    "in this build\n";
    return failure();
  }

  SmallString<256> tmpDir = stagingDirFor(dir);
  if (auto ec =
          llvm::sys::fs::remove_directories(tmpDir, /*IgnoreErrors=*/false)) {
    llvm::errs() << "failed to clear stale temp dir " << tmpDir << ": "
                 << ec.message() << "\n";
    return failure();
  }
  SmallString<256> tmpConfigs(tmpDir);
  llvm::sys::path::append(tmpConfigs, "configs");
  if (auto ec = llvm::sys::fs::create_directories(tmpConfigs)) {
    llvm::errs() << "failed to create " << tmpConfigs << ": " << ec.message()
                 << "\n";
    return failure();
  }
  return success();
}

FailureOr<std::string>
rocmlir::tuningdriver::writeArtifactBlob(StringRef dir, size_t configIndex,
                                         StringRef hsacoBinary) {
  // "configs/" (8) + ".hsaco.z" (8) + NUL (1) = 17 fixed bytes, leaving 23 for
  // the index. A 64-bit size_t is at most 20 digits, so 40 always fits (and
  // snprintf is length-bounded regardless). Naming the blob after configIndex
  // (rather than a shared success counter) lets parallel compile threads write
  // concurrently without coordination.
  char nameBuf[40];
  std::snprintf(nameBuf, sizeof(nameBuf), "configs/%06zu.hsaco.z", configIndex);
  std::string blobRel(nameBuf);

  SmallString<256> blobPath = stagingDirFor(dir);
  llvm::sys::path::append(blobPath, blobRel);
  if (failed(writeFileContents(blobPath, compressFramed(hsacoBinary))))
    return failure();
  return blobRel;
}

LogicalResult rocmlir::tuningdriver::finalizeArtifactBundle(
    StringRef dir, ArrayRef<CompilationResult> results,
    const BufferLayout &layout, StringRef arch, int64_t numCUs,
    int64_t numChiplets) {
  SmallString<256> tmpDir = stagingDirFor(dir);

  std::string manifestStr;
  {
    llvm::raw_string_ostream os(manifestStr);
    llvm::json::OStream J(os);
    J.object([&] {
      J.attribute("hipVersion", static_cast<int64_t>(HIP_VERSION));
      J.attribute("arch", arch);
      J.attribute("numCUs", numCUs);
      J.attribute("numChiplets", numChiplets);
      J.attribute("numConfigs", static_cast<int64_t>(results.size()));
      J.attributeArray("buffers", [&] {
        for (size_t bi = 0; bi < layout.byteLengths.size(); ++bi) {
          J.object([&] {
            J.attribute("bytes", static_cast<int64_t>(layout.byteLengths[bi]));
            J.attribute("dataType", static_cast<int64_t>(static_cast<uint32_t>(
                                        layout.dataTypes[bi])));
          });
        }
      });
      J.attributeArray("configs", [&] {
        for (size_t i = 0; i < results.size(); ++i) {
          const CompilationResult &r = results[i];
          J.object([&] {
            J.attribute("perfConfig", StringRef(r.perfConfig));
            J.attribute("status", statusToString(r.status));
            // Only Success produced a code object; every other status has
            // neither a blob nor kernel metadata to record.
            if (r.status == CompilationStatus::Success) {
              // Relative blob path recorded by writeArtifactBlob; the blob is
              // already staged on disk.
              J.attribute("blob", StringRef(r.blobPath));
              J.attributeArray("kernels", [&] {
                for (const rock::KernelInfo &k : r.kernels) {
                  J.object([&] {
                    J.attribute("name", k.name);
                    J.attribute("gridSize", k.gridSize);
                    J.attribute("blockSize", k.blockSize);
                    J.attribute("clusterSize", k.clusterSize);
                  });
                }
              });
            }
          });
        }
      });
    });
  }

  SmallString<256> outManifest(tmpDir);
  llvm::sys::path::append(outManifest, "manifest.json.z");
  std::string compressedManifest = compressFramed(manifestStr);
  if (failed(writeFileContents(outManifest, StringRef(compressedManifest))))
    return failure();

  // Keep the previous bundle recoverable until the staged bundle is installed.
  SmallString<256> oldDir(dir);
  oldDir += ".old";
  bool movedExistingBundle = false;
  if (llvm::sys::fs::exists(dir)) {
    if (llvm::sys::fs::exists(oldDir)) {
      if (auto ec = llvm::sys::fs::remove_directories(oldDir,
                                                      /*IgnoreErrors=*/false)) {
        llvm::errs() << "failed to clear stale backup " << oldDir << ": "
                     << ec.message() << "\n";
        return failure();
      }
    }
    if (auto ec = llvm::sys::fs::rename(dir, oldDir)) {
      llvm::errs() << "failed to preserve existing bundle " << dir << " at "
                   << oldDir << ": " << ec.message() << "\n";
      return failure();
    }
    movedExistingBundle = true;
  }

  if (auto ec = llvm::sys::fs::rename(tmpDir, dir)) {
    llvm::errs() << "failed to install staged bundle " << tmpDir << " at "
                 << dir << ": " << ec.message()
                 << "; staged bundle remains recoverable at " << tmpDir << "\n";
    if (movedExistingBundle) {
      if (auto restoreEc = llvm::sys::fs::rename(oldDir, dir)) {
        llvm::errs() << "failed to restore previous bundle from " << oldDir
                     << " to " << dir << ": " << restoreEc.message()
                     << "; previous bundle remains recoverable at " << oldDir
                     << "\n";
      }
    }
    return failure();
  }

  if (llvm::sys::fs::exists(oldDir)) {
    if (auto ec =
            llvm::sys::fs::remove_directories(oldDir, /*IgnoreErrors=*/false)) {
      llvm::errs() << "warning: installed bundle at " << dir
                   << " but failed to remove backup " << oldDir << ": "
                   << ec.message() << "\n";
    }
  }
  return success();
}

LogicalResult rocmlir::tuningdriver::loadArtifacts(
    StringRef dir, std::vector<CompilationResult> &outResults,
    BufferLayout &outLayout, ManifestInfo &outInfo) {
  if (!llvm::compression::zstd::isAvailable()) {
    llvm::errs() << "zstd is required for --benchmark-artifacts but is not "
                    "available in this build\n";
    return failure();
  }

  SmallString<256> manifestZ(dir);
  llvm::sys::path::append(manifestZ, "manifest.json.z");
  if (!llvm::sys::fs::exists(manifestZ)) {
    llvm::errs() << "no manifest.json.z found in " << dir << "\n";
    return failure();
  }
  FailureOr<std::string> framed = readFileContents(manifestZ);
  if (failed(framed))
    return failure();
  FailureOr<std::string> manifestTextOr = decompressFramed(*framed);
  if (failed(manifestTextOr))
    return failure();
  std::string manifestText = std::move(*manifestTextOr);

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(manifestText);
  if (!parsed) {
    llvm::errs() << "failed to parse manifest JSON: "
                 << llvm::toString(parsed.takeError()) << "\n";
    return failure();
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    llvm::errs() << "manifest JSON is not an object\n";
    return failure();
  }

  bool ok = true;
  auto reqStr = [&](StringRef key) -> std::string {
    if (auto v = root->getString(key))
      return v->str();
    llvm::errs() << "manifest missing string field '" << key << "'\n";
    ok = false;
    return "";
  };
  auto reqInt = [&](StringRef key) -> int64_t {
    if (auto v = root->getInteger(key))
      return *v;
    llvm::errs() << "manifest missing integer field '" << key << "'\n";
    ok = false;
    return 0;
  };
  outInfo = ManifestInfo{};
  outInfo.hipVersion = static_cast<uint64_t>(reqInt("hipVersion"));
  outInfo.arch = reqStr("arch");
  outInfo.numCUs = reqInt("numCUs");
  outInfo.numChiplets = reqInt("numChiplets");
  outInfo.numConfigs = reqInt("numConfigs");
  if (!ok)
    return failure();
  llvm::json::Array *buffers = root->getArray("buffers");
  if (!buffers) {
    llvm::errs() << "manifest missing 'buffers' array\n";
    return failure();
  }
  for (llvm::json::Value &bv : *buffers) {
    llvm::json::Object *bo = bv.getAsObject();
    std::optional<int64_t> bytes = bo ? bo->getInteger("bytes") : std::nullopt;
    std::optional<int64_t> dt = bo ? bo->getInteger("dataType") : std::nullopt;
    if (!bytes || !dt) {
      llvm::errs() << "manifest 'buffers' entry missing bytes/dataType\n";
      return failure();
    }
    if (*bytes < 0 ||
        static_cast<uint64_t>(*bytes) > std::numeric_limits<size_t>::max()) {
      llvm::errs() << "manifest 'buffers' entry has invalid bytes value "
                   << *bytes << "\n";
      return failure();
    }
    if (*dt < 0 || *dt >= static_cast<int64_t>(benchmark::kNumDataTypes) ||
        *dt == static_cast<int64_t>(benchmark::DataType::UNKNOWN)) {
      llvm::errs() << "manifest 'buffers' entry has invalid dataType value "
                   << *dt << "\n";
      return failure();
    }
    outLayout.byteLengths.push_back(static_cast<size_t>(*bytes));
    outLayout.dataTypes.push_back(static_cast<benchmark::DataType>(*dt));
  }

  llvm::json::Array *configs = root->getArray("configs");
  if (!configs) {
    llvm::errs() << "manifest missing 'configs' array\n";
    return failure();
  }
  for (llvm::json::Value &cv : *configs) {
    llvm::json::Object *co = cv.getAsObject();
    std::optional<StringRef> pc =
        co ? co->getString("perfConfig") : std::nullopt;
    std::optional<StringRef> st = co ? co->getString("status") : std::nullopt;
    if (!pc || !st) {
      llvm::errs() << "manifest 'configs' entry missing perfConfig/status\n";
      return failure();
    }
    std::optional<CompilationStatus> status = statusFromString(*st);
    if (!status) {
      llvm::errs() << "unknown config status '" << *st << "' for config '"
                   << *pc << "'\n";
      return failure();
    }
    CompilationResult r;
    r.perfConfig = *pc;
    r.status = *status;
    if (r.status == CompilationStatus::Success) {
      std::optional<StringRef> blob = co->getString("blob");
      if (!blob) {
        llvm::errs() << "success config '" << *pc << "' missing 'blob'\n";
        return failure();
      }
      SmallString<256> bp(dir);
      llvm::sys::path::append(bp, *blob);
      r.blobPath.assign(bp.begin(), bp.end());
      llvm::json::Array *kernels = co->getArray("kernels");
      if (!kernels) {
        llvm::errs() << "success config '" << *pc << "' missing 'kernels'\n";
        return failure();
      }
      for (llvm::json::Value &kv : *kernels) {
        llvm::json::Object *ko = kv.getAsObject();
        std::optional<StringRef> name =
            ko ? ko->getString("name") : std::nullopt;
        std::optional<int64_t> grid =
            ko ? ko->getInteger("gridSize") : std::nullopt;
        std::optional<int64_t> block =
            ko ? ko->getInteger("blockSize") : std::nullopt;
        std::optional<int64_t> cluster =
            ko ? ko->getInteger("clusterSize") : std::nullopt;
        if (!name || !grid || !block || !cluster) {
          llvm::errs() << "kernel entry for config '" << *pc
                       << "' is missing a field\n";
          return failure();
        }
        rock::KernelInfo ki;
        ki.name = name->str();
        ki.gridSize = *grid;
        ki.blockSize = *block;
        ki.clusterSize = *cluster;
        r.kernels.push_back(std::move(ki));
      }
    }
    outResults.push_back(std::move(r));
  }
  return success();
}
